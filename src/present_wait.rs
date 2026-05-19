//! Background worker that calls `vkWaitForPresentKHR` to learn when a frame
//! actually reached the display. The result is the closest thing Vulkan
//! offers to a real "photons left the panel" timestamp without going
//! through window-system-specific protocols (wp_presentation_time, etc).
//!
//! Lifecycle:
//!
//! - Spawned once per `DeviceContext`, if the device exposes
//!   `VK_KHR_present_wait`.
//! - `enqueue(swapchain, present_id)` is called from the layer's
//!   `vkQueuePresentKHR` hook for every present with a non-zero
//!   `VkPresentIdKHR`.
//! - The worker pops items and blocks in `vkWaitForPresentKHR` with a
//!   ~50 ms timeout per call. On success we record the host-monotonic
//!   timestamp of return as the frame's `present_actual_us` in the
//!   [`MarkerHistory`] (filling `present_end_time_us` for the Reflex
//!   report).
//!
//! Timeouts and errors are logged at `trace` and discarded; one bad
//! frame must not stall subsequent reads.

use ash::vk;
use crossbeam_queue::SegQueue;
use parking_lot::{Condvar, Mutex};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::clock::DeviceClock;
use crate::dispatch::DeviceTable;
use crate::strategy::low_latency2::MarkerHistory;

/// 50 ms per wait. Vulkan spec allows the call to block until the present
/// completes; with VSync at 60 Hz we typically return in ≤16 ms. Cap so a
/// stuck driver can't pin the worker forever.
const WAIT_TIMEOUT_NS: u64 = 50_000_000;

#[derive(Clone, Copy)]
struct Pending {
    swapchain: vk::SwapchainKHR,
    present_id: u64,
}

struct Inner {
    device: vk::Device,
    fns: Arc<DeviceTable>,
    markers: Arc<MarkerHistory>,
    queue: SegQueue<Pending>,
    wake: (Mutex<()>, Condvar),
    stop: AtomicBool,
}

pub struct PresentWaitWorker {
    inner: Arc<Inner>,
    handle: Option<JoinHandle<()>>,
}

impl PresentWaitWorker {
    /// Returns `None` if `vkWaitForPresentKHR` isn't loaded. The Markers
    /// handle is the same one the LL2 strategy uses; we update its
    /// `present_end_time_us` entry per frame.
    pub fn new(
        device: vk::Device,
        fns: Arc<DeviceTable>,
        markers: Arc<MarkerHistory>,
    ) -> Option<Self> {
        fns.wait_for_present_khr?;
        let inner = Arc::new(Inner {
            device,
            fns,
            markers,
            queue: SegQueue::new(),
            wake: (Mutex::new(()), Condvar::new()),
            stop: AtomicBool::new(false),
        });
        let handle = {
            let inner = inner.clone();
            thread::Builder::new()
                .name("vkpace-present-wait".into())
                .spawn(move || run(inner))
                .ok()
        };
        Some(Self { inner, handle })
    }

    pub fn enqueue(&self, swapchain: vk::SwapchainKHR, present_id: u64) {
        if present_id == 0 {
            return;
        }
        self.inner.queue.push(Pending {
            swapchain,
            present_id,
        });
        let _g = self.inner.wake.0.lock();
        self.inner.wake.1.notify_one();
    }
}

impl Drop for PresentWaitWorker {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        self.inner.wake.1.notify_all();
        if let Some(j) = self.handle.take() {
            let _ = j.join();
        }
    }
}

// ─── VK_GOOGLE_display_timing fallback ─────────────────────────────────────
//
// When `VK_KHR_present_wait` is unavailable but the driver exposes
// `VK_GOOGLE_display_timing`, we can still surface per-present completion
// time. The GOOGLE entrypoint returns timings in present-submission order
// per swapchain, so the layer maintains a FIFO of `(KHR present_id)` per
// swapchain and pairs each polled timing with the head of the FIFO.
//
// Correlation is ordering-based; we never round-trip the KHR `present_id`
// through GOOGLE's 32-bit field. That avoids 32→64 wraparound surprises
// when the app's KHR present_ids exceed 32 bits.

/// Poll period between GOOGLE timing scans. ~5 ms keeps cost low at 240 Hz
/// (1-2 records per poll on average) without missing tails.
const GOOGLE_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Cap how many pending pids we hold per swapchain. Bounds memory if the
/// driver stops producing timing samples for some reason.
const GOOGLE_PENDING_CAP: usize = 256;

struct GoogleInner {
    device: vk::Device,
    fns: Arc<DeviceTable>,
    markers: Arc<MarkerHistory>,
    pending: Mutex<FxHashMap<vk::SwapchainKHR, VecDeque<u64>>>,
    wake: (Mutex<()>, Condvar),
    stop: AtomicBool,
    /// Clock domain of `actual_present_time`. Validated on first sample;
    /// `Unknown` → fall back to host-monotonic-at-poll-return (lossy but
    /// safe). The Vulkan spec leaves the domain implementation-defined; in
    /// practice every Linux driver returns CLOCK_MONOTONIC ns.
    domain: parking_lot::Mutex<GoogleClockDomain>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GoogleClockDomain {
    Unvalidated,
    Monotonic,
    Unknown,
}

pub struct GoogleTimingWorker {
    inner: Arc<GoogleInner>,
    handle: Option<JoinHandle<()>>,
}

impl GoogleTimingWorker {
    pub fn new(
        device: vk::Device,
        fns: Arc<DeviceTable>,
        markers: Arc<MarkerHistory>,
    ) -> Option<Self> {
        // Need the polling entrypoint to do anything useful.
        fns.get_past_presentation_timing_google?;
        let inner = Arc::new(GoogleInner {
            device,
            fns,
            markers,
            pending: Mutex::new(FxHashMap::default()),
            wake: (Mutex::new(()), Condvar::new()),
            stop: AtomicBool::new(false),
            domain: parking_lot::Mutex::new(GoogleClockDomain::Unvalidated),
        });
        let handle = {
            let inner = inner.clone();
            thread::Builder::new()
                .name("vkpace-google-timing".into())
                .spawn(move || run_google(inner))
                .ok()
        };
        Some(Self { inner, handle })
    }

    pub fn enqueue(&self, swapchain: vk::SwapchainKHR, present_id: u64) {
        if present_id == 0 {
            return;
        }
        let mut pending = self.inner.pending.lock();
        let fifo = pending.entry(swapchain).or_default();
        if fifo.len() >= GOOGLE_PENDING_CAP {
            fifo.pop_front();
        }
        fifo.push_back(present_id);
        drop(pending);
        let _g = self.inner.wake.0.lock();
        self.inner.wake.1.notify_one();
    }

    pub fn forget_swapchain(&self, swapchain: vk::SwapchainKHR) {
        self.inner.pending.lock().remove(&swapchain);
    }
}

impl Drop for GoogleTimingWorker {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        self.inner.wake.1.notify_all();
        if let Some(j) = self.handle.take() {
            let _ = j.join();
        }
    }
}

/// Sanity-check a candidate `actual_present_time` against the current host
/// monotonic clock. Values within ±5 s are assumed to be CLOCK_MONOTONIC ns
/// (the de-facto Linux convention); anything wilder probably means the
/// driver picked a device-clock or epoch-relative domain, and we'd rather
/// drop the sample than feed garbage into the latency overlay.
fn classify_google_domain(sample_ns: u64, now_ns: u64) -> GoogleClockDomain {
    const ACCEPT_SKEW_NS: i128 = 5_000_000_000;
    let skew = sample_ns as i128 - now_ns as i128;
    if skew.abs() <= ACCEPT_SKEW_NS {
        GoogleClockDomain::Monotonic
    } else {
        GoogleClockDomain::Unknown
    }
}

fn run_google(inner: Arc<GoogleInner>) {
    let poll = inner
        .fns
        .get_past_presentation_timing_google
        .expect("worker constructed only when PFN exists");

    // Reused per poll. Caps avoid spec-permitted unbounded responses.
    let mut scratch: Vec<vk::PastPresentationTimingGOOGLE> = Vec::with_capacity(32);
    let mut swapchain_scratch: Vec<vk::SwapchainKHR> = Vec::with_capacity(4);

    while !inner.stop.load(Ordering::Acquire) {
        // Snapshot swapchain list into a reused buffer (avoid per-tick alloc).
        swapchain_scratch.clear();
        swapchain_scratch.extend(inner.pending.lock().keys().copied());

        for sw in &swapchain_scratch {
            let sw = *sw;
            let mut count: u32 = 0;
            let r = unsafe { poll(inner.device, sw, &mut count, std::ptr::null_mut()) };
            if r != vk::Result::SUCCESS || count == 0 {
                continue;
            }
            // Bound the per-call alloc; we'll see the rest next tick.
            let want = count.min(64) as usize;
            scratch.clear();
            scratch.resize(want, vk::PastPresentationTimingGOOGLE::default());
            let mut got = want as u32;
            let r = unsafe { poll(inner.device, sw, &mut got, scratch.as_mut_ptr()) };
            if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
                continue;
            }
            scratch.truncate(got as usize);

            let now_ns = DeviceClock::now();
            let mut pending = inner.pending.lock();
            let Some(fifo) = pending.get_mut(&sw) else {
                continue;
            };
            // One-time domain validation against the first non-zero sample.
            {
                let mut domain = inner.domain.lock();
                if *domain == GoogleClockDomain::Unvalidated
                    && let Some(sample) = scratch.iter().find(|t| t.actual_present_time != 0)
                {
                    *domain = classify_google_domain(sample.actual_present_time, now_ns);
                    match *domain {
                        GoogleClockDomain::Monotonic => {
                            tracing::info!(
                                "GOOGLE_display_timing: actual_present_time domain looks like CLOCK_MONOTONIC"
                            );
                        }
                        GoogleClockDomain::Unknown => {
                            tracing::warn!(
                                sample = sample.actual_present_time,
                                now = now_ns,
                                "GOOGLE_display_timing: actual_present_time domain skewed; recording host-now as fallback"
                            );
                        }
                        GoogleClockDomain::Unvalidated => unreachable!(),
                    }
                }
            }
            let domain = *inner.domain.lock();
            for timing in &scratch {
                let Some(pid) = fifo.pop_front() else {
                    break;
                };
                // Pick host-ns according to the validated domain. We never
                // hand a wildly-skewed timestamp to the marker history.
                let host_ns = match domain {
                    GoogleClockDomain::Monotonic if timing.actual_present_time != 0 => {
                        timing.actual_present_time
                    }
                    _ => now_ns,
                };
                inner.markers.record_present_actual(pid, host_ns);
            }
        }

        // Sleep ~poll_interval but wake early if a new present arrives
        // (cuts latency on the very first frame after idle).
        let mut g = inner.wake.0.lock();
        if !inner.stop.load(Ordering::Acquire) {
            let _ = inner.wake.1.wait_for(&mut g, GOOGLE_POLL_INTERVAL);
        }
    }
}

fn run(inner: Arc<Inner>) {
    let wait = inner
        .fns
        .wait_for_present_khr
        .expect("loader must have provided vkWaitForPresentKHR");
    loop {
        while let Some(item) = inner.queue.pop() {
            let now_before = DeviceClock::now();
            let r = unsafe {
                wait(
                    inner.device,
                    item.swapchain,
                    item.present_id,
                    WAIT_TIMEOUT_NS,
                )
            };
            let now_after = DeviceClock::now();
            match r {
                vk::Result::SUCCESS => {
                    // Record on the markers history. host_ns → microseconds.
                    inner
                        .markers
                        .record_present_actual(item.present_id, now_after);
                    tracing::trace!(
                        present_id = item.present_id,
                        wait_us = (now_after - now_before) / 1_000,
                        "WaitForPresent completed"
                    );
                }
                vk::Result::TIMEOUT => {
                    tracing::trace!(
                        present_id = item.present_id,
                        "WaitForPresent timed out — dropping"
                    );
                }
                other => {
                    tracing::trace!(
                        present_id = item.present_id,
                        ?other,
                        "WaitForPresent returned non-success"
                    );
                }
            }
        }
        if inner.stop.load(Ordering::Acquire) {
            return;
        }
        let mut g = inner.wake.0.lock();
        if inner.queue.is_empty() && !inner.stop.load(Ordering::Acquire) {
            inner.wake.1.wait(&mut g);
        }
    }
}
