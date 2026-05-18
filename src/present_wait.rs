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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

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

fn run(inner: Arc<Inner>) {
    let wait = inner
        .fns
        .wait_for_present_khr
        .expect("loader must have provided vkWaitForPresentKHR");
    loop {
        while let Some(item) = inner.queue.pop() {
            let now_before = DeviceClock::now();
            let r = unsafe { wait(inner.device, item.swapchain, item.present_id, WAIT_TIMEOUT_NS) };
            let now_after = DeviceClock::now();
            match r {
                vk::Result::SUCCESS => {
                    // Record on the markers history. host_ns → microseconds.
                    inner.markers.record_present_actual(item.present_id, now_after);
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
