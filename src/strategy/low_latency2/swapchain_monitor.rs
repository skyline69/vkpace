//! Worker that pairs `vkLatencySleepNV` semaphores with the GPU work that
//! must complete before signalling them. The C++ version used `jthread`;
//! we use `std::thread` with an `AtomicBool` stop flag + `parking_lot::Condvar`.

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::delay_controller::DelayController;
use crate::device::DeviceContext;
use crate::strategy::low_latency2::markers::MarkerHistory;
use crate::strategy::low_latency2::semaphore_signal::SemaphoreSignal;
use crate::submission_span::SubmissionSpan;

/// Default wait budget for accumulated `SubmissionSpan`s before we signal
/// the Reflex sleep semaphore anyway. Half a frame at 120 Hz. Bounded so
/// a stuck submission span (broken timestamp, driver bug, etc.) never
/// stalls the game-side `vkLatencySleepNV` wait.
const DEFAULT_WAIT_BUDGET_US: u64 = 4_000;

/// Submission spans tagged with the `present_id` they belong to. Multiple
/// `attach_work` calls between two `notify_semaphore` calls accumulate
/// here, each carrying its own present_id.
struct TaggedSpans {
    present_id: u64,
    spans: Vec<SubmissionSpan>,
}

struct PendingSignal {
    signal: SemaphoreSignal,
    groups: Vec<TaggedSpans>,
}

struct Shared {
    device: Arc<DeviceContext>,
    markers: Arc<MarkerHistory>,
    state: Mutex<State>,
    cv: Condvar,
    stop: AtomicBool,
}

struct State {
    pending_signals: VecDeque<PendingSignal>,
    pending_groups: Vec<TaggedSpans>,
    present_delay_ns: u64,
    requested: bool,
    delay_controller: DelayController,
}

pub struct SwapchainMonitor {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl SwapchainMonitor {
    pub fn new(device: Arc<DeviceContext>, markers: Arc<MarkerHistory>) -> Self {
        let decoupled = device.instance.is_simulation_decoupled;
        let shared = Arc::new(Shared {
            device,
            markers,
            state: Mutex::new(State {
                pending_signals: VecDeque::new(),
                pending_groups: Vec::new(),
                present_delay_ns: 0,
                requested: false,
                delay_controller: DelayController::new(decoupled),
            }),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
        });

        let worker = {
            let shared = shared.clone();
            thread::spawn(move || run(shared))
        };

        Self {
            shared,
            worker: Some(worker),
        }
    }

    pub fn update_params(&self, requested: bool, minimum_interval_us: u64) {
        let mut st = self.shared.state.lock();
        st.requested = requested;
        st.present_delay_ns = minimum_interval_us.saturating_mul(1_000);
    }

    pub fn notify_semaphore(&self, signal: SemaphoreSignal) {
        let mut st = self.shared.state.lock();
        if !st.requested {
            drop(st);
            signal.signal(&self.shared.device);
            return;
        }
        let groups = std::mem::take(&mut st.pending_groups);
        st.pending_signals
            .push_back(PendingSignal { signal, groups });
        drop(st);
        self.shared.cv.notify_one();
    }

    pub fn attach_work(&self, present_id: u64, spans: Vec<SubmissionSpan>) {
        let mut st = self.shared.state.lock();
        if !st.requested {
            return;
        }
        st.pending_groups.push(TaggedSpans { present_id, spans });
    }
}

impl Drop for SwapchainMonitor {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.shared.cv.notify_all();
        if let Some(j) = self.worker.take() {
            let _ = j.join();
        }
    }
}

fn run(shared: Arc<Shared>) {
    loop {
        let pending = {
            let mut st = shared.state.lock();
            while st.pending_signals.is_empty() && !shared.stop.load(Ordering::Acquire) {
                shared.cv.wait(&mut st);
            }
            if st.pending_signals.is_empty() && shared.stop.load(Ordering::Acquire) {
                return;
            }
            st.pending_signals.pop_front().unwrap()
        };

        // Drain mode: still signal everything queued, but skip wait/delay.
        if shared.stop.load(Ordering::Acquire) {
            pending.signal.signal(&shared.device);
            continue;
        }

        let app_delay = shared.state.lock().present_delay_ns;
        let layer_delay = shared.device.effective_min_delay_ns();
        let delay_ns = app_delay.max(layer_delay);

        // Bounded wait across every group. Shared deadline: if the first
        // group eats the whole budget, later groups time out immediately.
        let budget = wait_budget();
        let deadline = crate::clock::DeviceClock::now() + budget.as_nanos() as u64;
        let clock = shared.device.clock.as_deref();

        for group in &pending.groups {
            let mut group_done = 0usize;
            for span in &group.spans {
                if !span.await_completed_until(deadline) {
                    break;
                }
                group_done += 1;
                // Pull GPU timestamps and record them on the marker history.
                // Pick the *widest* span in the group (earliest start, latest
                // end) for this present_id — overlapping queues all
                // contribute, but reporting the envelope matches what the
                // Reflex SDK expects.
                if let Some(clock) = clock
                    && let Some((s, e)) = span.completion_times_ns(clock)
                {
                    record_widest(&shared.markers, group.present_id, s, e);
                }
            }
            if group_done < group.spans.len() {
                tracing::trace!(
                    present_id = group.present_id,
                    completed = group_done,
                    total = group.spans.len(),
                    budget_us = budget.as_micros() as u64,
                    "LL2: submission span wait timed out — signalling anyway"
                );
            }
        }

        shared.state.lock().delay_controller.delay(delay_ns);
        pending.signal.signal(&shared.device);
    }
}

/// Update the marker history for `present_id` to be the envelope of any
/// previously-recorded times and the new `(start, end)`. Lets multiple
/// queues each contribute and we end up with the earliest start + latest
/// end across all of them.
fn record_widest(markers: &MarkerHistory, present_id: u64, start_ns: u64, end_ns: u64) {
    markers.record_gpu_timing(present_id, start_ns, end_ns);
}

fn wait_budget() -> Duration {
    static CACHE: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let us = std::env::var("VKPACE_LL2_WAIT_BUDGET_US")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_WAIT_BUDGET_US);
        Duration::from_micros(us)
    })
}
