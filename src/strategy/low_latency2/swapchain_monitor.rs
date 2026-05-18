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
use crate::strategy::low_latency2::semaphore_signal::SemaphoreSignal;
use crate::submission_span::{self, SubmissionSpan};

/// Default wait budget for accumulated `SubmissionSpan`s before we signal
/// the Reflex sleep semaphore anyway. Half a frame at 120 Hz. Bounded so
/// a stuck submission span (broken timestamp, driver bug, etc.) never
/// stalls the game-side `vkLatencySleepNV` wait.
const DEFAULT_WAIT_BUDGET_US: u64 = 4_000;

struct PendingSignal {
    signal: SemaphoreSignal,
    spans: Vec<SubmissionSpan>,
}

struct Shared {
    device: Arc<DeviceContext>,
    state: Mutex<State>,
    cv: Condvar,
    stop: AtomicBool,
}

struct State {
    pending_signals: VecDeque<PendingSignal>,
    pending_spans: Vec<SubmissionSpan>,
    present_delay_ns: u64,
    requested: bool,
    delay_controller: DelayController,
}

pub struct SwapchainMonitor {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl SwapchainMonitor {
    pub fn new(device: Arc<DeviceContext>) -> Self {
        let decoupled = device.instance.is_simulation_decoupled;
        let shared = Arc::new(Shared {
            device,
            state: Mutex::new(State {
                pending_signals: VecDeque::new(),
                pending_spans: Vec::new(),
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
        let spans = std::mem::take(&mut st.pending_spans);
        st.pending_signals
            .push_back(PendingSignal { signal, spans });
        drop(st);
        self.shared.cv.notify_one();
    }

    pub fn attach_work(&self, spans: Vec<SubmissionSpan>) {
        let mut st = self.shared.state.lock();
        if !st.requested {
            return;
        }
        st.pending_spans = spans;
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
        let layer_delay = shared.device.instance.config.fps_cap_min_delay_ns();
        let delay_ns = app_delay.max(layer_delay);

        // Bounded wait for the prior frame's GPU work. If a span doesn't
        // finish in time we signal anyway — the game-side waiter must not
        // stall longer than this budget regardless of GPU pathology.
        let budget = wait_budget();
        let total = pending.spans.len();
        let done = submission_span::await_all_until(&pending.spans, budget);
        if done < total {
            tracing::trace!(
                completed = done,
                total,
                budget_us = budget.as_micros() as u64,
                "LL2: submission span wait timed out — signalling anyway"
            );
        }

        shared.state.lock().delay_controller.delay(delay_ns);
        pending.signal.signal(&shared.device);
    }
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
