//! Frame's-worth of submissions for a queue, collapsed to just first + last
//! timestamps. The first handle is always the start; the second (optional)
//! is the end. Splitting like this keeps memory bounded regardless of how
//! many submissions a frame contains.

use std::sync::Arc;

use crate::clock::DeviceClock;
use crate::timestamp_pool::Handle;

pub struct SubmissionSpan {
    head: Arc<Handle>,
    tail: Option<Arc<Handle>>,
}

impl SubmissionSpan {
    pub fn new(handle: Arc<Handle>) -> Self {
        Self {
            head: handle,
            tail: None,
        }
    }

    pub fn extend(&mut self, handle: Arc<Handle>) {
        self.tail = Some(handle);
    }

    /// Non-blocking probe: GPU work on the trailing timestamp is finished.
    pub fn has_completed(&self) -> bool {
        self.tail.as_ref().unwrap_or(&self.head).has_end()
    }

    /// Read host-monotonic-nanoseconds for the head start and the tail end
    /// (or head end if no tail). Only call after `has_completed()` returns
    /// `true` (or `await_completed_until` returned `true`); both reads then
    /// resolve without further blocking because the head's start timestamp
    /// was written before the tail's end on the same submission. Returns
    /// `(start_ns, end_ns)`, or `None` if either timestamp couldn't be
    /// retrieved.
    pub fn completion_times_ns(&self, clock: &DeviceClock) -> Option<(u64, u64)> {
        let start = self.head.await_start_ns(clock)?;
        let end = self.tail.as_ref().unwrap_or(&self.head).await_end_ns(clock)?;
        Some((start, end))
    }

    /// Blocks until the GPU has flushed the trailing timestamp. Unbounded —
    /// only use when a deadline isn't possible (e.g. teardown).
    pub fn await_completed(&self, clock: &DeviceClock) {
        let _ = self.tail.as_ref().unwrap_or(&self.head).await_end_ns(clock);
    }

    /// Wait at most until `deadline_host_ns` for the trailing timestamp.
    /// Returns `true` if the work completed in time, `false` if the deadline
    /// expired first. Caller decides whether to proceed regardless: for
    /// Reflex semaphore pacing we always signal, since stalling the
    /// game-side wait is worse than dropping one frame's pacing accuracy.
    ///
    /// Backoff: a few spin iterations (~ns), then yields (~µs), then short
    /// sleeps (~100 µs) capped at the remaining budget. This burns roughly
    /// 1/10th the CPU of `thread::yield_now()` in a tight loop while still
    /// reacting within a few hundred microseconds when the GPU finishes.
    pub fn await_completed_until(&self, deadline_host_ns: u64) -> bool {
        const SPIN_ITERS: u32 = 32;
        const YIELD_ITERS: u32 = 8;
        let mut iters: u32 = 0;
        loop {
            if self.has_completed() {
                return true;
            }
            let now = DeviceClock::now();
            if now >= deadline_host_ns {
                return false;
            }
            if iters < SPIN_ITERS {
                std::hint::spin_loop();
            } else if iters < SPIN_ITERS + YIELD_ITERS {
                std::thread::yield_now();
            } else {
                // Sleep for the smaller of 100 µs and the remaining budget.
                // Driver `GetQueryPoolResults` is itself a syscall, so sleep
                // granularity isn't a bottleneck.
                let remaining_ns = deadline_host_ns - now;
                let sleep_ns = remaining_ns.min(100_000);
                std::thread::sleep(std::time::Duration::from_nanos(sleep_ns));
            }
            iters = iters.saturating_add(1);
        }
    }
}

