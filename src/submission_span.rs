//! Frame's-worth of submissions for a queue, collapsed to just first + last
//! timestamps. The first handle is always the start; the second (optional)
//! is the end. Splitting like this keeps memory bounded regardless of how
//! many submissions a frame contains.

use std::sync::Arc;
use std::time::Duration;

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

    /// Blocks until the GPU has flushed the trailing timestamp. Unbounded —
    /// only use when a deadline isn't possible (e.g. teardown).
    pub fn await_completed(&self, clock: &DeviceClock) {
        let _ = self.tail.as_ref().unwrap_or(&self.head).await_end_ns(clock);
    }

    /// Wait at most `budget` for the trailing timestamp. Returns `true` if
    /// the work completed within the budget, `false` if the deadline expired
    /// first. Caller decides whether to proceed regardless — for Reflex
    /// semaphore pacing, signalling on timeout is preferable to stalling the
    /// game-side wait.
    pub fn await_completed_until(&self, deadline_host_ns: u64) -> bool {
        loop {
            if self.has_completed() {
                return true;
            }
            if DeviceClock::now() >= deadline_host_ns {
                return false;
            }
            // Yield first; for very tight budgets the spin-loop falls into
            // pause-equivalent immediately.
            std::thread::yield_now();
        }
    }
}

/// Wait for every span in `spans` to complete, sharing a single deadline.
/// Returns the number of spans that completed within the budget.
pub fn await_all_until(spans: &[SubmissionSpan], budget: Duration) -> usize {
    let deadline = DeviceClock::now() + budget.as_nanos() as u64;
    let mut done = 0;
    for span in spans {
        if span.await_completed_until(deadline) {
            done += 1;
        } else {
            break;
        }
    }
    done
}
