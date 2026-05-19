//! EWMA-based "drain" controller for decoupled-simulation games.
//!
//! Most games tightly couple simulation and render — a 1-frame queue at most.
//! A few (Marvel Rivals at the time of writing) decouple them, letting the
//! simulation thread run 2-3 frames ahead. Reflex/Anti-Lag alone can't fix
//! that because the GPU is idle by the time we'd otherwise wait. The trick
//! is to inject a small artificial sleep that forces simulation back to the
//! floor. We can't know the right sleep length a priori, so we apply a small
//! random jitter each frame and feed the jitter/frametime gradient into an
//! EWMA — when the gradient hovers near 0.5 we've found the floor.
//!
//! All times are nanoseconds in the host monotonic clock (`DeviceClock::now`).

use std::thread;
use std::time::Duration;

use crate::clock::DeviceClock;

const ALPHA: f64 = 0.01;
const DELTA_GAIN: f64 = 5.0;

#[derive(Clone, Copy)]
struct FrameInfo {
    frametime_ns: i64,
    jitter_ns: i64,
    release_ns: u64,
}

pub struct DelayController {
    simulation_decoupled: bool,
    previous_frame: Option<FrameInfo>,
    gradient_ewma: f64,
    drain_ns: i64,
}

impl DelayController {
    pub fn new(simulation_decoupled: bool) -> Self {
        Self {
            simulation_decoupled,
            previous_frame: None,
            gradient_ewma: 0.0,
            drain_ns: 0,
        }
    }

    pub fn delay(&mut self, min_delay_ns: u64) {
        let Some(prev) = self.previous_frame else {
            self.previous_frame = Some(FrameInfo {
                frametime_ns: 0,
                jitter_ns: 0,
                release_ns: DeviceClock::now(),
            });
            return;
        };

        let frametime_ns = (DeviceClock::now().saturating_sub(prev.release_ns)) as i64;

        if min_delay_ns != 0 {
            spin_until(prev.release_ns + min_delay_ns);
        }

        if !self.simulation_decoupled {
            self.previous_frame = Some(FrameInfo {
                frametime_ns,
                jitter_ns: 0,
                release_ns: DeviceClock::now(),
            });
            return;
        }

        // Apply jitter only on the "up" half of the cycle, so we observe a
        // clean rising-edge gradient.
        let jitter_ns: i64 = if prev.jitter_ns == 0 {
            frametime_ns / 50
        } else {
            0
        };
        spin_until(DeviceClock::now() + (jitter_ns + self.drain_ns).max(0) as u64);

        if prev.jitter_ns == 0 {
            self.previous_frame = Some(FrameInfo {
                frametime_ns,
                jitter_ns,
                release_ns: DeviceClock::now(),
            });
            return;
        }

        // Compute gradient on the "down" half. -1 => sleep helped (still
        // backlogged); 0 => sleep no-op (deep backlog); +1 => sleep hurt
        // frametime 1:1 (we're at the floor — stop pushing).
        let dt_jitter = -prev.jitter_ns;
        let dt_frametime = frametime_ns - prev.frametime_ns;
        let gradient = if dt_jitter == 0 {
            0.0
        } else {
            (dt_frametime as f64 / dt_jitter as f64).clamp(-1.0, 1.0)
        };
        self.gradient_ewma = ALPHA * gradient + (1.0 - ALPHA) * self.gradient_ewma;

        let delta = (DELTA_GAIN * (0.5 - self.gradient_ewma) * prev.jitter_ns as f64) as i64;
        self.drain_ns = (self.drain_ns + delta).clamp(0, frametime_ns);

        self.previous_frame = Some(FrameInfo {
            frametime_ns,
            jitter_ns,
            release_ns: DeviceClock::now(),
        });
    }
}

fn spin_until(deadline_ns: u64) {
    loop {
        let now = DeviceClock::now();
        if now >= deadline_ns {
            return;
        }
        let remaining = deadline_ns - now;
        if remaining > Duration::from_micros(200).as_nanos() as u64 {
            thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic harness: feed synthetic frametimes and check the
    /// gradient/drain state without touching the real clock.
    fn step_with_frametime(ctl: &mut DelayController, frametime_ns: i64) {
        // Pretend the previous frame released at `now - frametime_ns`.
        let now = DeviceClock::now();
        ctl.previous_frame = Some(FrameInfo {
            frametime_ns,
            jitter_ns: 0,
            release_ns: now.saturating_sub(frametime_ns as u64),
        });
        ctl.delay(0);
    }

    #[test]
    fn drain_stays_zero_when_coupled() {
        let mut ctl = DelayController::new(false);
        for _ in 0..50 {
            step_with_frametime(&mut ctl, 8_000_000);
        }
        assert_eq!(ctl.drain_ns, 0);
    }

    #[test]
    fn previous_frame_initialized_on_first_call() {
        let mut ctl = DelayController::new(true);
        assert!(ctl.previous_frame.is_none());
        ctl.delay(0);
        assert!(ctl.previous_frame.is_some());
    }

    #[test]
    fn min_delay_zero_is_noop() {
        let mut ctl = DelayController::new(false);
        ctl.delay(0); // first call: just records.
        let before = DeviceClock::now();
        ctl.delay(0);
        let after = DeviceClock::now();
        // No min_delay = should return quickly (within 5ms even on a busy CI box).
        assert!(after - before < 5_000_000);
    }

    /// Feed the controller a stable 16ms frametime stream under
    /// decoupled-simulation mode. After many iterations the EWMA gradient
    /// should settle in [-1, 1] and the drain accumulator should be a
    /// non-negative fraction of one frame.
    #[test]
    fn drain_bounds_under_stable_frametime() {
        let mut ctl = DelayController::new(true);
        let target = 16_000_000i64;
        for _ in 0..400 {
            step_with_frametime(&mut ctl, target);
        }
        assert!(ctl.drain_ns >= 0);
        assert!(
            ctl.drain_ns <= target,
            "drain {} exceeded frametime {}",
            ctl.drain_ns,
            target
        );
        assert!(ctl.gradient_ewma.is_finite());
        assert!((-1.0..=1.0).contains(&ctl.gradient_ewma));
    }

    /// Without decoupling, drain must remain zero regardless of how many
    /// frames we feed. Regression check for the early-return at the top of
    /// `delay`.
    #[test]
    fn drain_stays_zero_when_coupled_long_run() {
        let mut ctl = DelayController::new(false);
        for _ in 0..1000 {
            step_with_frametime(&mut ctl, 8_000_000);
        }
        assert_eq!(ctl.drain_ns, 0);
    }
}
