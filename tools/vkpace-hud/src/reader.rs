//! Background thread that owns the unix-socket connection to the layer.

use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::parse::parse_line;
use crate::state::SharedState;

/// Read timeout per `read_line`. Short enough that `state.stop` is observed
/// within ~half a second of being set; long enough that idle sockets
/// don't churn CPU.
const READ_TIMEOUT: Duration = Duration::from_millis(500);

/// Exponential reconnect backoff bounds. Starts at MIN, doubles to MAX
/// on each failure; resets to MIN on a successful read.
const BACKOFF_MIN: Duration = Duration::from_millis(100);
const BACKOFF_MAX: Duration = Duration::from_secs(5);

pub fn spawn(path: PathBuf, state: Arc<SharedState>) -> JoinHandle<()> {
    thread::Builder::new()
        .name("vkpace-hud-reader".into())
        .spawn(move || run(path, state))
        .expect("spawn reader thread")
}

fn run(path: PathBuf, state: Arc<SharedState>) {
    let mut backoff = BACKOFF_MIN;
    while !state.stop.load(Ordering::Acquire) {
        let stream = match UnixStream::connect(&path) {
            Ok(s) => s,
            Err(_) => {
                state.connected.store(false, Ordering::Release);
                sleep_interruptible(&state, backoff);
                backoff = (backoff * 2).min(BACKOFF_MAX);
                continue;
            }
        };
        if stream.set_read_timeout(Some(READ_TIMEOUT)).is_err() {
            // Some platforms reject zero-duration timeouts; ours is non-zero
            // so this should be unreachable. If it ever happens, fall back
            // to reconnect.
            continue;
        }
        state.connected.store(true, Ordering::Release);
        backoff = BACKOFF_MIN;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        loop {
            if state.stop.load(Ordering::Acquire) {
                return;
            }
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF — layer closed the socket.
                Ok(_) => {
                    if let Some(rec) = parse_line(&line) {
                        state.push(rec);
                    }
                }
                Err(e) => {
                    // Read timeout is the expected path that lets us re-check
                    // `stop`; anything else means the socket died.
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut
                    {
                        continue;
                    }
                    break;
                }
            }
        }
        state.connected.store(false, Ordering::Release);
    }
}

/// Sleep up to `dur` but wake immediately when `state.stop` is set. Avoids
/// holding shutdown for the full backoff window.
fn sleep_interruptible(state: &SharedState, dur: Duration) {
    const STEP: Duration = Duration::from_millis(100);
    let mut remaining = dur;
    while remaining > Duration::ZERO {
        if state.stop.load(Ordering::Acquire) {
            return;
        }
        let chunk = remaining.min(STEP);
        thread::sleep(chunk);
        remaining = remaining.saturating_sub(chunk);
    }
}
