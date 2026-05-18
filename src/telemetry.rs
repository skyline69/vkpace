//! Lightweight telemetry.
//!
//! Two collection points:
//!
//! 1. **Counters** — `frames`, `submits`, `presents`, `injections` —
//!    `AtomicU64`, incremented from hot paths. Lock-free.
//! 2. **Optional unix socket** — when `LOW_LATENCY_LAYER_TELEMETRY_SOCKET`
//!    is set, a background thread accepts a single connection at a time and
//!    streams newline-delimited JSON records (one per present) to the peer.
//!    Recording is bounded by a small ring buffer; if the consumer can't
//!    keep up, oldest records drop.
//!
//! The hot path never blocks on the socket: it pushes into the ring under a
//! single mutex acquisition and notifies a Condvar.

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::io::Write;
use std::os::unix::net::UnixListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};

const RING_CAPACITY: usize = 1024;

#[derive(Default)]
pub struct Counters {
    pub frames: AtomicU64,
    pub submits: AtomicU64,
    pub presents: AtomicU64,
    pub injections: AtomicU64,
    pub acquires: AtomicU64,
}

impl Counters {
    /// One call to `vkQueueSubmit[2]` regardless of whether any of its
    /// submits was injected by us. `injected` is how many of the contained
    /// submits we wrapped with timestamps.
    #[inline]
    pub fn record_submit_call(&self, submit_count: u32, injected: u32) {
        self.submits
            .fetch_add(submit_count as u64, Ordering::Relaxed);
        self.injections
            .fetch_add(injected as u64, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_present(&self) {
        self.presents.fetch_add(1, Ordering::Relaxed);
        self.frames.fetch_add(1, Ordering::Relaxed);
    }
}

/// One record per present. `host_ns` is `CLOCK_MONOTONIC` at the point we
/// returned from the downstream `vkQueuePresentKHR`.
#[derive(Clone, Copy, Default)]
pub struct FrameRecord {
    pub host_ns: u64,
    pub frame_index: u64,
    pub queue_id: u64,
}

struct SocketShared {
    ring: Mutex<VecDeque<FrameRecord>>,
    cv: Condvar,
    stop: AtomicBool,
    socket_path: String,
}

pub struct Telemetry {
    pub counters: Arc<Counters>,
    socket: Option<Arc<SocketShared>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    stats_worker: Mutex<Option<JoinHandle<()>>>,
    stats_stop: Arc<AtomicBool>,
    stats_wake: Arc<(Mutex<()>, Condvar)>,
}

impl Telemetry {
    pub fn new(socket_path: Option<String>, stats_interval_s: u64) -> Self {
        let socket = socket_path.map(|p| {
            Arc::new(SocketShared {
                ring: Mutex::new(VecDeque::with_capacity(RING_CAPACITY)),
                cv: Condvar::new(),
                stop: AtomicBool::new(false),
                socket_path: p,
            })
        });

        let worker = if let Some(s) = socket.as_ref() {
            let s = s.clone();
            Some(
                thread::Builder::new()
                    .name("vkpace-telemetry".into())
                    .spawn(move || run_socket(s))
                    .expect("spawn telemetry"),
            )
        } else {
            None
        };

        let counters = Arc::new(Counters::default());
        let stats_stop = Arc::new(AtomicBool::new(false));
        let stats_wake = Arc::new((Mutex::new(()), Condvar::new()));
        let stats_worker = if stats_interval_s > 0 {
            let counters = counters.clone();
            let stop = stats_stop.clone();
            let wake = stats_wake.clone();
            Some(
                thread::Builder::new()
                    .name("vkpace-stats".into())
                    .spawn(move || run_stats(counters, stop, wake, stats_interval_s))
                    .expect("spawn stats"),
            )
        } else {
            None
        };

        Self {
            counters,
            socket,
            worker: Mutex::new(worker),
            stats_worker: Mutex::new(stats_worker),
            stats_stop,
            stats_wake,
        }
    }

    pub fn push_record(&self, rec: FrameRecord) {
        let Some(s) = self.socket.as_ref() else {
            return;
        };
        let mut ring = s.ring.lock();
        if ring.len() == RING_CAPACITY {
            ring.pop_front();
        }
        ring.push_back(rec);
        drop(ring);
        s.cv.notify_one();
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        if let Some(s) = self.socket.as_ref() {
            s.stop.store(true, Ordering::SeqCst);
            s.cv.notify_all();
            let _ = std::fs::remove_file(&s.socket_path);
        }
        self.stats_stop.store(true, Ordering::SeqCst);
        self.stats_wake.1.notify_all();
        if let Some(j) = self.worker.lock().take() {
            let _ = j.join();
        }
        if let Some(j) = self.stats_worker.lock().take() {
            let _ = j.join();
        }
    }
}

fn run_stats(
    counters: Arc<Counters>,
    stop: Arc<AtomicBool>,
    wake: Arc<(Mutex<()>, Condvar)>,
    interval_s: u64,
) {
    let interval = std::time::Duration::from_secs(interval_s);
    let mut last_frames = 0u64;
    while !stop.load(Ordering::Acquire) {
        let mut g = wake.0.lock();
        let _ = wake.1.wait_for(&mut g, interval);
        drop(g);
        if stop.load(Ordering::Acquire) {
            break;
        }
        let frames = counters.frames.load(Ordering::Relaxed);
        let submits = counters.submits.load(Ordering::Relaxed);
        let injections = counters.injections.load(Ordering::Relaxed);
        let acquires = counters.acquires.load(Ordering::Relaxed);
        let frames_delta = frames - last_frames;
        let fps = frames_delta as f64 / interval_s as f64;
        last_frames = frames;
        tracing::info!(
            frames,
            submits,
            injections,
            acquires,
            recent_fps = fps,
            "telemetry snapshot"
        );
    }
}

fn run_socket(shared: Arc<SocketShared>) {
    let _ = std::fs::remove_file(&shared.socket_path);
    let listener = match UnixListener::bind(&shared.socket_path) {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!(?e, path = %shared.socket_path, "telemetry: bind failed");
            return;
        }
    };
    listener.set_nonblocking(true).ok();
    tracing::info!(path = %shared.socket_path, "telemetry: listening");

    while !shared.stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((mut conn, _addr)) => {
                tracing::debug!("telemetry: client connected");
                let _ = conn.set_nonblocking(false);
                serve_client(&shared, &mut conn);
                tracing::debug!("telemetry: client disconnected");
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                tracing::warn!(?e, "telemetry: accept failed");
                break;
            }
        }
    }
}

fn serve_client(shared: &Arc<SocketShared>, conn: &mut std::os::unix::net::UnixStream) {
    let mut buf = String::with_capacity(256);
    loop {
        if shared.stop.load(Ordering::Acquire) {
            return;
        }
        let mut ring = shared.ring.lock();
        while ring.is_empty() && !shared.stop.load(Ordering::Acquire) {
            shared.cv.wait(&mut ring);
        }
        if shared.stop.load(Ordering::Acquire) {
            return;
        }
        let drained: Vec<FrameRecord> = ring.drain(..).collect();
        drop(ring);

        for rec in drained {
            buf.clear();
            // Minimal hand-rolled JSON; no serde to keep the dependency tree
            // small.
            use std::fmt::Write as _;
            let _ = writeln!(
                buf,
                r#"{{"ts":{},"frame":{},"queue":"0x{:x}"}}"#,
                rec.host_ns, rec.frame_index, rec.queue_id
            );
            if conn.write_all(buf.as_bytes()).is_err() {
                return;
            }
        }
    }
}
