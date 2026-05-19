//! Shared state between the socket reader thread and the egui UI.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;

/// Match the layer's `FrameRecord` shape (layer crate, `src/telemetry.rs:65-74`).
#[derive(Clone, Copy, Default, Debug)]
pub struct Record {
    pub ts_ns: u64,
    pub frame: u64,
    pub queue: u64,
    pub pid: u64,
    pub latency_us: u64,
}

/// Bounded ring of recent records. ~64 KiB at full capacity (4096 × 40 B);
/// trivially cheap to clone for the UI snapshot. A 60 s window at 240 fps
/// is 14 400 records, but the UI bins into 100 ms buckets so we only need
/// enough samples to keep last-second percentiles stable — 4096 covers
/// that with margin.
pub const RING_CAPACITY: usize = 4096;

pub struct SharedState {
    pub ring: Mutex<VecDeque<Record>>,
    pub stop: AtomicBool,
    pub connected: AtomicBool,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            ring: Mutex::new(VecDeque::with_capacity(RING_CAPACITY)),
            stop: AtomicBool::new(false),
            connected: AtomicBool::new(false),
        }
    }

    pub fn push(&self, rec: Record) {
        let mut ring = self.ring.lock();
        if ring.len() == RING_CAPACITY {
            ring.pop_front();
        }
        ring.push_back(rec);
    }

    /// Copy records with `ts_ns >= cutoff` into a fresh `Vec` for the UI
    /// thread. Cheap (one alloc, ≤ RING_CAPACITY copies) and means the UI
    /// holds no lock across egui rendering.
    pub fn snapshot_since(&self, cutoff_ns: u64) -> Vec<Record> {
        let ring = self.ring.lock();
        ring.iter()
            .filter(|r| r.ts_ns >= cutoff_ns)
            .copied()
            .collect()
    }

    /// Latest record's timestamp (game-process monotonic ns). Used by the
    /// UI as "now" so we don't have to assume the HUD's local clock is in
    /// the same domain as the game's `CLOCK_MONOTONIC`. Returns `None`
    /// when the ring is empty.
    pub fn latest_ts(&self) -> Option<u64> {
        self.ring.lock().back().map(|r| r.ts_ns)
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}
