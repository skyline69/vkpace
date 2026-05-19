//! Rolling-window aggregates over a snapshot of telemetry records.
//!
//! Cheap by design: at 240 fps a 1-second window is ≤ 240 samples, and an
//! in-place `sort_unstable` over that is on the order of microseconds.
//! Order-statistic trees are nice for >100k samples; not needed here.

use crate::state::Record;

/// Aggregates computed each UI frame.
#[derive(Clone, Copy, Default)]
pub struct LiveStats {
    /// Frames-per-second over the window.
    pub fps: f64,
    /// Click-to-photon p50 / p99 / max (µs) over the window.
    pub p50_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
    /// Number of latency samples that contributed (frames without an
    /// `INPUT_SAMPLE` marker report 0 and are filtered out here).
    pub samples: usize,
}

/// Compute aggregates over records whose `ts_ns >= cutoff_ns`. `now_ns`
/// is the snapshot's most-recent timestamp (used for the FPS denominator).
pub fn live_stats(records: &[Record], now_ns: u64, window_ns: u64) -> LiveStats {
    if records.is_empty() || window_ns == 0 {
        return LiveStats::default();
    }
    let cutoff = now_ns.saturating_sub(window_ns);
    let window: Vec<&Record> = records.iter().filter(|r| r.ts_ns >= cutoff).collect();
    if window.is_empty() {
        return LiveStats::default();
    }
    let frame_count = window.len();
    let fps = frame_count as f64 * 1_000_000_000.0 / window_ns as f64;

    let mut latencies: Vec<u64> = window
        .iter()
        .map(|r| r.latency_us)
        .filter(|&v| v > 0)
        .collect();
    if latencies.is_empty() {
        return LiveStats {
            fps,
            ..LiveStats::default()
        };
    }
    latencies.sort_unstable();
    let p = |q: f64| -> u64 {
        let idx = ((latencies.len() as f64 - 1.0) * q).round() as usize;
        latencies[idx.min(latencies.len() - 1)]
    };
    LiveStats {
        fps,
        p50_us: p(0.50),
        p99_us: p(0.99),
        max_us: *latencies.last().unwrap(),
        samples: latencies.len(),
    }
}

/// Bin records into `bin_ns` buckets relative to `now_ns - window_ns`.
/// Returns `(t_seconds_offset, value)` pairs suitable for plotting; the
/// `value` is the per-bin aggregate produced by `agg`.
///
/// **Empty bins are skipped.** Emitting `0.0` for them yanks the line
/// down to zero between samples and looks like a frame-drop spike when
/// the actual cause is just a bin offset that fell between two records.
/// Skipping leaves a gap in the line, which `egui_plot` renders as a
/// continuous segment across the missing point — visually correct.
pub fn bin_records<F>(
    records: &[Record],
    now_ns: u64,
    window_ns: u64,
    bin_ns: u64,
    mut agg: F,
) -> Vec<[f64; 2]>
where
    F: FnMut(&[&Record]) -> f64,
{
    if bin_ns == 0 || records.is_empty() {
        return Vec::new();
    }
    let cutoff = now_ns.saturating_sub(window_ns);
    let bin_count = (window_ns / bin_ns).max(1) as usize;
    let mut buckets: Vec<Vec<&Record>> = vec![Vec::new(); bin_count];
    for r in records {
        if r.ts_ns < cutoff {
            continue;
        }
        let idx = ((r.ts_ns - cutoff) / bin_ns) as usize;
        if idx < bin_count {
            buckets[idx].push(r);
        }
    }
    buckets
        .into_iter()
        .enumerate()
        .filter(|(_, b)| !b.is_empty())
        .map(|(i, bucket)| {
            // X axis in seconds, relative to `now`. Bin 0 is the oldest in
            // the window, bin (n-1) is the newest, so subtract from window.
            let t = -((window_ns as f64 - i as f64 * bin_ns as f64) / 1_000_000_000.0);
            [t, agg(&bucket)]
        })
        .collect()
}

/// Gap detector: returns `(frame_index, gap_size)` for any present where
/// the frame counter advanced by more than 1 in the window.
pub fn frame_gaps(records: &[Record]) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for w in records.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if b.frame > a.frame + 1 {
            out.push([b.frame as f64, (b.frame - a.frame - 1) as f64]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(ts_ns: u64, frame: u64, latency_us: u64) -> Record {
        Record {
            ts_ns,
            frame,
            queue: 0,
            pid: frame,
            latency_us,
        }
    }

    #[test]
    fn fps_over_known_window() {
        // 60 records spread over exactly 1s = 60 fps.
        let records: Vec<Record> = (0u64..60).map(|i| rec(i * 16_666_666, i, 0)).collect();
        let stats = live_stats(&records, 60 * 16_666_666, 1_000_000_000);
        // Floor at 59 (cutoff is strict ≥), tolerate boundary.
        assert!(
            stats.fps > 58.0 && stats.fps <= 61.0,
            "fps was {}",
            stats.fps
        );
    }

    #[test]
    fn latency_percentiles_on_known_distribution() {
        let records: Vec<Record> = (1..=100u64).map(|i| rec(i * 1_000_000, i, i)).collect();
        let stats = live_stats(&records, 100_000_000, 1_000_000_000);
        // Excel-style rank, matching the layer's percentile_p50_p99 helper.
        assert_eq!(stats.p50_us, 51);
        assert_eq!(stats.p99_us, 99);
        assert_eq!(stats.max_us, 100);
    }

    #[test]
    fn latencies_of_zero_are_skipped() {
        // Frames with no input_sample report latency_us=0; they must not
        // drag the median down to zero.
        let mut records: Vec<Record> = (1..=10u64).map(|i| rec(i * 1_000_000, i, 100)).collect();
        records.extend((11..=20u64).map(|i| rec(i * 1_000_000, i, 0)));
        let stats = live_stats(&records, 20_000_000, 1_000_000_000);
        assert_eq!(stats.samples, 10);
        assert_eq!(stats.p50_us, 100);
    }

    #[test]
    fn empty_input_returns_zero() {
        let stats = live_stats(&[], 0, 1_000_000_000);
        assert_eq!(stats.fps, 0.0);
        assert_eq!(stats.samples, 0);
    }

    #[test]
    fn frame_gaps_detects_dropped_presents() {
        let records = vec![
            rec(1, 1, 0),
            rec(2, 2, 0),
            // Driver dropped frames 3-5, next present is 6.
            rec(3, 6, 0),
            rec(4, 7, 0),
            // Another gap.
            rec(5, 10, 0),
        ];
        let gaps = frame_gaps(&records);
        assert_eq!(gaps, vec![[6.0, 3.0], [10.0, 2.0]]);
    }

    #[test]
    fn bin_records_skips_empty_bins() {
        // Sparse: 3 records across a 1s/100ms-bins window → only 3 entries,
        // not 10. Regression check for the "line snaps to zero between
        // samples" jitter we hit in the first live capture.
        let records = vec![
            rec(100_000_000, 1, 0),
            rec(500_000_000, 2, 0),
            rec(900_000_000, 3, 0),
        ];
        let bins = bin_records(&records, 1_000_000_000, 1_000_000_000, 100_000_000, |b| {
            b.len() as f64
        });
        assert_eq!(bins.len(), 3);
        assert!(bins.iter().all(|p| (p[1] - 1.0).abs() < f64::EPSILON));
    }

    #[test]
    fn bin_records_groups_by_time() {
        let records: Vec<Record> = (0u64..10).map(|i| rec(i * 100_000_000, i, 0)).collect();
        // 1s window, 100 ms bins → 10 bins, one record per bin.
        let bins = bin_records(&records, 1_000_000_000, 1_000_000_000, 100_000_000, |b| {
            b.len() as f64
        });
        assert_eq!(bins.len(), 10);
        assert!(bins.iter().all(|p| (p[1] - 1.0).abs() < f64::EPSILON));
    }
}
