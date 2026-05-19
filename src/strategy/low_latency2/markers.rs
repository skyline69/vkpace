//! Reflex latency marker history.
//!
//! `vkSetLatencyMarkerNV` is called by the application several times per
//! frame to mark logical stages: `INPUT_SAMPLE`, `SIMULATION_START/END`,
//! `RENDERSUBMIT_START/END`, `PRESENT_START/END`. The Reflex SDK / NV
//! overlay later calls `vkGetLatencyTimingsNV` to retrieve a recent
//! history of these timings as `VkLatencyTimingsFrameReportNV` entries.
//!
//! We record `CLOCK_MONOTONIC` microseconds at marker time and serve the
//! history back on demand. GPU-side timestamps (`gpu_render_start/end`)
//! are filled in from the present-time strategy if available.

use ash::vk;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

use crate::clock::DeviceClock;

/// Keep last N frames of marker data. Reflex SDK typically samples this at
/// 32-64 frames of history; 128 gives headroom without significant memory.
const HISTORY_CAP: usize = 128;

#[derive(Clone, Copy, Default)]
struct FrameMarkers {
    present_id: u64,
    input_sample_us: u64,
    sim_start_us: u64,
    sim_end_us: u64,
    render_submit_start_us: u64,
    render_submit_end_us: u64,
    present_start_us: u64,
    /// App-set `PRESENT_END` marker — per spec the moment the app finished
    /// issuing the present call. Kept verbatim so the Reflex report stays
    /// faithful to what the application reported.
    present_end_us: u64,
    /// Layer-measured actual display-side scanout time (host monotonic µs),
    /// sourced from `vkWaitForPresentKHR` or `VK_GOOGLE_display_timing`.
    /// Used for click-to-photon telemetry and exported in the Reflex report
    /// only when the app didn't supply its own `PRESENT_END` for the frame.
    display_actual_us: u64,
    gpu_render_start_us: u64,
    gpu_render_end_us: u64,
}

struct Inner {
    frames: VecDeque<FrameMarkers>,
    /// Sidecar `present_id → index in frames` map. Keeps `record()`
    /// O(1) instead of scanning the deque on every marker call
    /// (~6 markers × HISTORY_CAP frames). Indices are absolute frame
    /// numbers; subtract `head_index` to get the slot in `frames`.
    by_present_id: FxHashMap<u64, u64>,
    /// Absolute index of the oldest frame currently in `frames`.
    head_index: u64,
}

pub struct MarkerHistory {
    inner: Mutex<Inner>,
}

impl MarkerHistory {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                frames: VecDeque::with_capacity(HISTORY_CAP),
                by_present_id: FxHashMap::default(),
                head_index: 0,
            }),
        }
    }

    /// Record one marker. `present_id` may be 0 for SDK calls that don't
    /// know it yet; we still store under that key and merge later if a real
    /// present_id arrives on a different marker in the same frame.
    pub fn record(&self, present_id: u64, marker: vk::LatencyMarkerNV) {
        let now_us = DeviceClock::now() / 1_000;
        let mut inner = self.inner.lock();

        // O(1) lookup: hash → absolute index → slot.
        let entry = match inner.by_present_id.get(&present_id).copied() {
            Some(abs) => {
                let slot = (abs - inner.head_index) as usize;
                &mut inner.frames[slot]
            }
            None => {
                if inner.frames.len() == HISTORY_CAP {
                    if let Some(old) = inner.frames.pop_front() {
                        inner.by_present_id.remove(&old.present_id);
                    }
                    inner.head_index += 1;
                }
                let new_abs = inner.head_index + inner.frames.len() as u64;
                inner.frames.push_back(FrameMarkers {
                    present_id,
                    ..Default::default()
                });
                inner.by_present_id.insert(present_id, new_abs);
                inner.frames.back_mut().unwrap()
            }
        };

        match marker {
            vk::LatencyMarkerNV::INPUT_SAMPLE => entry.input_sample_us = now_us,
            vk::LatencyMarkerNV::SIMULATION_START => entry.sim_start_us = now_us,
            vk::LatencyMarkerNV::SIMULATION_END => entry.sim_end_us = now_us,
            vk::LatencyMarkerNV::RENDERSUBMIT_START => entry.render_submit_start_us = now_us,
            vk::LatencyMarkerNV::RENDERSUBMIT_END => entry.render_submit_end_us = now_us,
            vk::LatencyMarkerNV::PRESENT_START => entry.present_start_us = now_us,
            vk::LatencyMarkerNV::PRESENT_END => entry.present_end_us = now_us,
            _ => {
                // TRIGGER_FLASH + the OUT_OF_BAND_* variants — record nothing
                // for now; they're informational and apps don't expect us to
                // surface them in the history report.
            }
        }
    }

    /// Record GPU render start/end (host monotonic ns) for an already-seen
    /// `present_id`. Called from the swapchain monitor after a frame's
    /// submission span completes and we know the timestamps.
    pub fn record_gpu_timing(&self, present_id: u64, start_host_ns: u64, end_host_ns: u64) {
        let mut inner = self.inner.lock();
        if let Some(abs) = inner.by_present_id.get(&present_id).copied() {
            let slot = (abs - inner.head_index) as usize;
            let f = &mut inner.frames[slot];
            f.gpu_render_start_us = start_host_ns / 1_000;
            f.gpu_render_end_us = end_host_ns / 1_000;
        }
    }

    /// Record actual display-side present completion (host monotonic ns)
    /// for an already-seen `present_id`. Sourced from
    /// `vkWaitForPresentKHR` on the present-wait worker thread. We map
    /// this onto `present_end_time_us` in the Reflex report — the spec
    /// nominally means "app finished issuing the present call", but
    /// since vkpace targets latency reporting, the actual scanout
    /// timestamp is far more useful and is what the NVIDIA overlay
    /// surfaces as the frame's tail.
    pub fn record_present_actual(&self, present_id: u64, actual_host_ns: u64) {
        let mut inner = self.inner.lock();
        if let Some(abs) = inner.by_present_id.get(&present_id).copied() {
            let slot = (abs - inner.head_index) as usize;
            // Distinct field so we never clobber the app-set PRESENT_END
            // marker. The fill() path falls back to display_actual_us only
            // when present_end_us is empty.
            inner.frames[slot].display_actual_us = actual_host_ns / 1_000;
        }
    }

    /// Last frame's click-to-photon (µs) — newest frame with both an
    /// input-sample and a display-side endpoint. 0 if not yet measurable.
    /// Cheap snapshot for the per-present telemetry record.
    pub fn latest_latency_us(&self) -> u64 {
        let inner = self.inner.lock();
        for f in inner.frames.iter().rev() {
            if f.input_sample_us == 0 {
                continue;
            }
            let end = if f.display_actual_us != 0 {
                f.display_actual_us
            } else {
                f.present_end_us
            };
            if end > f.input_sample_us {
                return end - f.input_sample_us;
            }
        }
        0
    }

    /// Fill a caller-provided `VkGetLatencyMarkerInfoNV` buffer with the most
    /// recent frames. Two-call protocol per Vulkan convention: if
    /// `p_timings == NULL`, write `timing_count = HISTORY_CAP`; otherwise
    /// copy up to `timing_count` frames and write back the actual count.
    pub fn fill(&self, timings: &mut vk::GetLatencyMarkerInfoNV<'_>) {
        let inner = self.inner.lock();

        if timings.p_timings.is_null() {
            timings.timing_count = inner.frames.len() as u32;
            return;
        }

        let max = timings.timing_count as usize;
        let to_write = max.min(inner.frames.len());
        for (i, src) in inner.frames.iter().rev().take(to_write).enumerate() {
            let dst = unsafe { &mut *timings.p_timings.add(i) };
            // Prefer app's PRESENT_END marker; only fall back to layer's
            // measured display-actual when the app didn't supply one.
            let present_end = if src.present_end_us != 0 {
                src.present_end_us
            } else {
                src.display_actual_us
            };
            *dst = vk::LatencyTimingsFrameReportNV::default()
                .present_id(src.present_id)
                .input_sample_time_us(src.input_sample_us)
                .sim_start_time_us(src.sim_start_us)
                .sim_end_time_us(src.sim_end_us)
                .render_submit_start_time_us(src.render_submit_start_us)
                .render_submit_end_time_us(src.render_submit_end_us)
                .present_start_time_us(src.present_start_us)
                .present_end_time_us(present_end)
                .gpu_render_start_time_us(src.gpu_render_start_us)
                .gpu_render_end_time_us(src.gpu_render_end_us);
        }
        timings.timing_count = to_write as u32;
    }
}

impl Default for MarkerHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::telemetry::LatencySource for MarkerHistory {
    /// Append click-to-photon samples (µs) for every frame in the window
    /// that has both an input sample and a present-end timestamp. We prefer
    /// the layer-measured display-actual time when available (that's the
    /// real scanout moment); fall back to the app's PRESENT_END marker if
    /// the present-wait worker didn't catch this frame. Frames missing
    /// either endpoint are skipped.
    fn latencies_us(&self, out: &mut Vec<u64>) {
        let inner = self.inner.lock();
        for f in inner.frames.iter() {
            if f.input_sample_us == 0 {
                continue;
            }
            let end = if f.display_actual_us != 0 {
                f.display_actual_us
            } else {
                f.present_end_us
            };
            if end > f.input_sample_us {
                out.push(end - f.input_sample_us);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::LatencySource;

    #[test]
    fn record_dedupes_by_present_id() {
        let h = MarkerHistory::new();
        for m in [
            vk::LatencyMarkerNV::INPUT_SAMPLE,
            vk::LatencyMarkerNV::SIMULATION_START,
            vk::LatencyMarkerNV::PRESENT_END,
        ] {
            h.record(42, m);
        }
        let inner = h.inner.lock();
        assert_eq!(inner.frames.len(), 1);
        assert_eq!(inner.frames[0].present_id, 42);
    }

    /// Two-call protocol per Vulkan convention: first call with NULL
    /// p_timings must populate `timing_count`, second call (or any with
    /// non-NULL p_timings) writes that many entries.
    #[test]
    fn fill_two_call_protocol_returns_count_with_null_buffer() {
        let h = MarkerHistory::new();
        for pid in 1..=5u64 {
            h.record(pid, vk::LatencyMarkerNV::INPUT_SAMPLE);
        }
        let mut info = vk::GetLatencyMarkerInfoNV {
            p_timings: std::ptr::null_mut(),
            timing_count: 0,
            ..Default::default()
        };
        h.fill(&mut info);
        assert_eq!(info.timing_count, 5);
    }

    #[test]
    fn fill_writes_up_to_requested_count_in_reverse_order() {
        let h = MarkerHistory::new();
        for pid in 1..=5u64 {
            h.record(pid, vk::LatencyMarkerNV::INPUT_SAMPLE);
        }
        let mut buf: Vec<vk::LatencyTimingsFrameReportNV> =
            vec![vk::LatencyTimingsFrameReportNV::default(); 3];
        let mut info = vk::GetLatencyMarkerInfoNV {
            p_timings: buf.as_mut_ptr(),
            timing_count: buf.len() as u32,
            ..Default::default()
        };
        h.fill(&mut info);
        assert_eq!(info.timing_count, 3);
        // Newest-first per fill loop (`.iter().rev()`).
        assert_eq!(buf[0].present_id, 5);
        assert_eq!(buf[1].present_id, 4);
        assert_eq!(buf[2].present_id, 3);
    }

    #[test]
    fn history_cap_evicts_oldest() {
        let h = MarkerHistory::new();
        for pid in 0..(HISTORY_CAP + 32) as u64 {
            h.record(pid, vk::LatencyMarkerNV::INPUT_SAMPLE);
        }
        let inner = h.inner.lock();
        assert_eq!(inner.frames.len(), HISTORY_CAP);
        assert_eq!(inner.frames.front().unwrap().present_id, 32);
        assert_eq!(
            inner.frames.back().unwrap().present_id,
            (HISTORY_CAP + 31) as u64
        );
        // Sidecar map must mirror the deque exactly.
        assert_eq!(inner.by_present_id.len(), HISTORY_CAP);
    }

    #[test]
    fn latency_source_skips_frames_missing_endpoints() {
        let h = MarkerHistory::new();
        h.record(1, vk::LatencyMarkerNV::INPUT_SAMPLE);
        // Manually set present_end to avoid timing dependence on the clock.
        {
            let mut inner = h.inner.lock();
            inner.frames[0].input_sample_us = 100;
            inner.frames[0].present_end_us = 500;
        }
        h.record(2, vk::LatencyMarkerNV::INPUT_SAMPLE);
        // pid=2 has no present_end → skipped.
        let mut out = Vec::new();
        h.latencies_us(&mut out);
        assert_eq!(out, vec![400]);
    }

    #[test]
    fn record_present_actual_does_not_clobber_app_present_end() {
        let h = MarkerHistory::new();
        h.record(7, vk::LatencyMarkerNV::INPUT_SAMPLE);
        // App-set marker.
        {
            let mut inner = h.inner.lock();
            inner.frames[0].input_sample_us = 100;
            inner.frames[0].present_end_us = 800;
        }
        // Layer worker reports a different (later) actual display time.
        h.record_present_actual(7, 1_500_000); // 1500 µs after conversion
        let inner = h.inner.lock();
        // App marker preserved verbatim.
        assert_eq!(inner.frames[0].present_end_us, 800);
        // Layer-measured field carries the new value.
        assert_eq!(inner.frames[0].display_actual_us, 1500);
    }

    #[test]
    fn latency_source_prefers_display_actual_over_app_present_end() {
        let h = MarkerHistory::new();
        h.record(9, vk::LatencyMarkerNV::INPUT_SAMPLE);
        {
            let mut inner = h.inner.lock();
            inner.frames[0].input_sample_us = 100;
            inner.frames[0].present_end_us = 200;
            inner.frames[0].display_actual_us = 1_000;
        }
        let mut out = Vec::new();
        h.latencies_us(&mut out);
        // 1000 - 100 wins over 200 - 100 because display_actual reflects
        // real scanout time.
        assert_eq!(out, vec![900]);
    }
}
