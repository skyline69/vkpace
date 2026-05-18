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
    present_end_us: u64,
    gpu_render_start_us: u64,
    gpu_render_end_us: u64,
}

pub struct MarkerHistory {
    frames: Mutex<VecDeque<FrameMarkers>>,
}

impl MarkerHistory {
    pub fn new() -> Self {
        Self {
            frames: Mutex::new(VecDeque::with_capacity(HISTORY_CAP)),
        }
    }

    /// Record one marker. `present_id` may be 0 for SDK calls that don't
    /// know it yet; we still store under that key and merge later if a real
    /// present_id arrives on a different marker in the same frame.
    pub fn record(&self, present_id: u64, marker: vk::LatencyMarkerNV) {
        let now_us = DeviceClock::now() / 1_000;
        let mut frames = self.frames.lock();

        // Find the frame this marker belongs to, or push a new one.
        let entry = match frames.iter_mut().rev().find(|f| f.present_id == present_id) {
            Some(f) => f,
            None => {
                if frames.len() == HISTORY_CAP {
                    frames.pop_front();
                }
                frames.push_back(FrameMarkers {
                    present_id,
                    ..Default::default()
                });
                frames.back_mut().unwrap()
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
        let mut frames = self.frames.lock();
        if let Some(f) = frames.iter_mut().rev().find(|f| f.present_id == present_id) {
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
        let mut frames = self.frames.lock();
        if let Some(f) = frames.iter_mut().rev().find(|f| f.present_id == present_id) {
            f.present_end_us = actual_host_ns / 1_000;
        }
    }

    /// Fill a caller-provided `VkGetLatencyMarkerInfoNV` buffer with the most
    /// recent frames. Two-call protocol per Vulkan convention: if
    /// `p_timings == NULL`, write `timing_count = HISTORY_CAP`; otherwise
    /// copy up to `timing_count` frames and write back the actual count.
    pub fn fill(&self, timings: &mut vk::GetLatencyMarkerInfoNV<'_>) {
        let frames = self.frames.lock();

        if timings.p_timings.is_null() {
            timings.timing_count = frames.len() as u32;
            return;
        }

        let max = timings.timing_count as usize;
        let to_write = max.min(frames.len());
        for (i, src) in frames.iter().rev().take(to_write).enumerate() {
            let dst = unsafe { &mut *timings.p_timings.add(i) };
            *dst = vk::LatencyTimingsFrameReportNV::default()
                .present_id(src.present_id)
                .input_sample_time_us(src.input_sample_us)
                .sim_start_time_us(src.sim_start_us)
                .sim_end_time_us(src.sim_end_us)
                .render_submit_start_time_us(src.render_submit_start_us)
                .render_submit_end_time_us(src.render_submit_end_us)
                .present_start_time_us(src.present_start_us)
                .present_end_time_us(src.present_end_us)
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
