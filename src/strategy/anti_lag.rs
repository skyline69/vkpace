//! AMD `VK_AMD_anti_lag` strategy.
//!
//! The application calls `vkAntiLagUpdateAMD` twice per frame: once tagged
//! `INPUT` (we collect submission timestamps from now until `PRESENT`) and
//! once tagged `PRESENT` (we stop collecting). At the next `INPUT` we wait
//! on the previously-collected work and run the delay controller.

use ash::vk;
use parking_lot::Mutex;
use std::ffi::c_void;
use std::sync::Arc;

use crate::amd_anti_lag::{AntiLagDataAMD, AntiLagModeAMD, AntiLagStageAMD};
use crate::clock::DeviceClock;
use crate::delay_controller::DelayController;
use crate::device::DeviceContext;
use crate::strategy::low_latency2::MarkerHistory;
use crate::strategy::{DeviceStrategy, QueueStrategy, extend_or_init};
use crate::submission_span::SubmissionSpan;
use crate::timestamp_pool::Handle;

struct State {
    frame_index: Option<u64>,
    enabled: bool,
    delay_controller: DelayController,
}

pub struct AntiLagDeviceStrategy {
    device: Arc<DeviceContext>,
    state: Mutex<State>,
    /// Synthetic marker history so the telemetry stats worker can derive
    /// p50/p99 click-to-photon for AntiLag too. Keys frames by the
    /// app-supplied AntiLag frame_index (which doubles as our pid).
    markers: Arc<MarkerHistory>,
}

impl AntiLagDeviceStrategy {
    pub fn new(device: Arc<DeviceContext>) -> Self {
        let decoupled = device.instance.is_simulation_decoupled;
        let markers = Arc::new(MarkerHistory::new());
        // Register synthetic markers with global telemetry. Weak so
        // teardown auto-evicts.
        let src: Arc<dyn crate::telemetry::LatencySource> = markers.clone();
        crate::TELEMETRY.register_latency_source(Arc::downgrade(&src));
        Self {
            device,
            state: Mutex::new(State {
                frame_index: None,
                enabled: false,
                delay_controller: DelayController::new(decoupled),
            }),
            markers,
        }
    }

    /// Cheap predicate used on the QueueSubmit hot path to gate timestamp
    /// injection. AntiLag's input-stage handler only cares about submits
    /// happening between `INPUT` and `PRESENT` markers; outside that window
    /// the timestamps would be discarded anyway.
    pub fn should_track_submissions(&self) -> bool {
        let st = self.state.lock();
        st.enabled && st.frame_index.is_some()
    }

    /// Called from `vkAntiLagUpdateAMD`. Mode/presentation are app-driven.
    pub fn notify_update(&self, data: &AntiLagDataAMD) {
        tracing::trace!(mode = data.mode.0, max_fps = data.max_fps, "AntiLagUpdate");
        let min_delay_ns: u64 = {
            let mut st = self.state.lock();
            st.enabled = data.mode != AntiLagModeAMD::OFF;
            let presentation = unsafe { data.p_presentation_info.as_ref() };
            let app_cap = compute_min_delay_ns(data.max_fps);
            let layer_cap = self.device.effective_min_delay_ns();
            // Use the stricter of the two caps (longer min_delay).
            let min_delay_ns = app_cap.max(layer_cap);

            let Some(pres) = presentation else {
                return;
            };
            if !st.enabled {
                return;
            }
            match pres.stage {
                AntiLagStageAMD::PRESENT => {
                    st.frame_index = None;
                    // PRESENT marker carries display-side timing; record
                    // both PRESENT_END and the layer-measured "display
                    // actual" using current host-now (we don't have a
                    // post-scanout source on the AntiLag path).
                    self.markers
                        .record(pres.frame_index, vk::LatencyMarkerNV::PRESENT_END);
                    self.markers
                        .record_present_actual(pres.frame_index, DeviceClock::now());
                    return;
                }
                AntiLagStageAMD::INPUT => {
                    st.frame_index = Some(pres.frame_index);
                    self.markers
                        .record(pres.frame_index, vk::LatencyMarkerNV::INPUT_SAMPLE);
                }
                _ => {
                    tracing::trace!(stage = pres.stage.0, "AntiLag: unknown stage");
                    st.frame_index = Some(pres.frame_index);
                }
            }
            min_delay_ns
        };

        // Steal every queue's outstanding submission span before waiting.
        let mut work = Vec::new();
        for kv in self.device.queues.iter() {
            let queue = kv.value();
            if let Some(s) = queue.strategy.as_anti_lag()
                && let Some(span) = s.take_submission_span()
            {
                work.push(span);
            }
        }

        if let Some(clock) = self.device.clock.as_deref() {
            for span in &work {
                span.await_completed(clock);
            }
        }

        let mut st = self.state.lock();
        st.delay_controller.delay(min_delay_ns);
    }
}

impl DeviceStrategy for AntiLagDeviceStrategy {
    fn notify_create_swapchain(
        &self,
        _swapchain: vk::SwapchainKHR,
        _info: &vk::SwapchainCreateInfoKHR<'_>,
    ) {
    }
    fn notify_destroy_swapchain(&self, _swapchain: vk::SwapchainKHR) {}
    fn as_anti_lag(&self) -> Option<&AntiLagDeviceStrategy> {
        Some(self)
    }
}

fn compute_min_delay_ns(max_fps: u32) -> u64 {
    if max_fps == 0 {
        0
    } else {
        1_000_000_000 / max_fps as u64
    }
}

pub struct AntiLagQueueStrategy {
    span: Mutex<Option<SubmissionSpan>>,
}

impl AntiLagQueueStrategy {
    pub fn new() -> Self {
        Self {
            span: Mutex::new(None),
        }
    }

    fn take_submission_span(&self) -> Option<SubmissionSpan> {
        self.span.lock().take()
    }
}

impl Default for AntiLagQueueStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl QueueStrategy for AntiLagQueueStrategy {
    unsafe fn notify_submit(&self, _pnext_head: *const c_void, handle: Arc<Handle>) {
        extend_or_init(&mut self.span.lock(), handle);
    }

    fn notify_present(&self, _present: &vk::PresentInfoKHR<'_>) {}

    fn as_anti_lag(&self) -> Option<&AntiLagQueueStrategy> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_min_delay_handles_zero_and_common_caps() {
        assert_eq!(compute_min_delay_ns(0), 0);
        assert_eq!(compute_min_delay_ns(60), 16_666_666);
        assert_eq!(compute_min_delay_ns(144), 6_944_444);
        assert_eq!(compute_min_delay_ns(240), 4_166_666);
    }
}
