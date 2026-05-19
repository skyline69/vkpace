use ash::vk;
use rustc_hash::FxBuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::clock::DeviceClock;
use crate::dispatch::DeviceTable;
use crate::instance::InstanceContext;
use crate::physical_device::PhysicalDeviceContext;
use crate::present_wait::{GoogleTimingWorker, PresentWaitWorker};
use crate::queue::QueueContext;
use crate::registry::FxDashMap;
use crate::strategy::DeviceStrategy;

/// Per-`VkDevice` state.
///
/// `strategy` is `OnceLock`: written exactly once at end of `CreateDevice`
/// and read on every submit/present, so we want a wait-free read path
/// without any locking overhead.
pub struct DeviceContext {
    pub handle: vk::Device,
    pub fns: Arc<DeviceTable>,
    pub instance: Arc<InstanceContext>,
    pub physical_device: Arc<PhysicalDeviceContext>,
    pub layer_enabled: bool,
    pub clock: Option<Arc<DeviceClock>>,
    pub strategy: OnceLock<Arc<dyn DeviceStrategy>>,
    pub queues: FxDashMap<u64, Arc<QueueContext>>,
    /// Optional `vkWaitForPresentKHR` worker. Populated when the device
    /// exposes `VK_KHR_present_wait` *and* the active strategy is LL2
    /// (Reflex) — only LL2 supplies the `present_id`s we need.
    pub present_wait: OnceLock<PresentWaitWorker>,
    /// Fallback for drivers that expose `VK_GOOGLE_display_timing` but not
    /// `VK_KHR_present_wait` (older Mesa, some mobile drivers). Filled in
    /// instead of `present_wait`, never alongside.
    pub google_timing: OnceLock<GoogleTimingWorker>,
    /// Effective VRR-derived min-frame delay in ns. 0 until we've sampled
    /// the display refresh duration via
    /// `vkGetRefreshCycleDurationGOOGLE`. Combined with the explicit
    /// `fps_cap` (whichever is stricter) when pacing each frame.
    pub vrr_target_delay_ns: AtomicU64,
}

impl DeviceContext {
    pub fn new(
        physical_device: Arc<PhysicalDeviceContext>,
        handle: vk::Device,
        fns: DeviceTable,
        layer_enabled: bool,
    ) -> Arc<Self> {
        let fns = Arc::new(fns);
        let clock = if layer_enabled {
            DeviceClock::new(
                handle,
                fns.clone(),
                physical_device.properties.limits.timestamp_period,
            )
            .map(Arc::new)
        } else {
            None
        };
        let instance = physical_device.instance.clone();

        let ctx = Arc::new(Self {
            handle,
            fns,
            instance,
            physical_device,
            layer_enabled,
            clock,
            strategy: OnceLock::new(),
            queues: FxDashMap::with_hasher(FxBuildHasher),
            present_wait: OnceLock::new(),
            google_timing: OnceLock::new(),
            vrr_target_delay_ns: AtomicU64::new(0),
        });

        if layer_enabled {
            let strategy = crate::strategy::make_device_strategy(&ctx);
            ctx.strategy
                .set(strategy.clone())
                .ok()
                .expect("DeviceContext::strategy set twice");

            // Spin up a present-completion worker — prefer the precise
            // `vkWaitForPresentKHR` path, fall back to the FIFO-ordered
            // `VK_GOOGLE_display_timing` polling worker. Either populates
            // `record_present_actual` on the LL2 marker history. AntiLag
            // doesn't currently surface present_ids in a correlatable form.
            if let Some(ll2) = strategy.as_low_latency2() {
                let markers = ll2.markers_arc();
                // Surface this device's marker history to the global
                // telemetry stats worker as a click-to-photon source.
                let src: std::sync::Arc<dyn crate::telemetry::LatencySource> = markers.clone();
                crate::TELEMETRY.register_latency_source(std::sync::Arc::downgrade(&src));
                if let Some(worker) =
                    PresentWaitWorker::new(ctx.handle, ctx.fns.clone(), markers.clone())
                {
                    let _ = ctx.present_wait.set(worker);
                } else if let Some(worker) =
                    GoogleTimingWorker::new(ctx.handle, ctx.fns.clone(), markers)
                {
                    let _ = ctx.google_timing.set(worker);
                }
            }
        }

        ctx
    }

    /// Effective minimum inter-frame delay (ns) applied by the pacer.
    /// Combines the explicit `fps_cap` with the VRR-derived target, taking
    /// the stricter (longer-delay) of the two.
    #[inline]
    pub fn effective_min_delay_ns(&self) -> u64 {
        let cap = self.instance.config.fps_cap_min_delay_ns();
        let vrr = self.vrr_target_delay_ns.load(Ordering::Relaxed);
        cap.max(vrr)
    }

    /// Lazy-init the VRR soft target on first present that knows a
    /// swapchain handle. Cheap on the steady-state path: once set, never
    /// re-queried.
    pub fn ensure_vrr_target(&self, swapchain: vk::SwapchainKHR) {
        if !self.instance.config.vrr_enabled {
            return;
        }
        if self.vrr_target_delay_ns.load(Ordering::Relaxed) != 0 {
            return;
        }
        let Some(query) = self.fns.get_refresh_cycle_duration_google else {
            return;
        };
        let mut props = vk::RefreshCycleDurationGOOGLE::default();
        let r = unsafe { query(self.handle, swapchain, &mut props) };
        if r != vk::Result::SUCCESS || props.refresh_duration == 0 {
            return;
        }
        // refresh_duration is ns/frame. Subtract `vrr_offset_hz` to derive
        // a slightly slower target — common LFC trick to stay inside the
        // VRR window without bumping the cap.
        let refresh_hz = 1_000_000_000u64 / props.refresh_duration;
        let cfg = &self.instance.config;
        let target_hz = refresh_hz.saturating_sub(cfg.vrr_offset_hz as u64).max(1);
        let delay_ns = 1_000_000_000u64 / target_hz;
        self.vrr_target_delay_ns.store(delay_ns, Ordering::Relaxed);
        tracing::info!(
            refresh_hz,
            target_hz,
            offset_hz = cfg.vrr_offset_hz,
            delay_ns,
            "VRR soft target initialised"
        );
    }

    /// Drain pending GPU work and clear all queues before the layer state
    /// is freed. Called from the `vkDestroyDevice` hook.
    pub fn drain_for_destroy(&self) {
        let r = unsafe { (self.fns.device_wait_idle)(self.handle) };
        if r != vk::Result::SUCCESS {
            tracing::warn!(?r, "vkDeviceWaitIdle returned non-success during drain");
        }
        self.queues.clear();
    }
}
