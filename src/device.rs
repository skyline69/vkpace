use ash::vk;
use rustc_hash::FxBuildHasher;
use std::sync::{Arc, OnceLock};

use crate::clock::DeviceClock;
use crate::dispatch::DeviceTable;
use crate::instance::InstanceContext;
use crate::physical_device::PhysicalDeviceContext;
use crate::present_wait::PresentWaitWorker;
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
        });

        if layer_enabled {
            let strategy = crate::strategy::make_device_strategy(&ctx);
            ctx.strategy
                .set(strategy.clone())
                .ok()
                .expect("DeviceContext::strategy set twice");

            // Spin up the present-wait worker, but only when LL2 is the
            // active strategy and the device exposes vkWaitForPresentKHR.
            // Other paths (AntiLag) don't currently surface present_ids in
            // a form we can correlate with marker history.
            if let Some(ll2) = strategy.as_low_latency2()
                && let Some(worker) =
                    PresentWaitWorker::new(ctx.handle, ctx.fns.clone(), ll2.markers_arc())
            {
                let _ = ctx.present_wait.set(worker);
            }
        }

        ctx
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
