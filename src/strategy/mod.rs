//! Strategy abstraction — the actual latency-reduction policy.
//!
//! Each device picks one strategy at create time, depending on whether the
//! layer should expose AMD Anti-Lag or NV Low-Latency-2 (set by env var).
//! The queue-side strategy is its sibling and lives per-queue.

use ash::vk;
use std::sync::Arc;

use crate::device::DeviceContext;
use crate::submission_span::SubmissionSpan;
use crate::timestamp_pool::Handle;

pub mod anti_lag;
pub mod low_latency2;

pub trait DeviceStrategy: Send + Sync {
    /// Called from `vkCreateSwapchainKHR`.
    fn notify_create_swapchain(
        &self,
        swapchain: vk::SwapchainKHR,
        info: &vk::SwapchainCreateInfoKHR<'_>,
    );

    /// Called from `vkDestroySwapchainKHR`.
    fn notify_destroy_swapchain(&self, swapchain: vk::SwapchainKHR);

    /// Test-friendly downcast helpers.
    fn as_anti_lag(&self) -> Option<&anti_lag::AntiLagDeviceStrategy> {
        None
    }
    fn as_low_latency2(&self) -> Option<&low_latency2::LowLatency2DeviceStrategy> {
        None
    }
}

pub trait QueueStrategy: Send + Sync {
    /// Track a submission whose timestamps live in `handle`. `pnext_head`
    /// (the `pNext` from the original `VkSubmitInfo[2]`) is used by the
    /// LowLatency2 strategy to extract `VkLatencySubmissionPresentIdNV`.
    ///
    /// # Safety
    /// `pnext_head` must be a valid pNext chain pointer (may be null).
    unsafe fn notify_submit(&self, pnext_head: *const std::ffi::c_void, handle: Arc<Handle>);

    fn notify_present(&self, present: &vk::PresentInfoKHR<'_>);

    fn as_anti_lag(&self) -> Option<&anti_lag::AntiLagQueueStrategy> {
        None
    }
    fn as_low_latency2(&self) -> Option<&low_latency2::LowLatency2QueueStrategy> {
        None
    }
}

pub fn make_device_strategy(device: &Arc<DeviceContext>) -> Arc<dyn DeviceStrategy> {
    if device.instance.config.expose_reflex {
        Arc::new(low_latency2::LowLatency2DeviceStrategy::new(device.clone()))
    } else {
        Arc::new(anti_lag::AntiLagDeviceStrategy::new(device.clone()))
    }
}

pub fn make_queue_strategy(device: &Arc<DeviceContext>) -> Box<dyn QueueStrategy> {
    if device.instance.config.expose_reflex {
        Box::new(low_latency2::LowLatency2QueueStrategy::new())
    } else {
        Box::new(anti_lag::AntiLagQueueStrategy::new())
    }
}

/// Convenience for strategies that aggregate per-queue submissions.
pub(super) fn extend_or_init(slot: &mut Option<SubmissionSpan>, handle: Arc<Handle>) {
    match slot.as_mut() {
        Some(span) => span.extend(handle),
        None => *slot = Some(SubmissionSpan::new(handle)),
    }
}
