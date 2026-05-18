//! NV `VK_NV_low_latency2` strategy.
//!
//! Each submission may carry a `VkLatencySubmissionPresentIdNV` in its pNext
//! chain. The queue side groups submissions by present-id. At present time
//! the device side claims those groups, attaches them to the active swapchain
//! monitor, and signals the wait semaphore once the GPU has finished.

use ash::vk;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

mod semaphore_signal;
mod swapchain_monitor;

use crate::device::DeviceContext;
use crate::pnext::find;
use crate::strategy::{DeviceStrategy, QueueStrategy};
use crate::submission_span::SubmissionSpan;
use crate::timestamp_pool::Handle;

pub use semaphore_signal::SemaphoreSignal;
pub use swapchain_monitor::SwapchainMonitor;

const MAX_TRACKED_PRESENTS: usize = 50;

pub struct LowLatency2DeviceStrategy {
    device: Arc<DeviceContext>,
    monitors: RwLock<HashMap<vk::SwapchainKHR, Arc<SwapchainMonitor>>>,
}

impl LowLatency2DeviceStrategy {
    pub fn new(device: Arc<DeviceContext>) -> Self {
        Self {
            device,
            monitors: RwLock::new(HashMap::new()),
        }
    }

    pub fn submit_swapchain_present_id(&self, swapchain: vk::SwapchainKHR, present_id: u64) {
        let mut work: Vec<SubmissionSpan> = Vec::new();
        for kv in self.device.queues.iter() {
            let queue = kv.value();
            let Some(s) = queue.strategy.as_low_latency2() else {
                continue;
            };
            if s.is_out_of_band.load(Ordering::Relaxed) {
                continue;
            }
            if let Some(span) = s.take_by_present_id(present_id) {
                work.push(span);
            }
        }
        let monitors = self.monitors.read();
        if let Some(m) = monitors.get(&swapchain) {
            m.attach_work(work);
        }
    }

    pub fn notify_latency_sleep_mode(
        &self,
        swapchain: vk::SwapchainKHR,
        info: Option<&vk::LatencySleepModeInfoNV<'_>>,
    ) {
        let monitors = self.monitors.read();
        let Some(m) = monitors.get(&swapchain) else {
            return;
        };
        match info {
            Some(info) => m.update_params(
                info.low_latency_mode != vk::FALSE,
                info.minimum_interval_us as u64,
            ),
            None => m.update_params(false, 0),
        }
    }

    pub fn notify_latency_sleep_nv(
        &self,
        swapchain: vk::SwapchainKHR,
        info: &vk::LatencySleepInfoNV<'_>,
    ) {
        tracing::trace!(?swapchain, value = info.value, "LatencySleepNV");
        let signal = SemaphoreSignal::new(info.signal_semaphore, info.value);
        let monitors = self.monitors.read();
        match monitors.get(&swapchain) {
            Some(m) => m.notify_semaphore(signal),
            // Must still signal — losing the semaphore would hang the app.
            None => signal.signal(&self.device),
        }
    }
}

impl DeviceStrategy for LowLatency2DeviceStrategy {
    fn notify_create_swapchain(
        &self,
        swapchain: vk::SwapchainKHR,
        info: &vk::SwapchainCreateInfoKHR<'_>,
    ) {
        tracing::debug!(?swapchain, "LL2 create_swapchain");
        // Default ON: any app touching VK_NV_low_latency2 wants pacing.
        // VkSwapchainLatencyCreateInfoNV can override.
        let mut requested = true;
        let slci = unsafe {
            find::<vk::SwapchainLatencyCreateInfoNV>(
                info.p_next,
                vk::StructureType::SWAPCHAIN_LATENCY_CREATE_INFO_NV,
            )
        };
        if let Some(p) = slci {
            requested = unsafe { (*p).latency_mode_enable != vk::FALSE };
        }

        let monitor = Arc::new(SwapchainMonitor::new(self.device.clone()));
        monitor.update_params(requested, 0);
        self.monitors.write().insert(swapchain, monitor);
    }

    fn notify_destroy_swapchain(&self, swapchain: vk::SwapchainKHR) {
        self.monitors.write().remove(&swapchain);
    }

    fn as_low_latency2(&self) -> Option<&LowLatency2DeviceStrategy> {
        Some(self)
    }
}

pub struct LowLatency2QueueStrategy {
    submissions: Mutex<HashMap<u64, SubmissionSpan>>,
    stale_present_ids: Mutex<VecDeque<u64>>,
    pub is_out_of_band: AtomicBool,
}

impl LowLatency2QueueStrategy {
    pub fn new() -> Self {
        Self {
            submissions: Mutex::new(HashMap::new()),
            stale_present_ids: Mutex::new(VecDeque::new()),
            is_out_of_band: AtomicBool::new(false),
        }
    }

    pub fn take_by_present_id(&self, present_id: u64) -> Option<SubmissionSpan> {
        self.submissions.lock().remove(&present_id)
    }

    pub fn mark_out_of_band(&self) {
        self.is_out_of_band.store(true, Ordering::Relaxed);
    }
}

impl Default for LowLatency2QueueStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl QueueStrategy for LowLatency2QueueStrategy {
    unsafe fn notify_submit(&self, pnext_head: *const c_void, handle: Arc<Handle>) {
        let present_id = unsafe {
            find::<vk::LatencySubmissionPresentIdNV>(
                pnext_head,
                vk::StructureType::LATENCY_SUBMISSION_PRESENT_ID_NV,
            )
        }
        .map(|p| unsafe { (*p).present_id })
        .unwrap_or(0);

        let mut submissions = self.submissions.lock();
        submissions
            .entry(present_id)
            .and_modify(|s| s.extend(handle.clone()))
            .or_insert_with(|| SubmissionSpan::new(handle.clone()));

        if present_id != 0 {
            let mut stale = self.stale_present_ids.lock();
            stale.push_back(present_id);
            if stale.len() > MAX_TRACKED_PRESENTS
                && let Some(old) = stale.pop_front()
            {
                submissions.remove(&old);
            }
        }
    }

    fn notify_present(&self, _present: &vk::PresentInfoKHR<'_>) {
        // Forwarding to the device strategy needs the owning DeviceContext.
        // That's done from the `vkQueuePresentKHR` entrypoint in `lib.rs`,
        // which already has the queue's device in scope.
    }

    fn as_low_latency2(&self) -> Option<&LowLatency2QueueStrategy> {
        Some(self)
    }
}

/// Walk a `VkPresentInfoKHR` and forward each (swapchain, present_id) pair
/// to the device strategy. Called from the layer's `vkQueuePresentKHR` hook.
pub fn forward_present(strategy: &LowLatency2DeviceStrategy, present: &vk::PresentInfoKHR<'_>) {
    let present_ids = unsafe {
        find::<vk::PresentIdKHR>(present.p_next, vk::StructureType::PRESENT_ID_KHR).and_then(|p| {
            let p = &*p;
            (!p.p_present_ids.is_null())
                .then(|| std::slice::from_raw_parts(p.p_present_ids, p.swapchain_count as usize))
        })
    };
    for i in 0..present.swapchain_count as usize {
        let swapchain = unsafe { *present.p_swapchains.add(i) };
        let pid = present_ids.map(|s| s[i]).unwrap_or(0);
        strategy.submit_swapchain_present_id(swapchain, pid);
    }
}
