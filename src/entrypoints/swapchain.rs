//! Swapchain lifecycle + acquire entrypoints.

use ash::vk;

use crate::clock;
use crate::device::DeviceContext;
use crate::registry;

/// `VK_PRESENT_MODE_FIFO_LATEST_READY_EXT` (numeric id 1000361000) lets the
/// driver pick the freshest queued image at vblank. Our fps_cap / VRR
/// target is computed from frame submission rate; under that mode it stops
/// being authoritative for visible frame cadence. Warn the user once so
/// surprising "fps_cap=144 yet I see 300 fps" reports get diagnosed fast.
const PRESENT_MODE_FIFO_LATEST_READY: i32 = 1_000_361_000;

fn warn_if_fifo_latest_ready(info: &vk::SwapchainCreateInfoKHR<'_>, ctx: &DeviceContext) {
    if info.present_mode.as_raw() != PRESENT_MODE_FIFO_LATEST_READY {
        return;
    }
    let cfg = &ctx.instance.config;
    if cfg.fps_cap == 0 && !cfg.vrr_enabled {
        return; // user didn't enable pacing — nothing to warn about
    }
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "swapchain present_mode = FIFO_LATEST_READY: fps_cap / VKPACE_VRR \
             are advisory under this mode (driver picks freshest queued image)"
        );
    });
}

pub(crate) unsafe extern "system" fn create_swapchain_khr(
    device: vk::Device,
    p_create_info: *const vk::SwapchainCreateInfoKHR<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_swapchain: *mut vk::SwapchainKHR,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let Some(create) = ctx.fns.create_swapchain_khr else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if let Some(info) = unsafe { p_create_info.as_ref() } {
            warn_if_fifo_latest_ready(info, &ctx);
        }
        let r = unsafe { create(device, p_create_info, p_allocator, p_swapchain) };
        if r != vk::Result::SUCCESS {
            return r;
        }
        if let (Some(info), Some(s)) = (unsafe { p_create_info.as_ref() }, ctx.strategy.get()) {
            s.notify_create_swapchain(unsafe { *p_swapchain }, info);
        }
        r
    })
}

pub(crate) unsafe extern "system" fn acquire_next_image_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    timeout: u64,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    p_image_index: *mut u32,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let Some(acquire) = ctx.fns.acquire_next_image_khr else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let start_ns = clock::DeviceClock::now();
        let r = unsafe { acquire(device, swapchain, timeout, semaphore, fence, p_image_index) };
        let elapsed = clock::DeviceClock::now().saturating_sub(start_ns);
        crate::TELEMETRY
            .counters
            .acquires
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if elapsed > 16_000_000 {
            tracing::debug!(swapchain = ?swapchain, elapsed_us = elapsed / 1_000, "long acquire");
        }
        r
    })
}

pub(crate) unsafe extern "system" fn acquire_next_image2_khr(
    device: vk::Device,
    p_acquire_info: *const vk::AcquireNextImageInfoKHR<'_>,
    p_image_index: *mut u32,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let Some(acquire) = ctx.fns.acquire_next_image2_khr else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let start_ns = clock::DeviceClock::now();
        let r = unsafe { acquire(device, p_acquire_info, p_image_index) };
        let elapsed = clock::DeviceClock::now().saturating_sub(start_ns);
        crate::TELEMETRY
            .counters
            .acquires
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if elapsed > 16_000_000 {
            tracing::debug!(elapsed_us = elapsed / 1_000, "long acquire2");
        }
        r
    })
}

pub(crate) unsafe extern "system" fn destroy_swapchain_khr(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    crate::catch::vk_void(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return;
        };
        if let Some(s) = ctx.strategy.get() {
            s.notify_destroy_swapchain(swapchain);
        }
        if let Some(w) = ctx.google_timing.get() {
            w.forget_swapchain(swapchain);
        }
        if let Some(destroy) = ctx.fns.destroy_swapchain_khr {
            unsafe { destroy(device, swapchain, p_allocator) };
        }
    })
}
