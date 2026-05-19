//! `VK_NV_low_latency2` device entrypoints + `VK_AMD_anti_lag` update hook.

use ash::vk;

use crate::amd_anti_lag;
use crate::registry;

pub(crate) unsafe extern "system" fn latency_sleep_nv(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_sleep_info: *const vk::LatencySleepInfoNV<'_>,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return vk::Result::SUCCESS;
        };
        if let (Some(info), Some(s)) = (
            unsafe { p_sleep_info.as_ref() },
            ctx.strategy.get().and_then(|s| s.as_low_latency2()),
        ) {
            s.notify_latency_sleep_nv(swapchain, info);
        }
        vk::Result::SUCCESS
    })
}

pub(crate) unsafe extern "system" fn set_latency_sleep_mode_nv(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    p_sleep_mode_info: *const vk::LatencySleepModeInfoNV<'_>,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return vk::Result::SUCCESS;
        };
        if let Some(s) = ctx.strategy.get().and_then(|s| s.as_low_latency2()) {
            s.notify_latency_sleep_mode(swapchain, unsafe { p_sleep_mode_info.as_ref() });
        }
        vk::Result::SUCCESS
    })
}

pub(crate) unsafe extern "system" fn set_latency_marker_nv(
    device: vk::Device,
    _swapchain: vk::SwapchainKHR,
    info: *const vk::SetLatencyMarkerInfoNV<'_>,
) {
    crate::catch::vk_void(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return;
        };
        let Some(info) = (unsafe { info.as_ref() }) else {
            return;
        };
        let Some(strategy) = ctx.strategy.get().and_then(|s| s.as_low_latency2()) else {
            return;
        };
        strategy.markers().record(info.present_id, info.marker);
    })
}

pub(crate) unsafe extern "system" fn get_latency_timings_nv(
    device: vk::Device,
    _swapchain: vk::SwapchainKHR,
    timings: *mut vk::GetLatencyMarkerInfoNV<'_>,
) {
    crate::catch::vk_void(|| {
        if timings.is_null() {
            return;
        }
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            unsafe { (*timings).timing_count = 0 };
            return;
        };
        match ctx.strategy.get().and_then(|s| s.as_low_latency2()) {
            Some(strategy) => unsafe {
                strategy.markers().fill(&mut *timings);
            },
            None => unsafe {
                (*timings).timing_count = 0;
            },
        }
    })
}

pub(crate) unsafe extern "system" fn queue_notify_out_of_band_nv(
    queue: vk::Queue,
    _info: *const vk::OutOfBandQueueTypeInfoNV<'_>,
) {
    crate::catch::vk_void(|| {
        if let Some(qctx) = registry::QUEUES
            .get(&registry::key(queue))
            .map(|r| r.clone())
            && let Some(s) = qctx.strategy.as_low_latency2()
        {
            s.mark_out_of_band();
        }
    })
}

pub(crate) unsafe extern "system" fn anti_lag_update_amd(
    device: vk::Device,
    p_data: *const amd_anti_lag::AntiLagDataAMD,
) {
    crate::catch::vk_void(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return;
        };
        if p_data.is_null() {
            return;
        }
        // Defensive copy: read the entire AntiLagDataAMD (and its referenced
        // AntiLagPresentationInfoAMD if any) onto the stack before doing further
        // work, so the application can't free the memory mid-call.
        let data = unsafe { std::ptr::read_unaligned(p_data) };
        let presentation_copy = unsafe { data.p_presentation_info.as_ref().copied() };
        let owned = amd_anti_lag::AntiLagDataAMD {
            s_type: data.s_type,
            p_next: std::ptr::null(),
            mode: data.mode,
            max_fps: data.max_fps,
            p_presentation_info: presentation_copy
                .as_ref()
                .map_or(std::ptr::null(), |p| p as *const _),
        };

        if let Some(s) = ctx.strategy.get().and_then(|s| s.as_anti_lag()) {
            s.notify_update(&owned);
        }
    })
}
