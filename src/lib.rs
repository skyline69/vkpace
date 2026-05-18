//! # vkpace
//!
//! Vulkan implicit layer that reduces input latency by spoofing AMD Anti-Lag
//! or NVIDIA Reflex (`VK_NV_low_latency2`) on top of any compliant driver.
//!
//! Entrypoints are exposed as `VkPace_GetInstanceProcAddr` and
//! `VkPace_GetDeviceProcAddr`; the loader patches our function pointers
//! into the chain via these.

#![allow(clippy::missing_safety_doc)]
// Crate name `VkLayer_VKPACE_reduce_latency` is mandated by the Vulkan loader's
// `libVkLayer_*.so` convention — overrides the snake_case lint.
#![allow(non_snake_case)]
// Every raw deref / FFI call must be in an explicit `unsafe {}` block, even
// inside an `unsafe fn`. Catches accidental unsafe-creep into helper code.
#![deny(unsafe_op_in_unsafe_fn)]

mod amd_anti_lag;
mod catch;
mod clock;
mod config;
mod delay_controller;
mod device;
mod dispatch;
mod instance;
mod physical_device;
mod pnext;
mod queue;
mod registry;
mod strategy;
mod submission_span;
mod telemetry;
mod timestamp_pool;
mod vk_layer;

use ash::vk;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;

use crate::config::{LAYER_NAME, LayerConfig, NVIDIA_VENDOR_ID};
use crate::device::DeviceContext;
use crate::dispatch::{DeviceTable, InstanceTable};
use crate::instance::InstanceContext;
use crate::physical_device::{PhysicalDeviceContext, REQUIRED_EXTENSIONS};
use crate::queue::QueueContext;
use crate::vk_layer::{VkLayerDeviceCreateInfo, VkLayerFunction, VkLayerInstanceCreateInfo};

static CONFIG: Lazy<LayerConfig> = Lazy::new(LayerConfig::from_env);

pub(crate) static TELEMETRY: Lazy<telemetry::Telemetry> = Lazy::new(|| {
    let socket = std::env::var("VKPACE_TELEMETRY_SOCKET").ok();
    let stats_interval = std::env::var("VKPACE_STATS_INTERVAL")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    telemetry::Telemetry::new(socket, stats_interval)
});

fn init_tracing_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_env("VKPACE_LOG")
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            )
            .with_writer(std::io::stderr)
            .compact()
            .try_init();
    });
}

// ─── Entry points ──────────────────────────────────────────────────────────

/// Loader handshake: agree on the layer/loader ABI version and hand back
/// our `GetInstanceProcAddr`/`GetDeviceProcAddr` entrypoints. Loaders that
/// don't know about this entrypoint fall back to GIPA-only.
///
/// # Safety
/// Called by the Vulkan loader exactly once, with a valid
/// `VkNegotiateLayerInterface`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    p_version_struct: *mut vk_layer::VkNegotiateLayerInterface,
) -> vk::Result {
    init_tracing_once();
    if p_version_struct.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let req = unsafe { &mut *p_version_struct };
    let chosen = req
        .loader_layer_interface_version
        .min(vk_layer::CURRENT_LOADER_LAYER_INTERFACE_VERSION);
    req.loader_layer_interface_version = chosen;
    req.pfn_get_instance_proc_addr = Some(VkPace_GetInstanceProcAddr);
    req.pfn_get_device_proc_addr = Some(VkPace_GetDeviceProcAddr);
    req.pfn_get_physical_device_proc_addr = None;
    tracing::info!(version = chosen, "NegotiateLoaderLayerInterfaceVersion");
    vk::Result::SUCCESS
}

/// # Safety
/// Called by the Vulkan loader. `instance` may be null for global queries.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn VkPace_GetInstanceProcAddr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    crate::catch::vk_pfn(|| {
        init_tracing_once();
        if p_name.is_null() {
            return None;
        }
        let name = unsafe { CStr::from_ptr(p_name) };

        if let Some(f) = INSTANCE_FUNCTIONS.get(name.to_bytes()) {
            return *f;
        }

        if instance == vk::Instance::null() {
            return None;
        }
        let ctx = match registry::INSTANCES.get(&registry::key(instance)) {
            Some(c) => c.clone(),
            None => return None,
        };
        unsafe { (ctx.fns.get_instance_proc_addr)(instance, p_name) }
    })
}

/// # Safety
/// Called by the Vulkan loader.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn VkPace_GetDeviceProcAddr(
    device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    crate::catch::vk_pfn(|| {
        if p_name.is_null() || device == vk::Device::null() {
            return None;
        }
        let name = unsafe { CStr::from_ptr(p_name) };
        if let Some(f) = DEVICE_FUNCTIONS.get(name.to_bytes()) {
            return *f;
        }
        let ctx = match registry::DEVICES.get(&registry::key(device)) {
            Some(c) => c.clone(),
            None => return None,
        };
        unsafe { (ctx.fns.get_device_proc_addr)(device, p_name) }
    })
}

// ─── Instance lifecycle ────────────────────────────────────────────────────

unsafe extern "system" fn create_instance(
    p_create_info: *const vk::InstanceCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(create_info) = (unsafe { p_create_info.as_ref() }) else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };

        // Walk pNext for VkLayerInstanceCreateInfo with function=LAYER_LINK_INFO.
        let link = unsafe { find_instance_link(create_info.p_next) };
        let Some(link) = link else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let layer_info = unsafe { (*link).u.p_layer_info };
        if layer_info.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let gipa = unsafe { (*layer_info).pfn_next_get_instance_proc_addr };
        let Some(gipa) = gipa else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        // Advance the layer chain.
        unsafe {
            (*link).u.p_layer_info = (*layer_info).p_next;
        }

        let create_inst: vk::PFN_vkCreateInstance = unsafe {
            std::mem::transmute(gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr()))
        };
        let r = unsafe { create_inst(p_create_info, p_allocator, p_instance) };
        if r != vk::Result::SUCCESS {
            return r;
        }

        let instance = unsafe { *p_instance };
        let table = unsafe { InstanceTable::load(gipa, instance) };
        let (built, app_name) =
            unsafe { InstanceContext::new(instance, table, create_info, &CONFIG) };
        let ctx = Arc::new(built);
        tracing::info!(
            instance = ?ctx.handle,
            app = ?app_name,
            expose_reflex = ctx.config.expose_reflex,
            spoof_nvidia = ctx.config.spoof_nvidia,
            spoof_model = if ctx.config.spoof_nvidia {
                ctx.config.spoof_profile.device_name
            } else {
                "(off)"
            },
            force_decoupled = ctx.config.force_decoupled,
            fps_cap = ctx.config.fps_cap,
            decoupled = ctx.is_simulation_decoupled,
            "CreateInstance"
        );
        registry::INSTANCES.insert(registry::key(instance), ctx);

        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn destroy_instance(
    instance: vk::Instance,
    p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    crate::catch::vk_void(|| {
        let Some((_, ctx)) = registry::INSTANCES.remove(&registry::key(instance)) else {
            return;
        };
        for kv in ctx.physical_devices.iter() {
            registry::PHYSICAL_DEVICES.remove(kv.key());
        }
        let destroy = ctx.fns.destroy_instance;
        drop(ctx);
        unsafe { destroy(instance, p_allocator) };
    })
}

unsafe extern "system" fn enumerate_physical_devices(
    instance: vk::Instance,
    p_count: *mut u32,
    p_devices: *mut vk::PhysicalDevice,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(ctx) = registry::INSTANCES
            .get(&registry::key(instance))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let r = unsafe { (ctx.fns.enumerate_physical_devices)(instance, p_count, p_devices) };
        if r != vk::Result::SUCCESS || p_devices.is_null() || p_count.is_null() {
            return r;
        }
        let count = unsafe { *p_count } as usize;
        let slice = unsafe { std::slice::from_raw_parts(p_devices, count) };
        for &pd in slice {
            let key = registry::key(pd);
            registry::PHYSICAL_DEVICES.entry(key).or_insert_with(|| {
                Arc::new(unsafe { PhysicalDeviceContext::new(ctx.clone(), pd) })
            });
            if let Some(p) = registry::PHYSICAL_DEVICES.get(&key) {
                ctx.physical_devices.insert(key, p.clone());
            }
        }
        vk::Result::SUCCESS
    })
}

// ─── Device lifecycle ──────────────────────────────────────────────────────

unsafe extern "system" fn create_device(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_device: *mut vk::Device,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(create_info) = (unsafe { p_create_info.as_ref() }) else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let Some(pd_ctx) = registry::PHYSICAL_DEVICES
            .get(&registry::key(physical_device))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };

        let enabled_exts: Vec<&CStr> = unsafe {
            std::slice::from_raw_parts(
                create_info.pp_enabled_extension_names,
                create_info.enabled_extension_count as usize,
            )
        }
        .iter()
        .map(|&p| unsafe { CStr::from_ptr(p) })
        .collect();

        let cfg = &pd_ctx.instance.config;
        let watched_ext: &CStr = if !cfg.expose_reflex {
            amd_anti_lag::AMD_ANTI_LAG_NAME
        } else {
            vk::NV_LOW_LATENCY2_NAME
        };
        let layer_enabled = enabled_exts.contains(&watched_ext);
        if layer_enabled && !pd_ctx.supports_required_extensions {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }

        let link = unsafe { find_device_link(create_info.p_next) };
        let Some(link) = link else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let layer_info = unsafe { (*link).u.p_layer_info };
        if layer_info.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let gipa = unsafe { (*layer_info).pfn_next_get_instance_proc_addr };
        let gdpa = unsafe { (*layer_info).pfn_next_get_device_proc_addr };
        let (Some(gipa), Some(gdpa)) = (gipa, gdpa) else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        unsafe {
            (*link).u.p_layer_info = (*layer_info).p_next;
        }

        // Build the next-create-info: original layout, but with extra extensions
        // and feature-struct mutations when our layer is enabled.
        let mut extra_ext_storage: Vec<*const c_char> =
            enabled_exts.iter().map(|c| c.as_ptr()).collect();
        if layer_enabled {
            for &req in REQUIRED_EXTENSIONS {
                if !enabled_exts.contains(&req) {
                    extra_ext_storage.push(req.as_ptr());
                }
            }
        }

        // Feature-struct handling. First flip any existing sync2/hostQueryReset
        // flags in the caller's chain; for missing ones we allocate fresh Append*
        // structs (kept alive for the duration of this fn) and prepend them.
        // Skip the Vulkan13Features branch on pre-1.3 instances etc. — passing a
        // struct the loader doesn't know about is a spec violation.
        let mut feature_appends = FeatureAppends::default();
        let api_caps = ApiCaps {
            v_1_2: pd_ctx.instance.supports_1_2(),
            v_1_3: pd_ctx.instance.supports_1_3(),
        };
        let mut p_next_head: *const c_void = create_info.p_next;
        if layer_enabled {
            unsafe {
                patch_features(p_next_head as *mut c_void, &mut feature_appends, api_caps);
                p_next_head = feature_appends.relink_into(p_next_head);
            }
        }

        let mut next_info = *create_info;
        next_info.p_next = p_next_head;
        next_info.pp_enabled_extension_names = extra_ext_storage.as_ptr();
        next_info.enabled_extension_count = extra_ext_storage.len() as u32;

        let create_dev: vk::PFN_vkCreateDevice =
            unsafe { std::mem::transmute(gipa(vk::Instance::null(), c"vkCreateDevice".as_ptr())) };
        let r = unsafe { create_dev(physical_device, &next_info, p_allocator, p_device) };
        if r != vk::Result::SUCCESS {
            return r;
        }

        let device = unsafe { *p_device };
        let table = unsafe { DeviceTable::load(gdpa, device) };
        let ctx = DeviceContext::new(pd_ctx, device, table, layer_enabled);
        tracing::info!(
            device = ?ctx.handle,
            layer_enabled,
            "CreateDevice"
        );
        registry::DEVICES.insert(registry::key(device), ctx);

        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn destroy_device(
    device: vk::Device,
    p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    crate::catch::vk_void(|| {
        let Some((_, ctx)) = registry::DEVICES.remove(&registry::key(device)) else {
            return;
        };
        // Drain in-flight GPU work + drop queues (and their timestamp/command
        // pools) *before* we call vkDestroyDevice. Otherwise the reaper thread
        // inside TimestampPool can race with device destruction.
        ctx.drain_for_destroy();
        for kv in ctx.queues.iter() {
            registry::QUEUES.remove(kv.key());
        }
        let destroy = ctx.fns.destroy_device;
        drop(ctx);
        unsafe { destroy(device, p_allocator) };
    })
}

unsafe extern "system" fn get_device_queue(
    device: vk::Device,
    family: u32,
    index: u32,
    p_queue: *mut vk::Queue,
) {
    crate::catch::vk_void(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return;
        };
        unsafe { (ctx.fns.get_device_queue)(device, family, index, p_queue) };
        let queue = unsafe { *p_queue };
        if queue == vk::Queue::null() {
            return;
        }
        register_queue(ctx, queue, family);
    })
}

unsafe extern "system" fn get_device_queue2(
    device: vk::Device,
    info: *const vk::DeviceQueueInfo2<'_>,
    p_queue: *mut vk::Queue,
) {
    crate::catch::vk_void(|| {
        let Some(ctx) = registry::DEVICES
            .get(&registry::key(device))
            .map(|r| r.clone())
        else {
            return;
        };
        let Some(get_q2) = ctx.fns.get_device_queue2 else {
            return;
        };
        unsafe { get_q2(device, info, p_queue) };
        let queue = unsafe { *p_queue };
        if queue == vk::Queue::null() {
            return;
        }
        let family = unsafe { (*info).queue_family_index };
        register_queue(ctx, queue, family);
    })
}

/// Whether the given submit should be wrapped with timestamp CBs.
///
/// - LL2 (`expose_reflex`): inject every graphics submit. The strategy
///   keys submission spans by the optional `VkLatencySubmissionPresentIdNV`
///   (defaulting to 0 when absent); present-side matches by
///   `VkPresentIdKHR`. We can't know which submit "matters" until present,
///   so we keep them all. Matches the C++ reference.
/// - AntiLag: only while AntiLag tracking is active (input→present window).
///   In the dormant window we'd just do work the strategy throws away.
unsafe fn should_inject_for_submit(qctx: &QueueContext, _p_next: *const c_void) -> bool {
    if !qctx.should_inject_timestamps() {
        return false;
    }
    if qctx.device.instance.config.expose_reflex {
        true
    } else {
        qctx.device
            .strategy
            .get()
            .and_then(|s| s.as_anti_lag())
            .is_some_and(|s| s.should_track_submissions())
    }
}

fn register_queue(device: Arc<DeviceContext>, queue: vk::Queue, family: u32) {
    let key = registry::key(queue);
    if registry::QUEUES.contains_key(&key) {
        return;
    }
    let Some(qctx) = (unsafe { QueueContext::new(device.clone(), family) }) else {
        tracing::warn!(?queue, family, "QueueContext init failed");
        return;
    };
    tracing::debug!(?queue, family, "register queue");
    registry::QUEUES.insert(key, qctx.clone());
    device.queues.insert(key, qctx);
}

// ─── Submit / present ──────────────────────────────────────────────────────

// Thread-local fallback storage for the QueueSubmit injection path. The
// SmallVec inline capacities cover the typical 1-4-submit / ≤8-CB hot path
// without heap allocation; for the (rare) spill case we hand the spilled
// `Vec` back to the thread-local arena instead of dropping it, so the next
// call on the same thread re-uses the buffer.
mod submit_arena {
    use ash::vk;
    use std::cell::RefCell;

    thread_local! {
        static SUBMIT1_CBS: RefCell<Vec<Vec<vk::CommandBuffer>>> = const { RefCell::new(Vec::new()) };
        static SUBMIT2_CBS: RefCell<Vec<Vec<vk::CommandBufferSubmitInfo<'static>>>> =
            const { RefCell::new(Vec::new()) };
    }

    pub fn take_submit1() -> Vec<vk::CommandBuffer> {
        SUBMIT1_CBS.with(|c| c.borrow_mut().pop().unwrap_or_default())
    }

    pub fn give_submit1(mut v: Vec<vk::CommandBuffer>) {
        v.clear();
        SUBMIT1_CBS.with(|c| c.borrow_mut().push(v));
    }

    pub fn take_submit2() -> Vec<vk::CommandBufferSubmitInfo<'static>> {
        SUBMIT2_CBS.with(|c| c.borrow_mut().pop().unwrap_or_default())
    }

    pub fn give_submit2(mut v: Vec<vk::CommandBufferSubmitInfo<'static>>) {
        v.clear();
        SUBMIT2_CBS.with(|c| c.borrow_mut().push(v));
    }
}

unsafe extern "system" fn queue_submit(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo<'_>,
    fence: vk::Fence,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(qctx) = registry::QUEUES
            .get(&registry::key(queue))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let fns = &qctx.device.fns;
        if submit_count == 0 {
            return unsafe { (fns.queue_submit)(queue, submit_count, p_submits, fence) };
        }
        let submits = unsafe { std::slice::from_raw_parts(p_submits, submit_count as usize) };

        // Per-submit decision. Skip injection on submits the strategy
        // wouldn't track anyway. Fast-path: if NO submit needs injection,
        // forward the caller's array verbatim — zero allocations.
        let inject_mask: SmallVec<[bool; 4]> = submits
            .iter()
            .map(|s| unsafe { should_inject_for_submit(&qctx, s.p_next) })
            .collect();
        if inject_mask.iter().all(|b| !b) {
            let r = unsafe { (fns.queue_submit)(queue, submit_count, p_submits, fence) };
            TELEMETRY.counters.record_submit_call(submit_count, 0);
            return r;
        }

        let mut handles: SmallVec<[Option<Arc<crate::timestamp_pool::Handle>>; 4]> =
            SmallVec::with_capacity(submits.len());
        let mut all_cbs: SmallVec<[Vec<vk::CommandBuffer>; 4]> =
            SmallVec::with_capacity(submits.len());
        let mut next_submits: SmallVec<[vk::SubmitInfo<'_>; 4]> =
            SmallVec::with_capacity(submits.len());

        for (submit, &inject) in submits.iter().zip(inject_mask.iter()) {
            if !inject {
                next_submits.push(*submit);
                handles.push(None);
                continue;
            }
            let Some(handle) = qctx.timestamp_pool.acquire() else {
                return unsafe { (fns.queue_submit)(queue, submit_count, p_submits, fence) };
            };
            let mut cbs = submit_arena::take_submit1();
            cbs.reserve(submit.command_buffer_count as usize + 2);
            cbs.push(handle.start_buffer());
            cbs.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    submit.p_command_buffers,
                    submit.command_buffer_count as usize,
                )
            });
            cbs.push(handle.end_buffer());
            let mut next = *submit;
            next.command_buffer_count = cbs.len() as u32;
            next.p_command_buffers = cbs.as_ptr();
            all_cbs.push(cbs);
            next_submits.push(next);
            handles.push(Some(handle));
        }

        let r = unsafe {
            (fns.queue_submit)(
                queue,
                next_submits.len() as u32,
                next_submits.as_ptr(),
                fence,
            )
        };
        for v in all_cbs.drain(..) {
            submit_arena::give_submit1(v);
        }
        if r != vk::Result::SUCCESS {
            return r;
        }
        let mut injected_count = 0u32;
        for (submit, handle) in submits.iter().zip(handles) {
            let Some(handle) = handle else { continue };
            injected_count += 1;
            handle
                .was_submitted
                .store(true, std::sync::atomic::Ordering::Relaxed);
            unsafe { qctx.strategy.notify_submit(submit.p_next, handle) };
        }
        TELEMETRY
            .counters
            .record_submit_call(submit_count, injected_count);
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn queue_submit2(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo2<'_>,
    fence: vk::Fence,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(qctx) = registry::QUEUES
            .get(&registry::key(queue))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let fns = &qctx.device.fns;
        let Some(submit2) = fns.queue_submit2 else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if submit_count == 0 {
            return unsafe { submit2(queue, submit_count, p_submits, fence) };
        }
        let submits = unsafe { std::slice::from_raw_parts(p_submits, submit_count as usize) };

        let inject_mask: SmallVec<[bool; 4]> = submits
            .iter()
            .map(|s| unsafe { should_inject_for_submit(&qctx, s.p_next) })
            .collect();
        if inject_mask.iter().all(|b| !b) {
            let r = unsafe { submit2(queue, submit_count, p_submits, fence) };
            TELEMETRY.counters.record_submit_call(submit_count, 0);
            return r;
        }

        let mut handles: SmallVec<[Option<Arc<crate::timestamp_pool::Handle>>; 4]> =
            SmallVec::with_capacity(submits.len());
        let mut all_cbs: SmallVec<[Vec<vk::CommandBufferSubmitInfo<'static>>; 4]> =
            SmallVec::with_capacity(submits.len());
        let mut next_submits: SmallVec<[vk::SubmitInfo2<'_>; 4]> =
            SmallVec::with_capacity(submits.len());

        for (submit, &inject) in submits.iter().zip(inject_mask.iter()) {
            if !inject {
                next_submits.push(*submit);
                handles.push(None);
                continue;
            }
            let Some(handle) = qctx.timestamp_pool.acquire() else {
                return unsafe { submit2(queue, submit_count, p_submits, fence) };
            };
            let start_cb: vk::CommandBufferSubmitInfo<'static> =
                vk::CommandBufferSubmitInfo::default().command_buffer(handle.start_buffer());
            let end_cb: vk::CommandBufferSubmitInfo<'static> =
                vk::CommandBufferSubmitInfo::default().command_buffer(handle.end_buffer());

            let mut cbs = submit_arena::take_submit2();
            cbs.reserve(submit.command_buffer_info_count as usize + 2);
            cbs.push(start_cb);
            let user_slice = unsafe {
                std::slice::from_raw_parts(
                    submit.p_command_buffer_infos,
                    submit.command_buffer_info_count as usize,
                )
            };
            cbs.extend(user_slice.iter().map(|s| {
                // SAFETY: same struct layout; only widening the lifetime
                // parameter for arena storage we control.
                unsafe {
                    std::mem::transmute::<
                        vk::CommandBufferSubmitInfo<'_>,
                        vk::CommandBufferSubmitInfo<'static>,
                    >(*s)
                }
            }));
            cbs.push(end_cb);
            let mut next = *submit;
            next.command_buffer_info_count = cbs.len() as u32;
            next.p_command_buffer_infos = cbs.as_ptr().cast::<vk::CommandBufferSubmitInfo<'_>>();
            all_cbs.push(cbs);
            next_submits.push(next);
            handles.push(Some(handle));
        }

        let r = unsafe {
            submit2(
                queue,
                next_submits.len() as u32,
                next_submits.as_ptr(),
                fence,
            )
        };
        for v in all_cbs.drain(..) {
            submit_arena::give_submit2(v);
        }
        if r != vk::Result::SUCCESS {
            return r;
        }
        let mut injected_count = 0u32;
        for (submit, handle) in submits.iter().zip(handles) {
            let Some(handle) = handle else { continue };
            injected_count += 1;
            handle
                .was_submitted
                .store(true, std::sync::atomic::Ordering::Relaxed);
            unsafe { qctx.strategy.notify_submit(submit.p_next, handle) };
        }
        TELEMETRY
            .counters
            .record_submit_call(submit_count, injected_count);
        vk::Result::SUCCESS
    })
}

unsafe extern "system" fn queue_present_khr(
    queue: vk::Queue,
    p_present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(qctx) = registry::QUEUES
            .get(&registry::key(queue))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let fns = &qctx.device.fns;
        let Some(present_khr) = fns.queue_present_khr else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let r = unsafe { present_khr(queue, p_present_info) };
        TELEMETRY.counters.inc_present();
        TELEMETRY.push_record(telemetry::FrameRecord {
            host_ns: clock::DeviceClock::now(),
            frame_index: TELEMETRY
                .counters
                .frames
                .load(std::sync::atomic::Ordering::Relaxed),
            queue_id: registry::key(queue),
        });

        if let Some(info) = unsafe { p_present_info.as_ref() } {
            qctx.strategy.notify_present(info);
            if let Some(s) = qctx.device.strategy.get().and_then(|s| s.as_low_latency2()) {
                strategy::low_latency2::forward_present(s, info);
            }
        }
        r
    })
}

// ─── Swapchain ─────────────────────────────────────────────────────────────

unsafe extern "system" fn create_swapchain_khr(
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

unsafe extern "system" fn acquire_next_image_khr(
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
        TELEMETRY
            .counters
            .acquires
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if elapsed > 16_000_000 {
            tracing::debug!(swapchain = ?swapchain, elapsed_us = elapsed / 1_000, "long acquire");
        }
        r
    })
}

unsafe extern "system" fn acquire_next_image2_khr(
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
        TELEMETRY
            .counters
            .acquires
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if elapsed > 16_000_000 {
            tracing::debug!(elapsed_us = elapsed / 1_000, "long acquire2");
        }
        r
    })
}

unsafe extern "system" fn destroy_swapchain_khr(
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
        if let Some(destroy) = ctx.fns.destroy_swapchain_khr {
            unsafe { destroy(device, swapchain, p_allocator) };
        }
    })
}

// ─── Spoofing / feature exposure ───────────────────────────────────────────

unsafe extern "system" fn get_physical_device_properties(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties,
) {
    crate::catch::vk_void(|| {
        let Some(pd) = registry::PHYSICAL_DEVICES
            .get(&registry::key(physical_device))
            .map(|r| r.clone())
        else {
            return;
        };
        unsafe { (pd.instance.fns.get_physical_device_properties)(physical_device, p_properties) };
        if pd.instance.config.spoof_nvidia {
            unsafe {
                apply_nvidia_spoof_v1(&pd.instance.config, &mut *p_properties);
            }
        }
    })
}

unsafe extern "system" fn get_physical_device_properties2(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties2<'_>,
) {
    crate::catch::vk_void(|| {
        let Some(pd) = registry::PHYSICAL_DEVICES
            .get(&registry::key(physical_device))
            .map(|r| r.clone())
        else {
            return;
        };
        let Some(getp2) = pd.instance.fns.get_physical_device_properties2 else {
            return;
        };
        unsafe { getp2(physical_device, p_properties) };
        if pd.instance.config.spoof_nvidia {
            unsafe {
                apply_nvidia_spoof_v1(&pd.instance.config, &mut (*p_properties).properties);
            }
        }
    })
}

unsafe fn apply_nvidia_spoof_v1(cfg: &LayerConfig, props: &mut vk::PhysicalDeviceProperties) {
    let profile = cfg.spoof_profile;
    props.vendor_id = NVIDIA_VENDOR_ID;
    props.device_id = profile.device_id;
    let bytes = profile.device_name.as_bytes();
    let len = bytes.len().min(props.device_name.len() - 1);
    for (i, &b) in bytes.iter().take(len).enumerate() {
        props.device_name[i] = b as c_char;
    }
    props.device_name[len] = 0;
}

unsafe extern "system" fn get_physical_device_features2(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures2<'_>,
) {
    crate::catch::vk_void(|| {
        let Some(pd) = registry::PHYSICAL_DEVICES
            .get(&registry::key(physical_device))
            .map(|r| r.clone())
        else {
            return;
        };
        let Some(getf2) = pd.instance.fns.get_physical_device_features2 else {
            return;
        };
        unsafe { getf2(physical_device, p_features) };
        if pd.instance.config.expose_reflex {
            return; // VK_NV_low_latency2 advertises through SurfaceCaps, not features.
        }
        let alf = unsafe {
            pnext::find_mut::<amd_anti_lag::PhysicalDeviceAntiLagFeaturesAMD>(
                (*p_features).p_next,
                amd_anti_lag::STRUCTURE_TYPE_PHYSICAL_DEVICE_ANTI_LAG_FEATURES_AMD,
            )
        };
        if let Some(p) = alf {
            unsafe {
                (*p).anti_lag = if pd.supports_required_extensions {
                    vk::TRUE
                } else {
                    vk::FALSE
                };
            }
        }
    })
}

unsafe extern "system" fn enumerate_device_extension_properties(
    physical_device: vk::PhysicalDevice,
    p_layer_name: *const c_char,
    p_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(pd) = registry::PHYSICAL_DEVICES
            .get(&registry::key(physical_device))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };

        let layer_filter = unsafe { p_layer_name.as_ref().map(|_| CStr::from_ptr(p_layer_name)) };
        let is_for_us = layer_filter
            .map(|n| n.to_bytes() == LAYER_NAME.as_bytes())
            .unwrap_or(false);
        if let Some(name) = layer_filter
            && !is_for_us
        {
            // Filtered to a different layer — straight pass-through.
            let _ = name;
            return unsafe {
                (pd.instance.fns.enumerate_device_extension_properties)(
                    physical_device,
                    p_layer_name,
                    p_count,
                    p_properties,
                )
            };
        }

        let our_ext = if pd.instance.config.expose_reflex {
            (vk::NV_LOW_LATENCY2_NAME, vk::NV_LOW_LATENCY2_SPEC_VERSION)
        } else {
            (
                amd_anti_lag::AMD_ANTI_LAG_NAME,
                amd_anti_lag::AMD_ANTI_LAG_SPEC_VERSION,
            )
        };
        let ext_prop = make_ext_prop(our_ext.0, our_ext.1);

        if is_for_us {
            if p_properties.is_null() {
                unsafe { *p_count = 1 };
                return vk::Result::SUCCESS;
            }
            if unsafe { *p_count } == 0 {
                return vk::Result::INCOMPLETE;
            }
            unsafe {
                *p_properties = ext_prop;
                *p_count = 1;
            }
            return vk::Result::SUCCESS;
        }

        // Underlying enumeration + merge.
        let mut underlying_count = 0u32;
        let r = unsafe {
            (pd.instance.fns.enumerate_device_extension_properties)(
                physical_device,
                std::ptr::null(),
                &mut underlying_count,
                std::ptr::null_mut(),
            )
        };
        if r != vk::Result::SUCCESS {
            return r;
        }
        let mut buf = vec![vk::ExtensionProperties::default(); underlying_count as usize];
        let r = unsafe {
            (pd.instance.fns.enumerate_device_extension_properties)(
                physical_device,
                std::ptr::null(),
                &mut underlying_count,
                buf.as_mut_ptr(),
            )
        };
        if r != vk::Result::SUCCESS {
            return r;
        }
        buf.truncate(underlying_count as usize);

        let already = buf.iter().any(|e| ext_name_eq(e, our_ext.0));
        let total = underlying_count + (!already) as u32;

        if p_properties.is_null() {
            unsafe { *p_count = total };
            return vk::Result::SUCCESS;
        }

        let cap = unsafe { *p_count };
        let to_copy = cap.min(underlying_count) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), p_properties, to_copy);
        }
        let mut written = to_copy as u32;
        if !already && written < cap {
            unsafe { *p_properties.add(written as usize) = ext_prop };
            written += 1;
        }
        unsafe { *p_count = written };
        if written < total {
            vk::Result::INCOMPLETE
        } else {
            vk::Result::SUCCESS
        }
    })
}

unsafe extern "system" fn get_physical_device_surface_capabilities2_khr(
    physical_device: vk::PhysicalDevice,
    p_surface_info: *const vk::PhysicalDeviceSurfaceInfo2KHR<'_>,
    p_surface_capabilities: *mut vk::SurfaceCapabilities2KHR<'_>,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(pd) = registry::PHYSICAL_DEVICES
            .get(&registry::key(physical_device))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let Some(get) = pd
            .instance
            .fns
            .get_physical_device_surface_capabilities2_khr
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let r = unsafe { get(physical_device, p_surface_info, p_surface_capabilities) };
        if r != vk::Result::SUCCESS || !pd.instance.config.expose_reflex {
            return r;
        }
        let head = unsafe { (*p_surface_capabilities).p_next };
        let lsc = unsafe {
            pnext::find_mut::<vk::LatencySurfaceCapabilitiesNV<'_>>(
                head,
                vk::StructureType::LATENCY_SURFACE_CAPABILITIES_NV,
            )
        };
        let Some(lsc) = lsc else { return r };
        let supported = [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO,
        ];
        unsafe {
            if (*lsc).p_present_modes.is_null() {
                (*lsc).present_mode_count = supported.len() as u32;
            } else {
                let cap = (*lsc).present_mode_count.min(supported.len() as u32);
                std::ptr::copy_nonoverlapping(
                    supported.as_ptr(),
                    (*lsc).p_present_modes,
                    cap as usize,
                );
                (*lsc).present_mode_count = cap;
            }
        }
        r
    })
}

// ─── NV Low-Latency2 entrypoints ───────────────────────────────────────────

unsafe extern "system" fn latency_sleep_nv(
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

unsafe extern "system" fn set_latency_sleep_mode_nv(
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

unsafe extern "system" fn set_latency_marker_nv(
    _device: vk::Device,
    _swapchain: vk::SwapchainKHR,
    _info: *const vk::SetLatencyMarkerInfoNV<'_>,
) {
    crate::catch::vk_void(|| {})
}

unsafe extern "system" fn get_latency_timings_nv(
    _device: vk::Device,
    _swapchain: vk::SwapchainKHR,
    timings: *mut vk::GetLatencyMarkerInfoNV<'_>,
) {
    crate::catch::vk_void(|| {
        if !timings.is_null() {
            unsafe { (*timings).timing_count = 0 };
        }
    })
}

unsafe extern "system" fn queue_notify_out_of_band_nv(
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

unsafe extern "system" fn anti_lag_update_amd(
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

// ─── pNext walk helpers (Vulkan layer-link infos) ──────────────────────────

unsafe fn find_instance_link(mut p: *const c_void) -> Option<*mut VkLayerInstanceCreateInfo> {
    while !p.is_null() {
        let head = p as *const vk_layer::Header;
        if unsafe { (*head).s_type } == vk::StructureType::LOADER_INSTANCE_CREATE_INFO {
            let info = p as *mut VkLayerInstanceCreateInfo;
            if unsafe { (*info).function } == VkLayerFunction::LINK_INFO {
                return Some(info);
            }
        }
        p = unsafe { (*head).p_next };
    }
    None
}

unsafe fn find_device_link(mut p: *const c_void) -> Option<*mut VkLayerDeviceCreateInfo> {
    while !p.is_null() {
        let head = p as *const vk_layer::Header;
        if unsafe { (*head).s_type } == vk::StructureType::LOADER_DEVICE_CREATE_INFO {
            let info = p as *mut VkLayerDeviceCreateInfo;
            if unsafe { (*info).function } == VkLayerFunction::LINK_INFO {
                return Some(info);
            }
        }
        p = unsafe { (*head).p_next };
    }
    None
}

/// Layout-compatible mirrors of the relevant feature structs. Defined here
/// so we can heap-allocate one and splice it into the device-create pNext
/// chain. The first three fields are the same in every Vulkan feature struct
/// (sType + pNext + Bool32) so we can splice without using ash's typestate.
#[repr(C)]
struct AppendSync2 {
    s_type: vk::StructureType,
    p_next: *const c_void,
    synchronization2: vk::Bool32,
}

#[repr(C)]
struct AppendHqr {
    s_type: vk::StructureType,
    p_next: *const c_void,
    host_query_reset: vk::Bool32,
}

#[derive(Default)]
struct FeatureAppends {
    sync2: Option<Box<AppendSync2>>,
    hqr: Option<Box<AppendHqr>>,
}

impl FeatureAppends {
    /// Link allocated structs into the chain, returning the new head.
    /// Ordering: hqr → sync2 → previous head. Caller must keep `self` alive
    /// until the downstream `vkCreateDevice` returns.
    unsafe fn relink_into(&mut self, mut head: *const c_void) -> *const c_void {
        if let Some(b) = self.sync2.as_mut() {
            b.p_next = head;
            head = b.as_ref() as *const AppendSync2 as *const c_void;
        }
        if let Some(b) = self.hqr.as_mut() {
            b.p_next = head;
            head = b.as_ref() as *const AppendHqr as *const c_void;
        }
        head
    }
}

#[derive(Clone, Copy)]
struct ApiCaps {
    v_1_2: bool,
    v_1_3: bool,
}

unsafe fn patch_features(p_next_head: *mut c_void, appends: &mut FeatureAppends, caps: ApiCaps) {
    // sync2: prefer the Vulkan13Features knob (only on instances ≥1.3),
    // else the dedicated Sync2Features struct, else append.
    let mut sync2_present = false;
    if caps.v_1_3
        && let Some(p) = unsafe {
            pnext::find_mut::<vk::PhysicalDeviceVulkan13Features<'_>>(
                p_next_head,
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            )
        }
    {
        unsafe { (*p).synchronization2 = vk::TRUE };
        sync2_present = true;
    }
    if !sync2_present
        && let Some(p) = unsafe {
            pnext::find_mut::<vk::PhysicalDeviceSynchronization2Features<'_>>(
                p_next_head,
                vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            )
        }
    {
        unsafe { (*p).synchronization2 = vk::TRUE };
        sync2_present = true;
    }
    if !sync2_present {
        tracing::debug!("appending VkPhysicalDeviceSynchronization2Features to pNext");
        appends.sync2 = Some(Box::new(AppendSync2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            p_next: std::ptr::null(),
            synchronization2: vk::TRUE,
        }));
    }

    // hostQueryReset: same shape, but Vulkan12Features only valid on 1.2+.
    let mut hqr_present = false;
    if caps.v_1_2
        && let Some(p) = unsafe {
            pnext::find_mut::<vk::PhysicalDeviceVulkan12Features<'_>>(
                p_next_head,
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            )
        }
    {
        unsafe { (*p).host_query_reset = vk::TRUE };
        hqr_present = true;
    }
    if !hqr_present
        && let Some(p) = unsafe {
            pnext::find_mut::<vk::PhysicalDeviceHostQueryResetFeatures<'_>>(
                p_next_head,
                vk::StructureType::PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
            )
        }
    {
        unsafe { (*p).host_query_reset = vk::TRUE };
        hqr_present = true;
    }
    if !hqr_present {
        tracing::debug!("appending VkPhysicalDeviceHostQueryResetFeatures to pNext");
        appends.hqr = Some(Box::new(AppendHqr {
            s_type: vk::StructureType::PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
            p_next: std::ptr::null(),
            host_query_reset: vk::TRUE,
        }));
    }
}

fn make_ext_prop(name: &CStr, spec_version: u32) -> vk::ExtensionProperties {
    let mut out = vk::ExtensionProperties::default();
    let bytes = name.to_bytes();
    let len = bytes.len().min(out.extension_name.len() - 1);
    for (i, &b) in bytes.iter().take(len).enumerate() {
        out.extension_name[i] = b as c_char;
    }
    out.spec_version = spec_version;
    out
}

fn ext_name_eq(prop: &vk::ExtensionProperties, name: &CStr) -> bool {
    let end = prop
        .extension_name
        .iter()
        .position(|c| *c == 0)
        .unwrap_or(prop.extension_name.len());
    let bytes =
        unsafe { std::slice::from_raw_parts(prop.extension_name.as_ptr().cast::<u8>(), end) };
    bytes == name.to_bytes()
}

// ─── Function tables ───────────────────────────────────────────────────────

static INSTANCE_FUNCTIONS: Lazy<FxHashMap<&'static [u8], vk::PFN_vkVoidFunction>> =
    Lazy::new(|| {
        use std::mem::transmute;
        let mut m = FxHashMap::<&'static [u8], vk::PFN_vkVoidFunction>::default();
        macro_rules! e {
            ($lit:literal, $f:expr) => {
                m.insert($lit, unsafe {
                    transmute::<*const (), vk::PFN_vkVoidFunction>($f as *const ())
                });
            };
        }
        e!(b"vkGetInstanceProcAddr", VkPace_GetInstanceProcAddr);
        e!(b"vkGetDeviceProcAddr", VkPace_GetDeviceProcAddr);
        e!(b"vkCreateInstance", create_instance);
        e!(b"vkDestroyInstance", destroy_instance);
        e!(b"vkEnumeratePhysicalDevices", enumerate_physical_devices);
        e!(b"vkCreateDevice", create_device);
        e!(
            b"vkEnumerateDeviceExtensionProperties",
            enumerate_device_extension_properties
        );
        e!(
            b"vkGetPhysicalDeviceProperties",
            get_physical_device_properties
        );
        e!(
            b"vkGetPhysicalDeviceProperties2",
            get_physical_device_properties2
        );
        e!(
            b"vkGetPhysicalDeviceProperties2KHR",
            get_physical_device_properties2
        );
        e!(
            b"vkGetPhysicalDeviceFeatures2",
            get_physical_device_features2
        );
        e!(
            b"vkGetPhysicalDeviceFeatures2KHR",
            get_physical_device_features2
        );
        e!(
            b"vkGetPhysicalDeviceSurfaceCapabilities2KHR",
            get_physical_device_surface_capabilities2_khr
        );
        m
    });

static DEVICE_FUNCTIONS: Lazy<FxHashMap<&'static [u8], vk::PFN_vkVoidFunction>> = Lazy::new(|| {
    use std::mem::transmute;
    let mut m = FxHashMap::<&'static [u8], vk::PFN_vkVoidFunction>::default();
    macro_rules! e {
        ($lit:literal, $f:expr) => {
            m.insert($lit, unsafe {
                transmute::<*const (), vk::PFN_vkVoidFunction>($f as *const ())
            });
        };
    }
    e!(b"vkGetDeviceProcAddr", VkPace_GetDeviceProcAddr);
    e!(b"vkDestroyDevice", destroy_device);
    e!(b"vkGetDeviceQueue", get_device_queue);
    e!(b"vkGetDeviceQueue2", get_device_queue2);
    e!(b"vkQueueSubmit", queue_submit);
    e!(b"vkQueueSubmit2", queue_submit2);
    e!(b"vkQueueSubmit2KHR", queue_submit2);
    e!(b"vkQueuePresentKHR", queue_present_khr);
    e!(b"vkCreateSwapchainKHR", create_swapchain_khr);
    e!(b"vkDestroySwapchainKHR", destroy_swapchain_khr);
    e!(b"vkAcquireNextImageKHR", acquire_next_image_khr);
    e!(b"vkAcquireNextImage2KHR", acquire_next_image2_khr);
    e!(b"vkAntiLagUpdateAMD", anti_lag_update_amd);
    e!(b"vkLatencySleepNV", latency_sleep_nv);
    e!(b"vkSetLatencySleepModeNV", set_latency_sleep_mode_nv);
    e!(b"vkSetLatencyMarkerNV", set_latency_marker_nv);
    e!(b"vkGetLatencyTimingsNV", get_latency_timings_nv);
    e!(b"vkQueueNotifyOutOfBandNV", queue_notify_out_of_band_nv);
    m
});
