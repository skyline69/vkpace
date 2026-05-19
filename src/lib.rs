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
// No dead code lurking. Anything unused gets removed, not allowed.
#![deny(dead_code)]

mod amd_anti_lag;
mod catch;
mod clock;
mod config;
mod delay_controller;
mod device;
mod dispatch;
mod entrypoints;
mod instance;
mod physical_device;
mod pnext;
mod present_wait;
mod queue;
mod registry;
mod strategy;
mod submission_span;
mod telemetry;
mod timestamp_pool;
mod vk_layer;

/// Narrow re-export surface for the `fuzz/` subcrate. Gated by the `fuzz`
/// Cargo feature so the shipped layer doesn't expose internals. Never
/// import this from production code.
#[cfg(feature = "fuzz")]
pub mod __fuzz_api {
    use std::ffi::c_void;

    /// Fuzz harness for `pnext::find`. Walks an arbitrary byte-shaped
    /// pNext chain — the harness fabricates a chain of `BaseHeader`s from
    /// the input bytes.
    ///
    /// # Safety
    /// `head` must outlive the call. Provided by the harness.
    pub unsafe fn pnext_find_any(head: *const c_void) -> bool {
        // Just exercise the walker; verifying we don't deref past the
        // chain or hit UB is the libfuzzer-side goal.
        unsafe {
            for (_, _) in crate::pnext::PNextIter::new(head) {
                std::hint::black_box(());
            }
        }
        true
    }

    /// Fuzz harness for the TOML loader. Writes `input` to a tempfile,
    /// sets `VKPACE_CONFIG`, calls `load_toml`. Detects panics / OOMs
    /// on any byte input.
    pub fn config_toml_load(input: &str) {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("vkpace-fuzz-{}.toml", std::process::id()));
        if std::fs::write(&path, input).is_err() {
            return;
        }
        // Safety: env access is process-global but fuzz binaries run
        // single-threaded.
        unsafe {
            std::env::set_var("VKPACE_CONFIG", &path);
        }
        let _ = crate::config::__fuzz_load_toml();
        let _ = std::fs::remove_file(&path);
    }
}

use ash::vk;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
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
    let prom_path = std::env::var("VKPACE_PROM_PATH")
        .ok()
        .map(std::path::PathBuf::from);
    telemetry::Telemetry::new(socket, stats_interval, prom_path)
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
        let mut enable_present_id = false;
        let mut enable_present_wait = false;
        if layer_enabled {
            for &req in REQUIRED_EXTENSIONS {
                if !enabled_exts.contains(&req) {
                    extra_ext_storage.push(req.as_ptr());
                }
            }
            // Opportunistic: enable present_id/present_wait when supported.
            // Unlocks real display-side completion time in the marker history.
            for opt in pd_ctx.supported_optionals.iter().copied() {
                if !enabled_exts.contains(&opt) {
                    extra_ext_storage.push(opt.as_ptr());
                }
                if opt == vk::KHR_PRESENT_ID_NAME {
                    enable_present_id = true;
                }
                if opt == vk::KHR_PRESENT_WAIT_NAME {
                    enable_present_wait = true;
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
            enable_present_id,
            enable_present_wait,
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
    unsafe { get_physical_device_properties2_dispatch(physical_device, p_properties, false) }
}

unsafe extern "system" fn get_physical_device_properties2_khr(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties2<'_>,
) {
    unsafe { get_physical_device_properties2_dispatch(physical_device, p_properties, true) }
}

unsafe fn get_physical_device_properties2_dispatch(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties2<'_>,
    khr: bool,
) {
    crate::catch::vk_void(|| {
        let Some(pd) = registry::PHYSICAL_DEVICES
            .get(&registry::key(physical_device))
            .map(|r| r.clone())
        else {
            return;
        };
        // Honor caller variant; fall back to the other slot only if the
        // requested PFN wasn't loaded.
        // Strict variant dispatch — no cross-variant fallback.
        let getp2 = if khr {
            pd.instance.fns.get_physical_device_properties2_khr
        } else {
            pd.instance.fns.get_physical_device_properties2
        };
        let Some(getp2) = getp2 else {
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
    unsafe { get_physical_device_features2_dispatch(physical_device, p_features, false) }
}

unsafe extern "system" fn get_physical_device_features2_khr(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures2<'_>,
) {
    unsafe { get_physical_device_features2_dispatch(physical_device, p_features, true) }
}

unsafe fn get_physical_device_features2_dispatch(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures2<'_>,
    khr: bool,
) {
    crate::catch::vk_void(|| {
        let Some(pd) = registry::PHYSICAL_DEVICES
            .get(&registry::key(physical_device))
            .map(|r| r.clone())
        else {
            return;
        };
        let getf2 = if khr {
            pd.instance.fns.get_physical_device_features2_khr
        } else {
            pd.instance.fns.get_physical_device_features2
        };
        let Some(getf2) = getf2 else {
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

#[repr(C)]
struct AppendPresentId {
    s_type: vk::StructureType,
    p_next: *const c_void,
    present_id: vk::Bool32,
}

#[repr(C)]
struct AppendPresentWait {
    s_type: vk::StructureType,
    p_next: *const c_void,
    present_wait: vk::Bool32,
}

#[derive(Default)]
struct FeatureAppends {
    sync2: Option<Box<AppendSync2>>,
    hqr: Option<Box<AppendHqr>>,
    present_id: Option<Box<AppendPresentId>>,
    present_wait: Option<Box<AppendPresentWait>>,
}

impl FeatureAppends {
    /// Link allocated structs into the chain, returning the new head.
    /// Caller must keep `self` alive until the downstream `vkCreateDevice`
    /// returns. Order is arbitrary — `pNext` is a set, not a sequence —
    /// but stable for diff readability.
    unsafe fn relink_into(&mut self, mut head: *const c_void) -> *const c_void {
        if let Some(b) = self.sync2.as_mut() {
            b.p_next = head;
            head = b.as_ref() as *const AppendSync2 as *const c_void;
        }
        if let Some(b) = self.hqr.as_mut() {
            b.p_next = head;
            head = b.as_ref() as *const AppendHqr as *const c_void;
        }
        if let Some(b) = self.present_id.as_mut() {
            b.p_next = head;
            head = b.as_ref() as *const AppendPresentId as *const c_void;
        }
        if let Some(b) = self.present_wait.as_mut() {
            b.p_next = head;
            head = b.as_ref() as *const AppendPresentWait as *const c_void;
        }
        head
    }
}

#[derive(Clone, Copy)]
struct ApiCaps {
    v_1_2: bool,
    v_1_3: bool,
    enable_present_id: bool,
    enable_present_wait: bool,
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

    // VK_KHR_present_id: enable when the physical device supports it AND
    // the caller didn't pre-supply a features struct. Same prefer-existing
    // pattern; flip flag in-place if already present.
    if caps.enable_present_id {
        if let Some(p) = unsafe {
            pnext::find_mut::<vk::PhysicalDevicePresentIdFeaturesKHR<'_>>(
                p_next_head,
                vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
            )
        } {
            unsafe { (*p).present_id = vk::TRUE };
        } else {
            tracing::debug!("appending VkPhysicalDevicePresentIdFeaturesKHR to pNext");
            appends.present_id = Some(Box::new(AppendPresentId {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
                p_next: std::ptr::null(),
                present_id: vk::TRUE,
            }));
        }
    }

    // VK_KHR_present_wait: same shape.
    if caps.enable_present_wait {
        if let Some(p) = unsafe {
            pnext::find_mut::<vk::PhysicalDevicePresentWaitFeaturesKHR<'_>>(
                p_next_head,
                vk::StructureType::PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
            )
        } {
            unsafe { (*p).present_wait = vk::TRUE };
        } else {
            tracing::debug!("appending VkPhysicalDevicePresentWaitFeaturesKHR to pNext");
            appends.present_wait = Some(Box::new(AppendPresentWait {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
                p_next: std::ptr::null(),
                present_wait: vk::TRUE,
            }));
        }
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
            get_physical_device_properties2_khr
        );
        e!(
            b"vkGetPhysicalDeviceFeatures2",
            get_physical_device_features2
        );
        e!(
            b"vkGetPhysicalDeviceFeatures2KHR",
            get_physical_device_features2_khr
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
    e!(b"vkQueueSubmit", entrypoints::queue::queue_submit);
    e!(b"vkQueueSubmit2", entrypoints::queue::queue_submit2);
    e!(b"vkQueueSubmit2KHR", entrypoints::queue::queue_submit2_khr);
    e!(b"vkQueuePresentKHR", entrypoints::queue::queue_present_khr);
    e!(
        b"vkCreateSwapchainKHR",
        entrypoints::swapchain::create_swapchain_khr
    );
    e!(
        b"vkDestroySwapchainKHR",
        entrypoints::swapchain::destroy_swapchain_khr
    );
    e!(
        b"vkAcquireNextImageKHR",
        entrypoints::swapchain::acquire_next_image_khr
    );
    e!(
        b"vkAcquireNextImage2KHR",
        entrypoints::swapchain::acquire_next_image2_khr
    );
    e!(
        b"vkAntiLagUpdateAMD",
        entrypoints::latency::anti_lag_update_amd
    );
    e!(b"vkLatencySleepNV", entrypoints::latency::latency_sleep_nv);
    e!(
        b"vkSetLatencySleepModeNV",
        entrypoints::latency::set_latency_sleep_mode_nv
    );
    e!(
        b"vkSetLatencyMarkerNV",
        entrypoints::latency::set_latency_marker_nv
    );
    e!(
        b"vkGetLatencyTimingsNV",
        entrypoints::latency::get_latency_timings_nv
    );
    e!(
        b"vkQueueNotifyOutOfBandNV",
        entrypoints::latency::queue_notify_out_of_band_nv
    );
    m
});
