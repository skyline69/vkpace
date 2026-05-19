use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

use crate::instance::InstanceContext;

pub const REQUIRED_EXTENSIONS: &[&CStr] = &[
    vk::KHR_SYNCHRONIZATION2_NAME,
    vk::KHR_CALIBRATED_TIMESTAMPS_NAME,
    vk::EXT_HOST_QUERY_RESET_NAME,
];

/// Extensions we'd like to have but won't refuse the layer over. Enabled
/// during `vkCreateDevice` only when the physical device advertises them.
/// Present-wait/present-id unlock the real display-side completion-time
/// path on Linux. `VK_GOOGLE_display_timing` is the fallback Mesa exposes
/// when present-wait isn't available — slightly coarser ordering-based
/// correlation, but enough to feed the Reflex overlay.
pub const OPTIONAL_EXTENSIONS: &[&CStr] = &[
    vk::KHR_PRESENT_ID_NAME,
    vk::KHR_PRESENT_WAIT_NAME,
    vk::GOOGLE_DISPLAY_TIMING_NAME,
];

pub struct PhysicalDeviceContext {
    pub instance: Arc<InstanceContext>,
    pub properties: vk::PhysicalDeviceProperties,
    pub queue_family_properties: Vec<vk::QueueFamilyProperties>,
    pub supports_required_extensions: bool,
    /// Subset of [`OPTIONAL_EXTENSIONS`] this device exposes. Indices align
    /// with the slice above. Caller may iterate to know what to enable +
    /// feature-patch at device creation time.
    pub supported_optionals: Vec<&'static CStr>,
}

impl PhysicalDeviceContext {
    /// # Safety
    /// Caller guarantees `handle` belongs to `instance`.
    pub unsafe fn new(instance: Arc<InstanceContext>, handle: vk::PhysicalDevice) -> Self {
        let fns = &instance.fns;
        let mut properties = vk::PhysicalDeviceProperties::default();
        unsafe { (fns.get_physical_device_properties)(handle, &mut properties) };

        let queue_family_properties = unsafe { query_queue_families(&instance, handle) };
        let available = unsafe { list_available_extensions(&instance, handle) };

        let supports_required_extensions = REQUIRED_EXTENSIONS
            .iter()
            .all(|w| available.iter().any(|h| h == w.to_bytes()));

        let supported_optionals: Vec<&'static CStr> = OPTIONAL_EXTENSIONS
            .iter()
            .copied()
            .filter(|w| available.iter().any(|h| h == w.to_bytes()))
            .collect();

        Self {
            instance,
            properties,
            queue_family_properties,
            supports_required_extensions,
            supported_optionals,
        }
    }
}

unsafe fn query_queue_families(
    instance: &InstanceContext,
    physical_device: vk::PhysicalDevice,
) -> Vec<vk::QueueFamilyProperties> {
    let fns = &instance.fns;
    let mut count = 0u32;
    unsafe {
        (fns.get_physical_device_queue_family_properties)(
            physical_device,
            &mut count,
            std::ptr::null_mut(),
        )
    };
    let mut props = vec![vk::QueueFamilyProperties::default(); count as usize];
    unsafe {
        (fns.get_physical_device_queue_family_properties)(
            physical_device,
            &mut count,
            props.as_mut_ptr(),
        )
    };
    props.truncate(count as usize);
    props
}

/// All device extensions the driver advertises for `physical_device`, each
/// returned as its raw byte-name (no NUL terminator).
unsafe fn list_available_extensions(
    instance: &InstanceContext,
    physical_device: vk::PhysicalDevice,
) -> Vec<Vec<u8>> {
    let fns = &instance.fns;
    let mut count = 0u32;
    let r = unsafe {
        (fns.enumerate_device_extension_properties)(
            physical_device,
            std::ptr::null(),
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if r != vk::Result::SUCCESS {
        return Vec::new();
    }
    let mut props = vec![vk::ExtensionProperties::default(); count as usize];
    let r = unsafe {
        (fns.enumerate_device_extension_properties)(
            physical_device,
            std::ptr::null(),
            &mut count,
            props.as_mut_ptr(),
        )
    };
    if r != vk::Result::SUCCESS {
        return Vec::new();
    }
    props.truncate(count as usize);
    props
        .iter()
        .filter_map(|p| ext_name(p).map(<[u8]>::to_vec))
        .collect()
}

fn ext_name(prop: &vk::ExtensionProperties) -> Option<&[u8]> {
    let end = prop
        .extension_name
        .iter()
        .position(|c| *c == 0)
        .unwrap_or(prop.extension_name.len());
    // SAFETY: c_char and u8 layout-compatible on all supported platforms.
    Some(unsafe { std::slice::from_raw_parts(prop.extension_name.as_ptr().cast::<u8>(), end) })
}
