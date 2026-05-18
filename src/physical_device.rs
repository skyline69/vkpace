use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;

use crate::instance::InstanceContext;

pub const REQUIRED_EXTENSIONS: &[&CStr] = &[
    vk::KHR_SYNCHRONIZATION2_NAME,
    vk::KHR_CALIBRATED_TIMESTAMPS_NAME,
    vk::EXT_HOST_QUERY_RESET_NAME,
];

pub struct PhysicalDeviceContext {
    pub instance: Arc<InstanceContext>,
    pub properties: vk::PhysicalDeviceProperties,
    pub queue_family_properties: Vec<vk::QueueFamilyProperties>,
    pub supports_required_extensions: bool,
}

impl PhysicalDeviceContext {
    /// # Safety
    /// Caller guarantees `handle` belongs to `instance`.
    pub unsafe fn new(instance: Arc<InstanceContext>, handle: vk::PhysicalDevice) -> Self {
        let fns = &instance.fns;
        let mut properties = vk::PhysicalDeviceProperties::default();
        unsafe { (fns.get_physical_device_properties)(handle, &mut properties) };

        let queue_family_properties = unsafe { query_queue_families(&instance, handle) };
        let supports_required_extensions = unsafe { check_required_extensions(&instance, handle) };

        Self {
            instance,
            properties,
            queue_family_properties,
            supports_required_extensions,
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

unsafe fn check_required_extensions(
    instance: &InstanceContext,
    physical_device: vk::PhysicalDevice,
) -> bool {
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
        return false;
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
        return false;
    }
    props.truncate(count as usize);

    REQUIRED_EXTENSIONS.iter().all(|wanted| {
        props
            .iter()
            .any(|have| ext_name(have) == Some(wanted.to_bytes()))
    })
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
