//! Loader-private layer link structs. These live in `vulkan/vk_layer.h` and
//! aren't part of the public ash bindings, so we declare them by hand.

use ash::vk;
use std::ffi::c_void;

use crate::dispatch::{PfnGetDeviceProcAddr, PfnGetInstanceProcAddr};

/// Loader/layer interface negotiated via `vkNegotiateLoaderLayerInterfaceVersion`.
///
/// Layers and the loader announce the maximum interface version they
/// understand and meet at the lower of the two. We advertise version 2,
/// which adds `pfnNextGetPhysicalDeviceProcAddr` (not currently used) and
/// avoids the loader falling back to the legacy `vk_layerGetPhysicalDevice…`
/// codepath.
pub const CURRENT_LOADER_LAYER_INTERFACE_VERSION: u32 = 2;

#[repr(C)]
pub struct VkNegotiateLayerInterface {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub loader_layer_interface_version: u32,
    pub pfn_get_instance_proc_addr: Option<PfnGetInstanceProcAddr>,
    pub pfn_get_device_proc_addr: Option<PfnGetDeviceProcAddr>,
    pub pfn_get_physical_device_proc_addr: Option<
        unsafe extern "system" fn(vk::Instance, *const std::ffi::c_char) -> vk::PFN_vkVoidFunction,
    >,
}

#[repr(C)]
pub struct Header {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VkLayerFunction(u32);

impl VkLayerFunction {
    pub const LINK_INFO: Self = Self(0);
}

#[repr(C)]
pub struct VkLayerInstanceLink {
    pub p_next: *mut VkLayerInstanceLink,
    pub pfn_next_get_instance_proc_addr: Option<PfnGetInstanceProcAddr>,
}

#[repr(C)]
pub union VkLayerInstanceCreateInfoUnion {
    pub p_layer_info: *mut VkLayerInstanceLink,
    pub pfn_set_instance_loader_data: *mut c_void,
}

#[repr(C)]
pub struct VkLayerInstanceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: VkLayerFunction,
    pub u: VkLayerInstanceCreateInfoUnion,
}

#[repr(C)]
pub struct VkLayerDeviceLink {
    pub p_next: *mut VkLayerDeviceLink,
    pub pfn_next_get_instance_proc_addr: Option<PfnGetInstanceProcAddr>,
    pub pfn_next_get_device_proc_addr: Option<PfnGetDeviceProcAddr>,
}

#[repr(C)]
pub union VkLayerDeviceCreateInfoUnion {
    pub p_layer_info: *mut VkLayerDeviceLink,
    pub pfn_set_device_loader_data: *mut c_void,
}

#[repr(C)]
pub struct VkLayerDeviceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: VkLayerFunction,
    pub u: VkLayerDeviceCreateInfoUnion,
}
