//! Slim, hand-rolled dispatch tables.
//!
//! We resolve only the entrypoints the layer actually calls. Each table is
//! immutable after construction and stored behind an `Arc`. Core 1.0/1.1
//! functions are **required** (we panic if any is missing — that signals a
//! broken loader/driver). 1.2/1.3 + extension functions are **optional**
//! (`Option<PFN>`); callers check before using and the layer disables
//! features that depend on missing functions rather than aborting.

use ash::vk;
use std::ffi::{CStr, c_char};
use std::mem;

pub type PfnGetInstanceProcAddr =
    unsafe extern "system" fn(vk::Instance, *const c_char) -> vk::PFN_vkVoidFunction;
pub type PfnGetDeviceProcAddr =
    unsafe extern "system" fn(vk::Device, *const c_char) -> vk::PFN_vkVoidFunction;

/// Resolve `vkFoo` from a `GetInstanceProcAddr`. Returns `None` if the
/// loader doesn't know the function.
///
/// # Safety
/// `gipa` must be a valid PFN_vkGetInstanceProcAddr from the next layer.
unsafe fn load_inst<F: Copy>(
    gipa: PfnGetInstanceProcAddr,
    instance: vk::Instance,
    name: &CStr,
) -> Option<F> {
    // `mem::transmute_copy` only checks min(size_of(F), size_of(PFN)). For
    // function-pointer types both are pointer-sized, so this is correct, but
    // the explicit assert documents the assumption.
    const {
        assert!(mem::size_of::<F>() == mem::size_of::<vk::PFN_vkVoidFunction>());
    }
    unsafe {
        let raw = gipa(instance, name.as_ptr())?;
        Some(mem::transmute_copy(&raw))
    }
}

unsafe fn load_dev<F: Copy>(
    gdpa: PfnGetDeviceProcAddr,
    device: vk::Device,
    name: &CStr,
) -> Option<F> {
    const {
        assert!(mem::size_of::<F>() == mem::size_of::<vk::PFN_vkVoidFunction>());
    }
    unsafe {
        let raw = gdpa(device, name.as_ptr())?;
        Some(mem::transmute_copy(&raw))
    }
}

pub struct InstanceTable {
    pub get_instance_proc_addr: PfnGetInstanceProcAddr,
    pub destroy_instance: vk::PFN_vkDestroyInstance,
    pub enumerate_physical_devices: vk::PFN_vkEnumeratePhysicalDevices,
    pub get_physical_device_properties: vk::PFN_vkGetPhysicalDeviceProperties,
    pub enumerate_device_extension_properties: vk::PFN_vkEnumerateDeviceExtensionProperties,
    pub get_physical_device_queue_family_properties:
        vk::PFN_vkGetPhysicalDeviceQueueFamilyProperties,
    // ── Optional (Vulkan 1.1+ / KHR extensions) ────────────────────────────
    pub get_physical_device_properties2: Option<vk::PFN_vkGetPhysicalDeviceProperties2>,
    pub get_physical_device_features2: Option<vk::PFN_vkGetPhysicalDeviceFeatures2>,
    pub get_physical_device_surface_capabilities2_khr:
        Option<vk::PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>,
}

impl InstanceTable {
    /// # Safety
    /// `gipa` and `instance` must be valid.
    pub unsafe fn load(gipa: PfnGetInstanceProcAddr, instance: vk::Instance) -> Self {
        unsafe {
            Self {
                get_instance_proc_addr: gipa,
                destroy_instance: load_inst(gipa, instance, c"vkDestroyInstance")
                    .expect("vkDestroyInstance must exist"),
                enumerate_physical_devices: load_inst(
                    gipa,
                    instance,
                    c"vkEnumeratePhysicalDevices",
                )
                .expect("vkEnumeratePhysicalDevices must exist"),
                get_physical_device_properties: load_inst(
                    gipa,
                    instance,
                    c"vkGetPhysicalDeviceProperties",
                )
                .expect("vkGetPhysicalDeviceProperties must exist"),
                enumerate_device_extension_properties: load_inst(
                    gipa,
                    instance,
                    c"vkEnumerateDeviceExtensionProperties",
                )
                .expect("vkEnumerateDeviceExtensionProperties must exist"),
                get_physical_device_queue_family_properties: load_inst(
                    gipa,
                    instance,
                    c"vkGetPhysicalDeviceQueueFamilyProperties",
                )
                .expect("vkGetPhysicalDeviceQueueFamilyProperties must exist"),

                get_physical_device_properties2: load_inst(
                    gipa,
                    instance,
                    c"vkGetPhysicalDeviceProperties2",
                )
                .or_else(|| load_inst(gipa, instance, c"vkGetPhysicalDeviceProperties2KHR")),
                get_physical_device_features2: load_inst(
                    gipa,
                    instance,
                    c"vkGetPhysicalDeviceFeatures2",
                )
                .or_else(|| load_inst(gipa, instance, c"vkGetPhysicalDeviceFeatures2KHR")),
                get_physical_device_surface_capabilities2_khr: load_inst(
                    gipa,
                    instance,
                    c"vkGetPhysicalDeviceSurfaceCapabilities2KHR",
                ),
            }
        }
    }
}

pub struct DeviceTable {
    pub get_device_proc_addr: PfnGetDeviceProcAddr,
    pub destroy_device: vk::PFN_vkDestroyDevice,
    pub get_device_queue: vk::PFN_vkGetDeviceQueue,
    pub queue_submit: vk::PFN_vkQueueSubmit,
    pub create_command_pool: vk::PFN_vkCreateCommandPool,
    pub destroy_command_pool: vk::PFN_vkDestroyCommandPool,
    pub allocate_command_buffers: vk::PFN_vkAllocateCommandBuffers,
    pub free_command_buffers: vk::PFN_vkFreeCommandBuffers,
    pub begin_command_buffer: vk::PFN_vkBeginCommandBuffer,
    pub end_command_buffer: vk::PFN_vkEndCommandBuffer,
    pub reset_command_buffer: vk::PFN_vkResetCommandBuffer,
    pub create_query_pool: vk::PFN_vkCreateQueryPool,
    pub destroy_query_pool: vk::PFN_vkDestroyQueryPool,
    pub get_query_pool_results: vk::PFN_vkGetQueryPoolResults,
    pub device_wait_idle: vk::PFN_vkDeviceWaitIdle,
    // ── Optional ───────────────────────────────────────────────────────────
    pub get_device_queue2: Option<vk::PFN_vkGetDeviceQueue2>,
    pub queue_submit2: Option<vk::PFN_vkQueueSubmit2>,
    pub queue_present_khr: Option<vk::PFN_vkQueuePresentKHR>,
    pub cmd_write_timestamp2: Option<vk::PFN_vkCmdWriteTimestamp2>,
    pub reset_query_pool: Option<vk::PFN_vkResetQueryPool>,
    pub get_calibrated_timestamps_khr: Option<vk::PFN_vkGetCalibratedTimestampsKHR>,
    pub create_swapchain_khr: Option<vk::PFN_vkCreateSwapchainKHR>,
    pub destroy_swapchain_khr: Option<vk::PFN_vkDestroySwapchainKHR>,
    pub acquire_next_image_khr: Option<vk::PFN_vkAcquireNextImageKHR>,
    pub acquire_next_image2_khr: Option<vk::PFN_vkAcquireNextImage2KHR>,
    pub signal_semaphore: Option<vk::PFN_vkSignalSemaphore>,
    pub get_semaphore_counter_value: Option<vk::PFN_vkGetSemaphoreCounterValue>,
    pub wait_for_present_khr: Option<vk::PFN_vkWaitForPresentKHR>,
}

impl DeviceTable {
    /// # Safety
    /// `gdpa` and `device` must be valid.
    pub unsafe fn load(gdpa: PfnGetDeviceProcAddr, device: vk::Device) -> Self {
        unsafe {
            Self {
                get_device_proc_addr: gdpa,
                destroy_device: load_dev(gdpa, device, c"vkDestroyDevice")
                    .expect("vkDestroyDevice must exist"),
                get_device_queue: load_dev(gdpa, device, c"vkGetDeviceQueue")
                    .expect("vkGetDeviceQueue must exist"),
                queue_submit: load_dev(gdpa, device, c"vkQueueSubmit")
                    .expect("vkQueueSubmit must exist"),
                create_command_pool: load_dev(gdpa, device, c"vkCreateCommandPool")
                    .expect("vkCreateCommandPool must exist"),
                destroy_command_pool: load_dev(gdpa, device, c"vkDestroyCommandPool")
                    .expect("vkDestroyCommandPool must exist"),
                allocate_command_buffers: load_dev(gdpa, device, c"vkAllocateCommandBuffers")
                    .expect("vkAllocateCommandBuffers must exist"),
                free_command_buffers: load_dev(gdpa, device, c"vkFreeCommandBuffers")
                    .expect("vkFreeCommandBuffers must exist"),
                begin_command_buffer: load_dev(gdpa, device, c"vkBeginCommandBuffer")
                    .expect("vkBeginCommandBuffer must exist"),
                end_command_buffer: load_dev(gdpa, device, c"vkEndCommandBuffer")
                    .expect("vkEndCommandBuffer must exist"),
                reset_command_buffer: load_dev(gdpa, device, c"vkResetCommandBuffer")
                    .expect("vkResetCommandBuffer must exist"),
                create_query_pool: load_dev(gdpa, device, c"vkCreateQueryPool")
                    .expect("vkCreateQueryPool must exist"),
                destroy_query_pool: load_dev(gdpa, device, c"vkDestroyQueryPool")
                    .expect("vkDestroyQueryPool must exist"),
                get_query_pool_results: load_dev(gdpa, device, c"vkGetQueryPoolResults")
                    .expect("vkGetQueryPoolResults must exist"),
                device_wait_idle: load_dev(gdpa, device, c"vkDeviceWaitIdle")
                    .expect("vkDeviceWaitIdle must exist"),

                get_device_queue2: load_dev(gdpa, device, c"vkGetDeviceQueue2"),
                queue_submit2: load_dev(gdpa, device, c"vkQueueSubmit2")
                    .or_else(|| load_dev(gdpa, device, c"vkQueueSubmit2KHR")),
                queue_present_khr: load_dev(gdpa, device, c"vkQueuePresentKHR"),
                cmd_write_timestamp2: load_dev(gdpa, device, c"vkCmdWriteTimestamp2")
                    .or_else(|| load_dev(gdpa, device, c"vkCmdWriteTimestamp2KHR")),
                reset_query_pool: load_dev(gdpa, device, c"vkResetQueryPool")
                    .or_else(|| load_dev(gdpa, device, c"vkResetQueryPoolEXT")),
                get_calibrated_timestamps_khr: load_dev(
                    gdpa,
                    device,
                    c"vkGetCalibratedTimestampsKHR",
                )
                .or_else(|| load_dev(gdpa, device, c"vkGetCalibratedTimestampsEXT")),
                create_swapchain_khr: load_dev(gdpa, device, c"vkCreateSwapchainKHR"),
                destroy_swapchain_khr: load_dev(gdpa, device, c"vkDestroySwapchainKHR"),
                acquire_next_image_khr: load_dev(gdpa, device, c"vkAcquireNextImageKHR"),
                acquire_next_image2_khr: load_dev(gdpa, device, c"vkAcquireNextImage2KHR"),
                signal_semaphore: load_dev(gdpa, device, c"vkSignalSemaphore")
                    .or_else(|| load_dev(gdpa, device, c"vkSignalSemaphoreKHR")),
                get_semaphore_counter_value: load_dev(gdpa, device, c"vkGetSemaphoreCounterValue")
                    .or_else(|| load_dev(gdpa, device, c"vkGetSemaphoreCounterValueKHR")),
                wait_for_present_khr: load_dev(gdpa, device, c"vkWaitForPresentKHR"),
            }
        }
    }
}
