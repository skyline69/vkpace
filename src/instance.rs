use ash::vk;
use rustc_hash::FxBuildHasher;
use std::ffi::CStr;
use std::sync::Arc;

use crate::config::LayerConfig;
use crate::dispatch::InstanceTable;
use crate::physical_device::PhysicalDeviceContext;
use crate::registry::FxDashMap;

pub struct InstanceContext {
    pub handle: vk::Instance,
    pub fns: InstanceTable,
    pub config: LayerConfig,
    pub is_simulation_decoupled: bool,
    /// `apiVersion` the application requested at instance create (0 if
    /// unspecified — per spec that means 1.0). We honor this when deciding
    /// which feature structs are safe to splice in at `vkCreateDevice`.
    pub api_version: u32,
    pub physical_devices: FxDashMap<u64, Arc<PhysicalDeviceContext>>,
}

impl InstanceContext {
    /// # Safety
    /// `create_info.pApplicationInfo` must satisfy the Vulkan spec.
    pub unsafe fn new(
        handle: vk::Instance,
        fns: InstanceTable,
        create_info: &vk::InstanceCreateInfo<'_>,
        base_config: &LayerConfig,
    ) -> (Self, Option<String>) {
        let (app_name, api_version) = unsafe { read_app_info(create_info) };
        let config = base_config.finalize_for_app(app_name.as_deref());
        let is_simulation_decoupled = config.is_known_decoupled(app_name.as_deref());
        (
            Self {
                handle,
                fns,
                config,
                is_simulation_decoupled,
                api_version,
                physical_devices: FxDashMap::with_hasher(FxBuildHasher),
            },
            app_name,
        )
    }

    #[inline]
    pub fn supports_1_2(&self) -> bool {
        self.api_version >= vk::API_VERSION_1_2
    }

    #[inline]
    pub fn supports_1_3(&self) -> bool {
        self.api_version >= vk::API_VERSION_1_3
    }
}

unsafe fn read_app_info(create_info: &vk::InstanceCreateInfo<'_>) -> (Option<String>, u32) {
    let Some(app_info) = (unsafe { create_info.p_application_info.as_ref() }) else {
        return (None, 0);
    };
    let name = if app_info.p_application_name.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(app_info.p_application_name) }
            .to_str()
            .ok()
            .map(str::to_owned)
    };
    (name, app_info.api_version)
}
