//! Hand-rolled `VK_AMD_anti_lag` bindings.
//!
//! Added to Vulkan 1.3.296. The pinned `ash` (0.38.0+1.3.281) predates this,
//! so we declare just what the layer actually consumes.

use ash::vk;
use std::ffi::{CStr, c_void};

pub const AMD_ANTI_LAG_NAME: &CStr = c"VK_AMD_anti_lag";
pub const AMD_ANTI_LAG_SPEC_VERSION: u32 = 1;

pub const STRUCTURE_TYPE_PHYSICAL_DEVICE_ANTI_LAG_FEATURES_AMD: vk::StructureType =
    vk::StructureType::from_raw(1000476000);

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AntiLagModeAMD(pub i32);
impl AntiLagModeAMD {
    /// Mode "off" — the only variant the strategy reads.
    /// Spec also defines DRIVER_CONTROL=0 and ON=1; they're treated by the
    /// strategy as "enabled" generically.
    pub const OFF: Self = Self(2);
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AntiLagStageAMD(pub i32);
impl AntiLagStageAMD {
    pub const INPUT: Self = Self(0);
    pub const PRESENT: Self = Self(1);
}

#[repr(C)]
pub struct PhysicalDeviceAntiLagFeaturesAMD {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub anti_lag: vk::Bool32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AntiLagPresentationInfoAMD {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub stage: AntiLagStageAMD,
    pub frame_index: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AntiLagDataAMD {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub mode: AntiLagModeAMD,
    pub max_fps: u32,
    pub p_presentation_info: *const AntiLagPresentationInfoAMD,
}
