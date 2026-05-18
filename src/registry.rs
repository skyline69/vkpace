//! Global registry of dispatchable Vulkan handles → layer state.
//!
//! We key on the handle's raw pointer value. Vulkan dispatchable handles are
//! stable pointers — the same physical device returned twice yields the same
//! pointer value — so identity-by-handle is sufficient and matches what the
//! C++ implementation does.

use ash::vk::Handle;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use rustc_hash::FxBuildHasher;
use std::sync::Arc;

use crate::device::DeviceContext;
use crate::instance::InstanceContext;
use crate::physical_device::PhysicalDeviceContext;
use crate::queue::QueueContext;

pub type FxDashMap<K, V> = DashMap<K, V, FxBuildHasher>;

pub static INSTANCES: Lazy<FxDashMap<u64, Arc<InstanceContext>>> =
    Lazy::new(|| DashMap::with_hasher(FxBuildHasher));
pub static PHYSICAL_DEVICES: Lazy<FxDashMap<u64, Arc<PhysicalDeviceContext>>> =
    Lazy::new(|| DashMap::with_hasher(FxBuildHasher));
pub static DEVICES: Lazy<FxDashMap<u64, Arc<DeviceContext>>> =
    Lazy::new(|| DashMap::with_hasher(FxBuildHasher));
pub static QUEUES: Lazy<FxDashMap<u64, Arc<QueueContext>>> =
    Lazy::new(|| DashMap::with_hasher(FxBuildHasher));

#[inline]
pub fn key<H: Handle>(handle: H) -> u64 {
    handle.as_raw()
}
