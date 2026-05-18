use ash::vk;
use std::sync::Arc;

use crate::device::DeviceContext;
use crate::dispatch::DeviceTable;
use crate::strategy::QueueStrategy;
use crate::timestamp_pool::TimestampPool;

/// RAII owner for a `VkCommandPool` — released on drop.
///
/// This is its own type so we can control drop order relative to the
/// `TimestampPool` (which frees command buffers allocated *from* this pool).
/// Fields drop in declaration order; placing `timestamp_pool` before
/// `command_pool` in `QueueContext` makes ordering correct.
pub struct CommandPoolOwner {
    pub handle: vk::CommandPool,
    device: vk::Device,
    fns: Arc<DeviceTable>,
}

impl Drop for CommandPoolOwner {
    fn drop(&mut self) {
        if self.handle != vk::CommandPool::null() {
            unsafe { (self.fns.destroy_command_pool)(self.device, self.handle, std::ptr::null()) };
        }
    }
}

/// Per-`VkQueue` state.
///
/// **Drop-order invariant.** Rust drops fields in declaration order. We
/// rely on:
///
/// 1. `strategy` first — releases pending `SubmissionSpan`s, which in turn
///    drop their `Handle`s, queuing them to the timestamp_pool reaper.
/// 2. `timestamp_pool` second — its `Drop` joins the reaper thread and
///    frees command buffers (via `vkFreeCommandBuffers`) and the query pool.
///    This *must* happen while `_command_pool` is still alive.
/// 3. `_command_pool` last — `Drop` destroys the underlying `VkCommandPool`.
///
/// **Do not reorder.** Rust has no stable lint for field declaration order,
/// so this comment is the guardrail.
pub struct QueueContext {
    pub device: Arc<DeviceContext>,
    pub family_properties: vk::QueueFamilyProperties,
    pub strategy: Box<dyn QueueStrategy>,
    pub timestamp_pool: Arc<TimestampPool>,
    /// Held purely for its `Drop` impl — destroys the underlying
    /// `VkCommandPool` after `timestamp_pool` has freed its CBs.
    _command_pool: CommandPoolOwner,
}

impl QueueContext {
    /// # Safety
    /// `family_index` must be a valid family of `device`.
    pub unsafe fn new(device: Arc<DeviceContext>, family_index: u32) -> Option<Arc<Self>> {
        let family_properties = device
            .physical_device
            .queue_family_properties
            .get(family_index as usize)
            .copied()?;

        let cpci = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(family_index);
        let mut cp_handle = vk::CommandPool::null();
        let r = unsafe {
            (device.fns.create_command_pool)(device.handle, &cpci, std::ptr::null(), &mut cp_handle)
        };
        if r != vk::Result::SUCCESS {
            return None;
        }
        let command_pool = CommandPoolOwner {
            handle: cp_handle,
            device: device.handle,
            fns: device.fns.clone(),
        };

        let timestamp_pool = Arc::new(TimestampPool::new(
            device.handle,
            command_pool.handle,
            device.fns.clone(),
        ));

        let strategy = crate::strategy::make_queue_strategy(&device);

        Some(Arc::new(Self {
            device,
            family_properties,
            strategy,
            timestamp_pool,
            _command_pool: command_pool,
        }))
    }

    pub fn should_inject_timestamps(&self) -> bool {
        if !self.device.layer_enabled || self.family_properties.timestamp_valid_bits == 0 {
            return false;
        }
        // Only inject on graphics queues. CS2 (and most modern engines) submit
        // heavily to async-compute / transfer queues — wrapping each with two
        // timestamp CBs adds ~10 driver calls per submit, which on a
        // 3000-submits/sec workload is enough to cause measurable frametime
        // variance. Reflex/AntiLag only cares about the graphics frame; the
        // C++ reference layer makes the same exclusion in
        // `AntiLagQueueStrategy::should_track_submissions`.
        self.family_properties
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS)
    }
}
