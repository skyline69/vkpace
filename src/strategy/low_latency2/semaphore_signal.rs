//! Timeline semaphore signaller with monotonic-but-not-strictly-increasing
//! tolerance.
//!
//! `VK_NV_low_latency2` permits repeated values (0, 1, 1, 1, 2, ...) but
//! `vkSignalSemaphore` rejects `value <= current`. We read the current value
//! and skip the call if we'd otherwise hit that error.

use ash::vk;

use crate::device::DeviceContext;

#[derive(Clone, Copy)]
pub struct SemaphoreSignal {
    semaphore: vk::Semaphore,
    value: u64,
}

impl SemaphoreSignal {
    pub fn new(semaphore: vk::Semaphore, value: u64) -> Self {
        Self { semaphore, value }
    }

    pub fn signal(&self, device: &DeviceContext) {
        let (Some(get_v), Some(sig)) = (
            device.fns.get_semaphore_counter_value,
            device.fns.signal_semaphore,
        ) else {
            tracing::warn!(
                "SemaphoreSignal: timeline semaphore entrypoints missing — dropping signal"
            );
            return;
        };
        let mut current = 0u64;
        let r = unsafe { get_v(device.handle, self.semaphore, &mut current) };
        if r != vk::Result::SUCCESS || current >= self.value {
            return;
        }
        let ssi = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(self.value);
        let _ = unsafe { sig(device.handle, &ssi) };
    }
}
