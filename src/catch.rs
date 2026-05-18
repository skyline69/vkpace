//! Panic-safe FFI shim.
//!
//! Unwinding across the Vulkan loader boundary is UB. Even with `panic =
//! "abort"` in release, layer hosts may build with the default unwinding
//! profile; we use `catch_unwind` so a bug here aborts the call rather than
//! taking down the process.

use ash::vk;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[inline]
pub fn vk_result<F: FnOnce() -> vk::Result>(f: F) -> vk::Result {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => {
            tracing::error!("layer entrypoint panicked");
            vk::Result::ERROR_UNKNOWN
        }
    }
}

#[inline]
pub fn vk_void<F: FnOnce()>(f: F) {
    if catch_unwind(AssertUnwindSafe(f)).is_err() {
        tracing::error!("layer entrypoint panicked (void return)");
    }
}

#[inline]
pub fn vk_pfn<F: FnOnce() -> vk::PFN_vkVoidFunction>(f: F) -> vk::PFN_vkVoidFunction {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => {
            tracing::error!("ProcAddr entrypoint panicked");
            None
        }
    }
}
