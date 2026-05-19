//! Vulkan entrypoints exposed via `VkPace_GetInstanceProcAddr` /
//! `VkPace_GetDeviceProcAddr`. Each submodule groups one ABI surface:
//!
//! - [`latency`]: `VK_NV_low_latency2` device entrypoints + AMD AntiLag.
//! - [`swapchain`]: `vkCreateSwapchainKHR` family + `vkAcquireNextImage*`.
//! - [`queue`]: `vkQueueSubmit[2[KHR]]`, `vkQueuePresentKHR`, submit-injection
//!   arena, and shared `should_inject_for_submit` predicate.
//!
//! Each entrypoint is `pub(crate)` so the function-table in `lib.rs` can
//! address it via path.

pub mod latency;
pub mod queue;
pub mod swapchain;
