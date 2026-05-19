//! `vkQueueSubmit` family + `vkQueuePresentKHR`, including the per-submit
//! timestamp-injection arena. Variant dispatch (`Submit2 / Submit2KHR`) is
//! handled here so the loader-side function table only stores trampolines.

use ash::vk;
use smallvec::SmallVec;
use std::ffi::c_void;
use std::sync::Arc;

use crate::clock;
use crate::pnext;
use crate::queue::QueueContext;
use crate::registry;
use crate::strategy;
use crate::telemetry;
use crate::timestamp_pool;

/// Whether the given submit should be wrapped with timestamp CBs.
///
/// - LL2 (`expose_reflex`): inject every graphics submit. The strategy
///   keys submission spans by the optional `VkLatencySubmissionPresentIdNV`
///   (defaulting to 0 when absent); present-side matches by
///   `VkPresentIdKHR`. We can't know which submit "matters" until present,
///   so we keep them all. Matches the C++ reference.
/// - AntiLag: only while AntiLag tracking is active (input→present window).
///   In the dormant window we'd just do work the strategy throws away.
pub(crate) unsafe fn should_inject_for_submit(qctx: &QueueContext, _p_next: *const c_void) -> bool {
    if !qctx.should_inject_timestamps() {
        return false;
    }
    if qctx.device.instance.config.expose_reflex {
        true
    } else {
        qctx.device
            .strategy
            .get()
            .and_then(|s| s.as_anti_lag())
            .is_some_and(|s| s.should_track_submissions())
    }
}

// Thread-local fallback storage for the QueueSubmit injection path. The
// SmallVec inline capacities cover the typical 1-4-submit / ≤8-CB hot path
// without heap allocation; for the (rare) spill case we hand the spilled
// `Vec` back to the thread-local arena instead of dropping it, so the next
// call on the same thread re-uses the buffer.
mod submit_arena {
    use ash::vk;
    use std::cell::RefCell;

    /// Discard returned buffers whose capacity exceeds this. A pathological
    /// submit (hundreds of command buffers) shouldn't permanently inflate
    /// the per-thread arena. Same value for both buffer kinds — typical
    /// hot path stays at ≤8.
    const MAX_CACHED_CAP: usize = 64;
    /// Cap how many buffers each thread holds in reserve. Re-use is most
    /// valuable at depth 1-2; deeper never amortizes.
    const MAX_CACHED_BUFFERS: usize = 4;

    thread_local! {
        static SUBMIT1_CBS: RefCell<Vec<Vec<vk::CommandBuffer>>> = const { RefCell::new(Vec::new()) };
        static SUBMIT2_CBS: RefCell<Vec<Vec<vk::CommandBufferSubmitInfo<'static>>>> =
            const { RefCell::new(Vec::new()) };
    }

    pub fn take_submit1() -> Vec<vk::CommandBuffer> {
        SUBMIT1_CBS.with(|c| c.borrow_mut().pop().unwrap_or_default())
    }

    pub fn give_submit1(mut v: Vec<vk::CommandBuffer>) {
        if v.capacity() > MAX_CACHED_CAP {
            return; // drop oversized — keeps arena bounded.
        }
        v.clear();
        SUBMIT1_CBS.with(|c| {
            let mut b = c.borrow_mut();
            if b.len() < MAX_CACHED_BUFFERS {
                b.push(v);
            }
        });
    }

    pub fn take_submit2() -> Vec<vk::CommandBufferSubmitInfo<'static>> {
        SUBMIT2_CBS.with(|c| c.borrow_mut().pop().unwrap_or_default())
    }

    pub fn give_submit2(mut v: Vec<vk::CommandBufferSubmitInfo<'static>>) {
        if v.capacity() > MAX_CACHED_CAP {
            return;
        }
        v.clear();
        SUBMIT2_CBS.with(|c| {
            let mut b = c.borrow_mut();
            if b.len() < MAX_CACHED_BUFFERS {
                b.push(v);
            }
        });
    }
}

pub(crate) unsafe extern "system" fn queue_submit(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo<'_>,
    fence: vk::Fence,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(qctx) = registry::QUEUES
            .get(&registry::key(queue))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let fns = &qctx.device.fns;
        if submit_count == 0 {
            return unsafe { (fns.queue_submit)(queue, submit_count, p_submits, fence) };
        }
        let submits = unsafe { std::slice::from_raw_parts(p_submits, submit_count as usize) };

        // Per-submit decision. Skip injection on submits the strategy
        // wouldn't track anyway. Fast-path: if NO submit needs injection,
        // forward the caller's array verbatim — zero allocations.
        let inject_mask: SmallVec<[bool; 4]> = submits
            .iter()
            .map(|s| unsafe { should_inject_for_submit(&qctx, s.p_next) })
            .collect();
        if inject_mask.iter().all(|b| !b) {
            let r = unsafe { (fns.queue_submit)(queue, submit_count, p_submits, fence) };
            crate::TELEMETRY
                .counters
                .record_submit_call(submit_count, 0);
            return r;
        }

        let mut handles: SmallVec<[Option<Arc<timestamp_pool::Handle>>; 4]> =
            SmallVec::with_capacity(submits.len());
        let mut all_cbs: SmallVec<[Vec<vk::CommandBuffer>; 4]> =
            SmallVec::with_capacity(submits.len());
        let mut next_submits: SmallVec<[vk::SubmitInfo<'_>; 4]> =
            SmallVec::with_capacity(submits.len());

        for (submit, &inject) in submits.iter().zip(inject_mask.iter()) {
            if !inject {
                next_submits.push(*submit);
                handles.push(None);
                continue;
            }
            let Some(handle) = qctx.timestamp_pool.acquire() else {
                return unsafe { (fns.queue_submit)(queue, submit_count, p_submits, fence) };
            };
            let mut cbs = submit_arena::take_submit1();
            cbs.reserve(submit.command_buffer_count as usize + 2);
            cbs.push(handle.start_buffer());
            cbs.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    submit.p_command_buffers,
                    submit.command_buffer_count as usize,
                )
            });
            cbs.push(handle.end_buffer());
            let mut next = *submit;
            next.command_buffer_count = cbs.len() as u32;
            next.p_command_buffers = cbs.as_ptr();
            all_cbs.push(cbs);
            next_submits.push(next);
            handles.push(Some(handle));
        }

        let r = unsafe {
            (fns.queue_submit)(
                queue,
                next_submits.len() as u32,
                next_submits.as_ptr(),
                fence,
            )
        };
        for v in all_cbs.drain(..) {
            submit_arena::give_submit1(v);
        }
        if r != vk::Result::SUCCESS {
            return r;
        }
        let mut injected_count = 0u32;
        for (submit, handle) in submits.iter().zip(handles) {
            let Some(handle) = handle else { continue };
            injected_count += 1;
            handle
                .was_submitted
                .store(true, std::sync::atomic::Ordering::Relaxed);
            unsafe { qctx.strategy.notify_submit(submit.p_next, handle) };
        }
        crate::TELEMETRY
            .counters
            .record_submit_call(submit_count, injected_count);
        vk::Result::SUCCESS
    })
}

pub(crate) unsafe extern "system" fn queue_submit2(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo2<'_>,
    fence: vk::Fence,
) -> vk::Result {
    unsafe { queue_submit2_dispatch(queue, submit_count, p_submits, fence, Submit2Variant::Core) }
}

pub(crate) unsafe extern "system" fn queue_submit2_khr(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo2<'_>,
    fence: vk::Fence,
) -> vk::Result {
    unsafe { queue_submit2_dispatch(queue, submit_count, p_submits, fence, Submit2Variant::Khr) }
}

#[derive(Clone, Copy)]
enum Submit2Variant {
    Core,
    Khr,
}

unsafe fn queue_submit2_dispatch(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo2<'_>,
    fence: vk::Fence,
    variant: Submit2Variant,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(qctx) = registry::QUEUES
            .get(&registry::key(queue))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let fns = &qctx.device.fns;
        let submit2 = match variant {
            Submit2Variant::Core => fns.queue_submit2,
            Submit2Variant::Khr => fns.queue_submit2_khr,
        };
        let Some(submit2) = submit2 else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        if submit_count == 0 {
            return unsafe { submit2(queue, submit_count, p_submits, fence) };
        }
        let submits = unsafe { std::slice::from_raw_parts(p_submits, submit_count as usize) };

        let inject_mask: SmallVec<[bool; 4]> = submits
            .iter()
            .map(|s| unsafe { should_inject_for_submit(&qctx, s.p_next) })
            .collect();
        if inject_mask.iter().all(|b| !b) {
            let r = unsafe { submit2(queue, submit_count, p_submits, fence) };
            crate::TELEMETRY
                .counters
                .record_submit_call(submit_count, 0);
            return r;
        }

        let mut handles: SmallVec<[Option<Arc<timestamp_pool::Handle>>; 4]> =
            SmallVec::with_capacity(submits.len());
        let mut all_cbs: SmallVec<[Vec<vk::CommandBufferSubmitInfo<'static>>; 4]> =
            SmallVec::with_capacity(submits.len());
        let mut next_submits: SmallVec<[vk::SubmitInfo2<'_>; 4]> =
            SmallVec::with_capacity(submits.len());

        for (submit, &inject) in submits.iter().zip(inject_mask.iter()) {
            if !inject {
                next_submits.push(*submit);
                handles.push(None);
                continue;
            }
            let Some(handle) = qctx.timestamp_pool.acquire() else {
                return unsafe { submit2(queue, submit_count, p_submits, fence) };
            };
            let start_cb: vk::CommandBufferSubmitInfo<'static> =
                vk::CommandBufferSubmitInfo::default().command_buffer(handle.start_buffer());
            let end_cb: vk::CommandBufferSubmitInfo<'static> =
                vk::CommandBufferSubmitInfo::default().command_buffer(handle.end_buffer());

            let mut cbs = submit_arena::take_submit2();
            cbs.reserve(submit.command_buffer_info_count as usize + 2);
            cbs.push(start_cb);
            let user_slice = unsafe {
                std::slice::from_raw_parts(
                    submit.p_command_buffer_infos,
                    submit.command_buffer_info_count as usize,
                )
            };
            // ash's `CommandBufferSubmitInfo` lifetime parameter is phantom-only
            // (pNext is a raw pointer with no Rust-side borrow). Copy the POD
            // through `read_unaligned` so we don't carry the caller's lifetime
            // into the arena, then re-tag with `'static`. Safe: identical
            // layout, no borrowed contents.
            cbs.extend(user_slice.iter().map(|s| {
                let copy: vk::CommandBufferSubmitInfo<'_> = unsafe { std::ptr::read(s) };
                unsafe {
                    std::mem::transmute::<
                        vk::CommandBufferSubmitInfo<'_>,
                        vk::CommandBufferSubmitInfo<'static>,
                    >(copy)
                }
            }));
            cbs.push(end_cb);
            let mut next = *submit;
            next.command_buffer_info_count = cbs.len() as u32;
            next.p_command_buffer_infos = cbs.as_ptr().cast::<vk::CommandBufferSubmitInfo<'_>>();
            all_cbs.push(cbs);
            next_submits.push(next);
            handles.push(Some(handle));
        }

        let r = unsafe {
            submit2(
                queue,
                next_submits.len() as u32,
                next_submits.as_ptr(),
                fence,
            )
        };
        for v in all_cbs.drain(..) {
            submit_arena::give_submit2(v);
        }
        if r != vk::Result::SUCCESS {
            return r;
        }
        let mut injected_count = 0u32;
        for (submit, handle) in submits.iter().zip(handles) {
            let Some(handle) = handle else { continue };
            injected_count += 1;
            handle
                .was_submitted
                .store(true, std::sync::atomic::Ordering::Relaxed);
            unsafe { qctx.strategy.notify_submit(submit.p_next, handle) };
        }
        crate::TELEMETRY
            .counters
            .record_submit_call(submit_count, injected_count);
        vk::Result::SUCCESS
    })
}

/// Extract the first present_id in a `VkPresentInfoKHR` chain, or 0 if
/// the app didn't attach `VkPresentIdKHR`.
unsafe fn first_present_id(info: &vk::PresentInfoKHR<'_>) -> u64 {
    let Some(p) = (unsafe {
        pnext::find::<vk::PresentIdKHR>(info.p_next, vk::StructureType::PRESENT_ID_KHR)
    }) else {
        return 0;
    };
    let p = unsafe { &*p };
    if p.p_present_ids.is_null() || p.swapchain_count == 0 {
        return 0;
    }
    unsafe { *p.p_present_ids }
}

pub(crate) unsafe extern "system" fn queue_present_khr(
    queue: vk::Queue,
    p_present_info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    crate::catch::vk_result(|| {
        let Some(qctx) = registry::QUEUES
            .get(&registry::key(queue))
            .map(|r| r.clone())
        else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let fns = &qctx.device.fns;
        let Some(present_khr) = fns.queue_present_khr else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        let r = unsafe { present_khr(queue, p_present_info) };
        crate::TELEMETRY.counters.inc_present();
        if crate::TELEMETRY.socket_enabled() {
            let pid = if let Some(info) = unsafe { p_present_info.as_ref() } {
                unsafe { first_present_id(info) }
            } else {
                0
            };
            let latency_us = qctx
                .device
                .strategy
                .get()
                .and_then(|s| s.as_low_latency2())
                .map(|ll2| ll2.markers().latest_latency_us())
                .unwrap_or(0);
            crate::TELEMETRY.push_record(telemetry::FrameRecord {
                host_ns: clock::DeviceClock::now(),
                frame_index: crate::TELEMETRY
                    .counters
                    .frames
                    .load(std::sync::atomic::Ordering::Relaxed),
                queue_id: registry::key(queue),
                present_id: pid,
                latency_us,
            });
        }

        if let Some(info) = unsafe { p_present_info.as_ref() } {
            qctx.strategy.notify_present(info);
            if let Some(s) = qctx.device.strategy.get().and_then(|s| s.as_low_latency2()) {
                strategy::low_latency2::forward_present(s, info);
            }
            if info.swapchain_count > 0 {
                let sw = unsafe { *info.p_swapchains };
                qctx.device.ensure_vrr_target(sw);
            }
            let khr = qctx.device.present_wait.get();
            let google = qctx.device.google_timing.get();
            if khr.is_some() || google.is_some() {
                let present_ids = unsafe {
                    pnext::find::<vk::PresentIdKHR>(info.p_next, vk::StructureType::PRESENT_ID_KHR)
                        .and_then(|p| {
                            let p = &*p;
                            (!p.p_present_ids.is_null()).then(|| {
                                std::slice::from_raw_parts(
                                    p.p_present_ids,
                                    p.swapchain_count as usize,
                                )
                            })
                        })
                };
                for i in 0..info.swapchain_count as usize {
                    let swapchain = unsafe { *info.p_swapchains.add(i) };
                    let pid = present_ids.map(|s| s[i]).unwrap_or(0);
                    if pid == 0 {
                        continue;
                    }
                    if let Some(w) = khr {
                        w.enqueue(swapchain, pid);
                    } else if let Some(w) = google {
                        w.enqueue(swapchain, pid);
                    }
                }
            }
        }
        r
    })
}
