//! Safe iteration over Vulkan `pNext` chains.
//!
//! A `pNext` chain is a singly-linked list of C structs whose first two
//! fields are always `(StructureType, *const c_void)`. We model that as a
//! `BaseHeader` and provide iterators that yield `NonNull<BaseHeader>` so
//! consumers don't deal with raw `*const c_void`.
//!
//! ## Safety invariant
//!
//! Once you construct a [`PNextIter`] or [`PNextIterMut`] from a head pointer,
//! that head must remain valid for the iterator's lifetime, and no other
//! mutator may touch the chain concurrently. Vulkan layer entrypoints satisfy
//! this trivially — the chain is built by the caller before the entrypoint
//! returns and not mutated afterwards.

use ash::vk;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BaseHeader {
    pub s_type: vk::StructureType,
    pub p_next: *const BaseHeader,
}

pub struct PNextIter<'a> {
    cur: *const BaseHeader,
    _marker: PhantomData<&'a BaseHeader>,
}

impl<'a> PNextIter<'a> {
    /// # Safety
    /// Caller guarantees the chain rooted at `head` is well-formed and
    /// stable for `'a`.
    pub unsafe fn new(head: *const c_void) -> Self {
        Self {
            cur: head.cast(),
            _marker: PhantomData,
        }
    }
}

impl<'a> Iterator for PNextIter<'a> {
    type Item = (vk::StructureType, NonNull<BaseHeader>);

    fn next(&mut self) -> Option<Self::Item> {
        let cur = NonNull::new(self.cur as *mut BaseHeader)?;
        // SAFETY: caller-upheld invariant from `new`.
        let header = unsafe { *cur.as_ptr() };
        self.cur = header.p_next;
        Some((header.s_type, cur))
    }
}

pub struct PNextIterMut<'a> {
    cur: *mut BaseHeader,
    _marker: PhantomData<&'a mut BaseHeader>,
}

impl<'a> PNextIterMut<'a> {
    /// # Safety
    /// See [`PNextIter::new`]; additionally the chain must be exclusively
    /// owned by the caller for `'a`.
    pub unsafe fn new(head: *mut c_void) -> Self {
        Self {
            cur: head.cast(),
            _marker: PhantomData,
        }
    }
}

impl<'a> Iterator for PNextIterMut<'a> {
    type Item = (vk::StructureType, NonNull<BaseHeader>);

    fn next(&mut self) -> Option<Self::Item> {
        let cur = NonNull::new(self.cur)?;
        let header = unsafe { *cur.as_ptr() };
        self.cur = header.p_next as *mut BaseHeader;
        Some((header.s_type, cur))
    }
}

/// Find first link with matching `s_type`, returning a typed const pointer.
///
/// # Safety
/// See [`PNextIter::new`].
pub unsafe fn find<T>(head: *const c_void, stype: vk::StructureType) -> Option<*const T> {
    unsafe { PNextIter::new(head) }
        .find(|(t, _)| *t == stype)
        .map(|(_, p)| p.as_ptr() as *const T)
}

/// Mutable variant. Returns a typed mutable pointer.
///
/// # Safety
/// See [`PNextIterMut::new`].
pub unsafe fn find_mut<T>(head: *mut c_void, stype: vk::StructureType) -> Option<*mut T> {
    unsafe { PNextIterMut::new(head) }
        .find(|(t, _)| *t == stype)
        .map(|(_, p)| p.as_ptr() as *mut T)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    struct Link {
        s_type: vk::StructureType,
        p_next: *const c_void,
        payload: u32,
    }

    #[test]
    fn finds_typed_link() {
        let tail = Link {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            payload: 42,
        };
        let head = Link {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: &tail as *const _ as *const c_void,
            payload: 7,
        };
        let p = unsafe {
            find::<Link>(
                &head as *const _ as *const c_void,
                vk::StructureType::APPLICATION_INFO,
            )
        };
        assert!(p.is_some());
        assert_eq!(unsafe { (*p.unwrap()).payload }, 42);
    }

    #[test]
    fn returns_none_when_missing() {
        let head = Link {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            payload: 0,
        };
        let p = unsafe {
            find::<Link>(
                &head as *const _ as *const c_void,
                vk::StructureType::APPLICATION_INFO,
            )
        };
        assert!(p.is_none());
    }

    #[test]
    fn null_head_is_safe() {
        let mut iter = unsafe { PNextIter::new(std::ptr::null()) };
        assert!(iter.next().is_none());
    }

    #[test]
    fn iterates_full_chain() {
        let third = Link {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: std::ptr::null(),
            payload: 3,
        };
        let second = Link {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: &third as *const _ as *const c_void,
            payload: 2,
        };
        let head = Link {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: &second as *const _ as *const c_void,
            payload: 1,
        };
        let stypes: Vec<i32> = unsafe {
            PNextIter::new(&head as *const _ as *const c_void)
                .map(|(t, _)| t.as_raw())
                .collect()
        };
        assert_eq!(
            stypes,
            vec![
                vk::StructureType::INSTANCE_CREATE_INFO.as_raw(),
                vk::StructureType::APPLICATION_INFO.as_raw(),
                vk::StructureType::DEVICE_CREATE_INFO.as_raw(),
            ]
        );
    }
}
