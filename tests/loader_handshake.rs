//! Integration tests for the loader handshake path.
//!
//! Exercises `vkNegotiateLoaderLayerInterfaceVersion`, then calls
//! `VkPace_GetInstanceProcAddr` / `VkPace_GetDeviceProcAddr` for the
//! exact strings the loader queries and asserts non-null results.
//!
//! We need to call into the `cdylib`'s exported symbols. Rust integration
//! tests link the crate as a normal lib (which it isn't — it's `cdylib`), so
//! we re-declare the entry points here via `extern "C"` and the test binary
//! resolves them at runtime through `dlopen` on the built `.so`.

use std::ffi::{CStr, CString, c_char, c_void};

#[repr(C)]
struct NegotiateLayerInterface {
    s_type: i32,
    p_next: *mut c_void,
    loader_layer_interface_version: u32,
    pfn_get_instance_proc_addr: Option<
        unsafe extern "system" fn(
            *mut c_void,
            *const c_char,
        ) -> Option<unsafe extern "system" fn()>,
    >,
    pfn_get_device_proc_addr: Option<
        unsafe extern "system" fn(
            *mut c_void,
            *const c_char,
        ) -> Option<unsafe extern "system" fn()>,
    >,
    pfn_get_physical_device_proc_addr: Option<
        unsafe extern "system" fn(
            *mut c_void,
            *const c_char,
        ) -> Option<unsafe extern "system" fn()>,
    >,
}

fn open_layer() -> *mut c_void {
    let candidates = [
        "../target/release/libVkLayer_VKPACE_reduce_latency.so",
        "../target/debug/libVkLayer_VKPACE_reduce_latency.so",
        "./target/release/libVkLayer_VKPACE_reduce_latency.so",
        "./target/debug/libVkLayer_VKPACE_reduce_latency.so",
    ];
    for path in candidates {
        if let Ok(c) = CString::new(path) {
            let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !h.is_null() {
                return h;
            }
        }
    }
    panic!(
        "could not load built libVkLayer_VKPACE_reduce_latency.so — \
         run `cargo build` first"
    );
}

unsafe fn dlsym<F>(handle: *mut c_void, name: &CStr) -> Option<F> {
    let p = unsafe { libc::dlsym(handle, name.as_ptr()) };
    if p.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute_copy(&p) })
    }
}

#[test]
fn negotiate_returns_version_2() {
    let h = open_layer();
    let negotiate: unsafe extern "system" fn(*mut NegotiateLayerInterface) -> i32 =
        unsafe { dlsym(h, c"vkNegotiateLoaderLayerInterfaceVersion") }
            .expect("vkNegotiateLoaderLayerInterfaceVersion missing");

    // Pretend we're a loader that supports version 5.
    let mut s = NegotiateLayerInterface {
        s_type: 1,
        p_next: std::ptr::null_mut(),
        loader_layer_interface_version: 5,
        pfn_get_instance_proc_addr: None,
        pfn_get_device_proc_addr: None,
        pfn_get_physical_device_proc_addr: None,
    };
    let r = unsafe { negotiate(&mut s) };
    assert_eq!(r, 0, "expected VK_SUCCESS");
    assert_eq!(s.loader_layer_interface_version, 2);
    assert!(s.pfn_get_instance_proc_addr.is_some());
    assert!(s.pfn_get_device_proc_addr.is_some());
}

#[test]
fn gipa_resolves_known_global_entrypoints() {
    let h = open_layer();
    let gipa: unsafe extern "system" fn(
        *mut c_void,
        *const c_char,
    ) -> Option<unsafe extern "system" fn()> = unsafe { dlsym(h, c"VkPace_GetInstanceProcAddr") }
        .expect("VkPace_GetInstanceProcAddr missing");

    // Global queries: instance=NULL. The layer must answer for the loader's
    // baseline set even without a created instance.
    for name in [
        c"vkCreateInstance",
        c"vkGetInstanceProcAddr",
        c"vkGetDeviceProcAddr",
    ] {
        let p = unsafe { gipa(std::ptr::null_mut(), name.as_ptr()) };
        assert!(p.is_some(), "GIPA returned NULL for {:?}", name);
    }
}

#[test]
fn gipa_returns_none_for_unknown_global() {
    let h = open_layer();
    let gipa: unsafe extern "system" fn(
        *mut c_void,
        *const c_char,
    ) -> Option<unsafe extern "system" fn()> = unsafe { dlsym(h, c"VkPace_GetInstanceProcAddr") }
        .expect("VkPace_GetInstanceProcAddr missing");

    let p = unsafe {
        gipa(
            std::ptr::null_mut(),
            c"vkObviouslyNotARealFunction".as_ptr(),
        )
    };
    assert!(p.is_none());
}

#[test]
fn gipa_returns_none_for_null_name() {
    let h = open_layer();
    let gipa: unsafe extern "system" fn(
        *mut c_void,
        *const c_char,
    ) -> Option<unsafe extern "system" fn()> = unsafe { dlsym(h, c"VkPace_GetInstanceProcAddr") }
        .expect("VkPace_GetInstanceProcAddr missing");

    let p = unsafe { gipa(std::ptr::null_mut(), std::ptr::null()) };
    assert!(p.is_none());
}
