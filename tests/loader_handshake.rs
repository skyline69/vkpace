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
    const LIB: &str = "libVkLayer_VKPACE_reduce_latency.so";
    const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
    let manifest = std::path::Path::new(MANIFEST_DIR);

    // Search paths in priority order. We include `deps/` because some Cargo
    // versions place the cdylib there for test runs, plus the standard
    // top-level locations a regular `cargo build` produces.
    let candidates: Vec<std::path::PathBuf> = ["release", "debug"]
        .iter()
        .flat_map(|profile| {
            [
                manifest.join("target").join(profile).join(LIB),
                manifest.join("target").join(profile).join("deps").join(LIB),
            ]
        })
        .chain(std::iter::once(std::path::PathBuf::from(LIB)))
        .collect();

    let mut tried = Vec::new();
    for path in &candidates {
        let Some(s) = path.to_str() else { continue };
        if !path.exists() {
            tried.push(s.to_string());
            continue;
        }
        if let Ok(c) = CString::new(s) {
            let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !h.is_null() {
                return h;
            }
            tried.push(format!("{s} (dlopen failed)"));
        }
    }

    // Last-chance: if no prebuilt cdylib exists yet, build it ourselves.
    // CI hits this when `cargo test` is the entry point.
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "--lib"])
        .current_dir(MANIFEST_DIR)
        .status();
    if let Ok(s) = status
        && s.success()
    {
        let p = manifest.join("target").join("debug").join(LIB);
        if let Some(ps) = p.to_str()
            && let Ok(c) = CString::new(ps)
        {
            let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !h.is_null() {
                return h;
            }
        }
    }

    panic!("could not load libVkLayer_VKPACE_reduce_latency.so; tried: {tried:?}");
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
