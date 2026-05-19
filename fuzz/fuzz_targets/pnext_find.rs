#![no_main]
//! Fuzz the pNext chain walker. We fabricate a chain of `(s_type, p_next)`
//! headers from the input bytes. The walker should never deref past the
//! end of the slice we hand it, regardless of what `p_next` values look
//! like (`null`, dangling, anything).

use libfuzzer_sys::fuzz_target;

#[repr(C)]
struct Header {
    s_type: i32,
    p_next: *const Header,
}

fuzz_target!(|data: &[u8]| {
    if data.len() < std::mem::size_of::<Header>() {
        return;
    }
    let n = data.len() / std::mem::size_of::<Header>();
    if n == 0 {
        return;
    }
    // Build a vector of headers from the raw bytes; link them so the
    // walker actually traverses the chain. The terminating header's
    // p_next stays whatever bit-pattern came from the input — the walker
    // must tolerate a wild value.
    let mut headers: Vec<Header> = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * std::mem::size_of::<Header>();
        let s_type = i32::from_ne_bytes(data[off..off + 4].try_into().unwrap());
        let p_next_bytes: [u8; std::mem::size_of::<usize>()] = data
            [off + 8..off + 8 + std::mem::size_of::<usize>()]
            .try_into()
            .unwrap_or([0u8; std::mem::size_of::<usize>()]);
        let p_next = usize::from_ne_bytes(p_next_bytes) as *const Header;
        headers.push(Header { s_type, p_next });
    }
    // Wire the chain in-order so the walker traverses every entry.
    for i in 0..headers.len() - 1 {
        let next_ptr = &headers[i + 1] as *const Header;
        headers[i].p_next = next_ptr;
    }
    headers.last_mut().unwrap().p_next = std::ptr::null();
    let head: *const std::ffi::c_void = headers.as_ptr() as *const std::ffi::c_void;
    unsafe {
        let _ = VkLayer_VKPACE_reduce_latency::__fuzz_api::pnext_find_any(head);
    }
});
