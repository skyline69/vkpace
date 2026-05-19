#![no_main]
//! Fuzz the hand-rolled TOML loader on arbitrary UTF-8 input. The loader
//! must not panic, OOM, or hang for any input — invalid configs just
//! produce zero overrides + a warning.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    // Cap input size — pathological inputs (10 MB of `[app."`) waste fuzz
    // budget without exercising any new branches.
    if s.len() > 64 * 1024 {
        return;
    }
    VkLayer_VKPACE_reduce_latency::__fuzz_api::config_toml_load(s);
});
