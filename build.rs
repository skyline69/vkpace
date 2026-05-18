use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_in = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("VkLayer_VKPACE_reduce_latency.json.in");

    let target_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap())
        .ancestors()
        .nth(3)
        .unwrap()
        .to_path_buf();

    let lib_path = target_dir.join("libVkLayer_VKPACE_reduce_latency.so");
    let manifest_out = target_dir.join("VkLayer_VKPACE_reduce_latency.json");

    let template = fs::read_to_string(&manifest_in).expect("manifest template missing");
    let rendered = template.replace("@LIB_PATH@", &lib_path.display().to_string());
    fs::write(&manifest_out, rendered).expect("write manifest");

    println!("cargo:rerun-if-changed={}", manifest_in.display());
}
