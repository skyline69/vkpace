# vkpace

Vulkan implicit layer that reduces input latency by exposing `VK_AMD_anti_lag`
or `VK_NV_low_latency2` on top of any compliant driver. Apps that opt into
those extensions get the same frame pacing as NVIDIA Reflex or AMD Anti-Lag,
regardless of which GPU sits underneath.

Pure Rust, single `cdylib`, no driver patches.

## Build

```
cargo build --release
```

Produces `target/release/libVkLayer_VKPACE_reduce_latency.so` and a matching
manifest JSON with an absolute `library_path`.

For a binary that runs on any x86-64-v3 box (Haswell or newer) — useful
when distributing builds — use the bundled alias:

```
cargo portable
```

Output lands in `target/release-portable/`.

## Use it without installing

Set Steam launch options on the game you want to test:

```
VK_ADD_LAYER_PATH=/path/to/vkpace/target/release VK_LOADER_LAYERS_ENABLE=VK_LAYER_VKPACE_reduce_latency VKPACE_REFLEX=1 %command%
```

If the app supports Reflex in its graphics menu, leave that setting *off* so
vkpace is the only pacer.

## Install system wide

```
install -Dm755 target/release/libVkLayer_VKPACE_reduce_latency.so \
    ~/.local/lib/libVkLayer_VKPACE_reduce_latency.so
install -Dm644 target/release/VkLayer_VKPACE_reduce_latency.json \
    ~/.local/share/vulkan/implicit_layer.d/VkLayer_VKPACE_reduce_latency.json
```

After install, the layer loads automatically. Set `DISABLE_VKPACE=1` to opt
out per process.

## Config

Environment variables (all optional):

```
VKPACE_REFLEX=1                   # expose VK_NV_low_latency2 (default: VK_AMD_anti_lag)
VKPACE_SPOOF_NVIDIA=1             # rewrite PhysicalDeviceProperties to look NVIDIA
VKPACE_SPOOF_MODEL=RTX_5090       # any preset key from RTX_2060 through RTX_5090
VKPACE_FORCE_DECOUPLED=1          # assume decoupled sim thread (run drain controller)
VKPACE_FPS_CAP=144                # hard FPS cap, combined with any app cap
VKPACE_VRR=1                      # soft FPS target = display refresh - VKPACE_VRR_OFFSET
VKPACE_VRR_OFFSET=3               # Hz subtracted from refresh for the soft target
VKPACE_LL2_WAIT_BUDGET_US=4000    # max time we hold the Reflex sleep semaphore
VKPACE_LOG=info                   # tracing-subscriber env filter
VKPACE_STATS_INTERVAL=5           # seconds between counter snapshot logs (0 = off)
VKPACE_TELEMETRY_SOCKET=/tmp/v.sock  # opt-in unix socket that streams per-present JSON
```

Per-app overrides live in `~/.config/vkpace/config.toml`:

```toml
[app."cs2"]
expose_reflex = true
fps_cap       = 300

[app."Marvel-Win64-Shipping.exe"]
expose_reflex   = true
spoof_nvidia    = true
spoof_model     = "RTX_5090"
force_decoupled = true
```

TOML keys take precedence over environment.

## Requirements

Any GPU and driver that support `VK_KHR_synchronization2`,
`VK_KHR_calibrated_timestamps`, and `VK_EXT_host_query_reset`. That covers
every RTX 20 series and newer, AMD RDNA1 and newer, Intel Xe, and any
modern Mesa driver.

## Intel Xe desktops

Intel iGPUs are a common Wayland-desktop case where many games refuse to
show a Reflex toggle because the driver reports `vendorID = 0x8086`. Set
`VKPACE_SPOOF_NVIDIA=1` (optionally with `VKPACE_SPOOF_MODEL`) so the app's
hardware probe sees an NVIDIA GPU and exposes the toggle; the layer still
runs on the actual Xe device underneath. Per-app TOML overrides apply the
same way as on AMD.

## Per-app config matching

`pApplicationInfo->pApplicationName` is the primary match key. Native Linux
apps frequently leave that field null; vkpace falls back to the basename
of `/proc/self/exe`, so a `[app."my-game"]` stanza still matches when
launched as `./my-game`. Proton wrappers set the `.exe` filename, so
`[app."Game-Win64-Shipping.exe"]` continues to work unchanged.

## Present-time source

Frame-end timestamps for the Reflex overlay come from `VK_KHR_present_wait`
when the driver exposes it (most current Mesa + NVIDIA), and fall back to
`VK_GOOGLE_display_timing` otherwise. The GOOGLE path is FIFO-ordered, so
correlation is one present_id off if the app drops a present, but the
overlay still gets non-zero timings on older Mesa.

## Inspiration

[`Korthos-Software/low_latency_layer`](https://github.com/Korthos-Software/low_latency_layer)
proved the approach was viable on Linux. vkpace is an independent ground-up
Rust implementation; no source is shared.

## License

MIT.
