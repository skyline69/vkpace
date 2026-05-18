use std::env;
use std::fs;
use std::path::PathBuf;

pub const LAYER_NAME: &str = "VK_LAYER_VKPACE_reduce_latency";

pub const NVIDIA_VENDOR_ID: u32 = 0x10DE;

const REFLEX_ENV: &str = "VKPACE_REFLEX";
const SPOOF_NVIDIA_ENV: &str = "VKPACE_SPOOF_NVIDIA";
const SPOOF_MODEL_ENV: &str = "VKPACE_SPOOF_MODEL";
const FORCE_DECOUPLED_ENV: &str = "VKPACE_FORCE_DECOUPLED";
const FPS_CAP_ENV: &str = "VKPACE_FPS_CAP";
const CONFIG_PATH_ENV: &str = "VKPACE_CONFIG";

const DECOUPLED_SIMULATION_APPS: &[&str] = &["Marvel-Win64-Shipping.exe"];

/// (preset key, PCI device ID, marketing name)
///
/// IDs are the most common consumer variant for each model — Vulkan apps
/// almost never key on the exact die revision, but the table is exhaustive
/// enough that anything checking for a 20/30/40/50-series card will be
/// satisfied.
pub const NVIDIA_PRESETS: &[(&str, u32, &str)] = &[
    // Turing (RTX 20)
    ("RTX_2060", 0x1F08, "NVIDIA GeForce RTX 2060"),
    ("RTX_2060_SUPER", 0x1F47, "NVIDIA GeForce RTX 2060 SUPER"),
    ("RTX_2070", 0x1F02, "NVIDIA GeForce RTX 2070"),
    ("RTX_2070_SUPER", 0x1E84, "NVIDIA GeForce RTX 2070 SUPER"),
    ("RTX_2080", 0x1E87, "NVIDIA GeForce RTX 2080"),
    ("RTX_2080_SUPER", 0x1E81, "NVIDIA GeForce RTX 2080 SUPER"),
    ("RTX_2080_TI", 0x1E07, "NVIDIA GeForce RTX 2080 Ti"),
    // Ampere (RTX 30)
    ("RTX_3050", 0x2507, "NVIDIA GeForce RTX 3050"),
    ("RTX_3060", 0x2504, "NVIDIA GeForce RTX 3060"),
    ("RTX_3060_TI", 0x2489, "NVIDIA GeForce RTX 3060 Ti"),
    ("RTX_3070", 0x2484, "NVIDIA GeForce RTX 3070"),
    ("RTX_3070_TI", 0x2482, "NVIDIA GeForce RTX 3070 Ti"),
    ("RTX_3080", 0x2206, "NVIDIA GeForce RTX 3080"),
    ("RTX_3080_TI", 0x2208, "NVIDIA GeForce RTX 3080 Ti"),
    ("RTX_3090", 0x2204, "NVIDIA GeForce RTX 3090"),
    ("RTX_3090_TI", 0x2203, "NVIDIA GeForce RTX 3090 Ti"),
    // Ada Lovelace (RTX 40)
    ("RTX_4060", 0x2882, "NVIDIA GeForce RTX 4060"),
    ("RTX_4060_TI", 0x2803, "NVIDIA GeForce RTX 4060 Ti"),
    ("RTX_4070", 0x2786, "NVIDIA GeForce RTX 4070"),
    ("RTX_4070_SUPER", 0x2783, "NVIDIA GeForce RTX 4070 SUPER"),
    ("RTX_4070_TI", 0x2782, "NVIDIA GeForce RTX 4070 Ti"),
    (
        "RTX_4070_TI_SUPER",
        0x2705,
        "NVIDIA GeForce RTX 4070 Ti SUPER",
    ),
    ("RTX_4080", 0x2704, "NVIDIA GeForce RTX 4080"),
    ("RTX_4080_SUPER", 0x2702, "NVIDIA GeForce RTX 4080 SUPER"),
    ("RTX_4090", 0x2684, "NVIDIA GeForce RTX 4090"),
    // Blackwell (RTX 50)
    ("RTX_5060", 0x2D04, "NVIDIA GeForce RTX 5060"),
    ("RTX_5060_TI", 0x2D03, "NVIDIA GeForce RTX 5060 Ti"),
    ("RTX_5070", 0x2C05, "NVIDIA GeForce RTX 5070"),
    ("RTX_5070_TI", 0x2C04, "NVIDIA GeForce RTX 5070 Ti"),
    ("RTX_5080", 0x2C02, "NVIDIA GeForce RTX 5080"),
    ("RTX_5090", 0x2B85, "NVIDIA GeForce RTX 5090"),
];

pub const DEFAULT_PRESET: &str = "RTX_5090";

#[derive(Debug, Clone, Copy)]
pub struct SpoofProfile {
    pub device_id: u32,
    pub device_name: &'static str,
}

/// Layer-wide configuration. Resolved once at first `vkGetInstanceProcAddr`
/// call. Precedence (highest to lowest):
///
/// 1. Per-app override in TOML matching `pApplicationName`
/// 2. `[defaults]` section in TOML
/// 3. Environment variables
/// 4. Built-in defaults
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub expose_reflex: bool,
    pub spoof_nvidia: bool,
    pub force_decoupled: bool,
    pub spoof_profile: SpoofProfile,
    /// Hard FPS cap applied in the delay controller. 0 = no cap.
    pub fps_cap: u32,
    /// Per-app overrides, sourced from TOML. Keyed by exact match against
    /// `pApplicationInfo.pApplicationName`.
    pub per_app: Vec<AppOverride>,
}

#[derive(Debug, Clone)]
pub struct AppOverride {
    pub app_name: String,
    pub expose_reflex: Option<bool>,
    pub spoof_nvidia: Option<bool>,
    pub force_decoupled: Option<bool>,
    pub fps_cap: Option<u32>,
    pub spoof_model: Option<String>,
}

impl LayerConfig {
    pub fn from_env() -> Self {
        let spoof_nvidia = env_flag(SPOOF_NVIDIA_ENV);
        let spoof_profile = resolve_spoof_profile(None);
        let fps_cap = env::var(FPS_CAP_ENV)
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        let per_app = load_toml().unwrap_or_default();
        Self {
            expose_reflex: env_flag(REFLEX_ENV),
            spoof_nvidia,
            force_decoupled: env_flag(FORCE_DECOUPLED_ENV),
            spoof_profile,
            fps_cap,
            per_app,
        }
    }

    /// Apply any matching per-app override, returning a finalized snapshot
    /// for the running instance.
    pub fn finalize_for_app(&self, app_name: Option<&str>) -> Self {
        let mut out = self.clone();
        if let Some(name) = app_name
            && let Some(over) = self.per_app.iter().find(|o| o.app_name == name)
        {
            if let Some(v) = over.expose_reflex {
                out.expose_reflex = v;
            }
            if let Some(v) = over.spoof_nvidia {
                out.spoof_nvidia = v;
            }
            if let Some(v) = over.force_decoupled {
                out.force_decoupled = v;
            }
            if let Some(v) = over.fps_cap {
                out.fps_cap = v;
            }
            if let Some(ref model) = over.spoof_model {
                out.spoof_profile = resolve_spoof_profile(Some(model));
            }
            tracing::info!(app = %name, "applied per-app config override");
        }
        out
    }

    pub fn is_known_decoupled(&self, app_name: Option<&str>) -> bool {
        self.force_decoupled || app_name.is_some_and(|n| DECOUPLED_SIMULATION_APPS.contains(&n))
    }

    /// `min_delay` in nanoseconds for an FPS cap. 0 = uncapped.
    pub fn fps_cap_min_delay_ns(&self) -> u64 {
        if self.fps_cap == 0 {
            0
        } else {
            1_000_000_000 / self.fps_cap as u64
        }
    }
}

fn env_flag(name: &str) -> bool {
    env::var(name).ok().as_deref() == Some("1")
}

fn resolve_spoof_profile(explicit: Option<&str>) -> SpoofProfile {
    let env_val = env::var(SPOOF_MODEL_ENV).ok();
    let requested = explicit.map(|s| s.to_owned()).or(env_val);
    let key = requested.as_deref().unwrap_or(DEFAULT_PRESET);
    let key_upper = key.to_ascii_uppercase();

    if let Some(&(_, device_id, device_name)) =
        NVIDIA_PRESETS.iter().find(|(k, _, _)| *k == key_upper)
    {
        return SpoofProfile {
            device_id,
            device_name,
        };
    }

    // Unknown preset: warn-and-fall-back to the default. The first env-var
    // read happens before tracing is initialized, so we use eprintln here.
    if requested.is_some() {
        eprintln!(
            "[vkpace] unknown {SPOOF_MODEL_ENV}={key}; \
             valid keys: {}",
            NVIDIA_PRESETS
                .iter()
                .map(|(k, _, _)| *k)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let &(_, device_id, device_name) = NVIDIA_PRESETS
        .iter()
        .find(|(k, _, _)| *k == DEFAULT_PRESET)
        .expect("default preset must exist");
    SpoofProfile {
        device_id,
        device_name,
    }
}

/// Locate the TOML config: env var override, then `$XDG_CONFIG_HOME` or
/// `$HOME/.config/vkpace/config.toml`. Missing file = no
/// overrides (return empty list).
fn locate_toml() -> Option<PathBuf> {
    if let Ok(p) = env::var(CONFIG_PATH_ENV) {
        return Some(PathBuf::from(p));
    }
    let base = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("vkpace").join("config.toml"))
}

/// Minimal hand-rolled TOML reader. We accept exactly the shape:
///
/// ```toml
/// [app."Marvel-Win64-Shipping.exe"]
/// expose_reflex = true
/// spoof_nvidia = true
/// spoof_model = "RTX_4090"
/// force_decoupled = true
/// fps_cap = 144
/// ```
///
/// Anything else is ignored with a warning. We deliberately don't pull in
/// `toml`/`serde` — keeping the dep tree small matters for a layer that may
/// be loaded into every Vulkan process on the system.
fn load_toml() -> Option<Vec<AppOverride>> {
    let path = locate_toml()?;
    let contents = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            eprintln!("[vkpace] config read failed ({path:?}): {e}");
            return None;
        }
    };
    let mut overrides: Vec<AppOverride> = Vec::new();
    let mut current: Option<AppOverride> = None;
    for (lineno, raw_line) in contents.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("[app.") {
            if let Some(prev) = current.take() {
                overrides.push(prev);
            }
            let name = rest.trim_end_matches(']').trim_matches('"');
            current = Some(AppOverride {
                app_name: name.to_owned(),
                expose_reflex: None,
                spoof_nvidia: None,
                force_decoupled: None,
                fps_cap: None,
                spoof_model: None,
            });
            continue;
        }
        let Some(cur) = current.as_mut() else {
            eprintln!(
                "[vkpace] config line {} outside [app.\"…\"] section: {}",
                lineno + 1,
                line
            );
            continue;
        };
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"');
        match k {
            "expose_reflex" => cur.expose_reflex = parse_bool(v),
            "spoof_nvidia" => cur.spoof_nvidia = parse_bool(v),
            "force_decoupled" => cur.force_decoupled = parse_bool(v),
            "fps_cap" => cur.fps_cap = v.parse().ok(),
            "spoof_model" => cur.spoof_model = Some(v.to_owned()),
            unknown => eprintln!("[vkpace] unknown config key: {unknown}"),
        }
    }
    if let Some(prev) = current.take() {
        overrides.push(prev);
    }
    Some(overrides)
}

fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_preset_resolves_to_5090() {
        let p = NVIDIA_PRESETS
            .iter()
            .find(|(k, _, _)| *k == DEFAULT_PRESET)
            .unwrap();
        assert_eq!(p.1, 0x2B85);
        assert!(p.2.contains("5090"));
    }

    #[test]
    fn presets_cover_all_four_generations() {
        let has_gen = |prefix: &str| NVIDIA_PRESETS.iter().any(|(k, _, _)| k.starts_with(prefix));
        assert!(has_gen("RTX_2"));
        assert!(has_gen("RTX_3"));
        assert!(has_gen("RTX_4"));
        assert!(has_gen("RTX_5"));
    }

    #[test]
    fn preset_keys_unique() {
        let mut keys: Vec<&str> = NVIDIA_PRESETS.iter().map(|(k, _, _)| *k).collect();
        keys.sort();
        let n = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), n, "duplicate preset key");
    }

    #[test]
    fn known_decoupled_app() {
        let cfg = LayerConfig {
            expose_reflex: false,
            spoof_nvidia: false,
            force_decoupled: false,
            spoof_profile: SpoofProfile {
                device_id: 0,
                device_name: "",
            },
            fps_cap: 0,
            per_app: Vec::new(),
        };
        assert!(cfg.is_known_decoupled(Some("Marvel-Win64-Shipping.exe")));
        assert!(!cfg.is_known_decoupled(Some("unknown.exe")));
        assert!(!cfg.is_known_decoupled(None));
    }

    #[test]
    fn force_decoupled_overrides() {
        let cfg = LayerConfig {
            expose_reflex: false,
            spoof_nvidia: false,
            force_decoupled: true,
            spoof_profile: SpoofProfile {
                device_id: 0,
                device_name: "",
            },
            fps_cap: 0,
            per_app: Vec::new(),
        };
        assert!(cfg.is_known_decoupled(None));
        assert!(cfg.is_known_decoupled(Some("anything.exe")));
    }

    #[test]
    fn fps_cap_to_min_delay_ns() {
        let mut cfg = LayerConfig {
            expose_reflex: false,
            spoof_nvidia: false,
            force_decoupled: false,
            spoof_profile: SpoofProfile {
                device_id: 0,
                device_name: "",
            },
            fps_cap: 0,
            per_app: Vec::new(),
        };
        assert_eq!(cfg.fps_cap_min_delay_ns(), 0);
        cfg.fps_cap = 60;
        assert_eq!(cfg.fps_cap_min_delay_ns(), 16_666_666);
        cfg.fps_cap = 240;
        assert_eq!(cfg.fps_cap_min_delay_ns(), 4_166_666);
    }

    #[test]
    fn per_app_override_applied() {
        let cfg = LayerConfig {
            expose_reflex: false,
            spoof_nvidia: false,
            force_decoupled: false,
            spoof_profile: SpoofProfile {
                device_id: 0,
                device_name: "",
            },
            fps_cap: 0,
            per_app: vec![AppOverride {
                app_name: "test.exe".into(),
                expose_reflex: Some(true),
                spoof_nvidia: Some(true),
                force_decoupled: None,
                fps_cap: Some(144),
                spoof_model: Some("RTX_4090".into()),
            }],
        };
        let final_cfg = cfg.finalize_for_app(Some("test.exe"));
        assert!(final_cfg.expose_reflex);
        assert!(final_cfg.spoof_nvidia);
        assert_eq!(final_cfg.fps_cap, 144);
        assert_eq!(final_cfg.spoof_profile.device_id, 0x2684);
    }

    #[test]
    fn per_app_override_skipped_when_no_match() {
        let cfg = LayerConfig {
            expose_reflex: false,
            spoof_nvidia: false,
            force_decoupled: false,
            spoof_profile: SpoofProfile {
                device_id: 0,
                device_name: "",
            },
            fps_cap: 0,
            per_app: vec![AppOverride {
                app_name: "test.exe".into(),
                expose_reflex: Some(true),
                spoof_nvidia: None,
                force_decoupled: None,
                fps_cap: None,
                spoof_model: None,
            }],
        };
        let final_cfg = cfg.finalize_for_app(Some("other.exe"));
        assert!(!final_cfg.expose_reflex);
    }
}
