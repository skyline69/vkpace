//! vkpace-hud — live telemetry HUD for the vkpace Vulkan layer.
//!
//! Connects to the unix socket the layer exposes when
//! `VKPACE_TELEMETRY_SOCKET=/path/to/sock` is set, parses one record per
//! present, and graphs fps / latency percentiles / dropped-frame gaps
//! over a rolling 60 s window.

mod parse;
mod reader;
mod state;
mod stats;
mod theme;
mod ui;

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;

const DEFAULT_SOCKET: &str = "/tmp/vkpace.sock";

fn main() -> eframe::Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET));

    let state = Arc::new(state::SharedState::new());
    let _reader = reader::spawn(path.clone(), state.clone());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 540.0])
            .with_title("vkpace HUD"),
        ..Default::default()
    };
    let app_state = state.clone();
    eframe::run_native(
        "vkpace HUD",
        options,
        Box::new(move |cc| {
            theme::install(&cc.egui_ctx);
            Ok(Box::new(ui::HudApp::new(app_state)))
        }),
    )
}
