//! Visual palette + egui Visuals tuning.
//!
//! Goal: a dark theme that reads at a glance, with line colors that don't
//! collide and metric-tile chrome that gives the HUD a "tool" feel rather
//! than "default eframe demo".

use eframe::egui::{self, Color32, FontData, FontDefinitions, FontFamily, Rounding, Stroke};

/// JetBrains Mono Regular (OFL licensed; see assets/OFL.txt). Bundled
/// because the host font set on user machines is unpredictable — running
/// without it produces the default Ubuntu/ProggyClean pair, which is
/// fine but doesn't match the instrument-panel aesthetic.
const FONT_REGULAR: &[u8] = include_bytes!("../assets/JetBrainsMono-Regular.otf");
const FONT_MEDIUM: &[u8] = include_bytes!("../assets/JetBrainsMono-Medium.otf");

// ── Palette ────────────────────────────────────────────────────────────
// Backgrounds: layered greys, slight blue tint so the HUD reads as
// "instrument panel" rather than "Notepad".
pub const BG_WINDOW: Color32 = Color32::from_rgb(18, 20, 26);
pub const BG_PANEL: Color32 = Color32::from_rgb(24, 27, 35);
pub const BG_CARD: Color32 = Color32::from_rgb(31, 35, 45);
pub const BG_CARD_HOVER: Color32 = Color32::from_rgb(38, 43, 55);

// Foreground tiers — weak/normal/strong for text hierarchy.
pub const FG_WEAK: Color32 = Color32::from_rgb(120, 130, 150);
pub const FG_NORMAL: Color32 = Color32::from_rgb(200, 208, 222);

// Subtle separator + plot grid lines.
pub const SEPARATOR: Color32 = Color32::from_rgb(46, 51, 64);

// Status colours — used both for the connection pill and for the metric
// values when they cross a "good / warn / bad" threshold.
pub const OK_GREEN: Color32 = Color32::from_rgb(96, 220, 140);
pub const WARN_AMBER: Color32 = Color32::from_rgb(255, 188, 90);
pub const BAD_RED: Color32 = Color32::from_rgb(255, 110, 110);

// Plot line series colors. Chosen for distinguishability + colour-blind
// reasonableness: green / cyan / amber / pink / blue separates well under
// both deuteranopia and protanopia palettes.
pub const LINE_FPS: Color32 = Color32::from_rgb(120, 220, 160);
pub const LINE_FRAMETIME: Color32 = Color32::from_rgb(120, 200, 250);
pub const LINE_P50: Color32 = Color32::from_rgb(120, 200, 250);
pub const LINE_P99: Color32 = Color32::from_rgb(255, 188, 90);
pub const LINE_MAX: Color32 = Color32::from_rgb(255, 110, 110);

/// Install palette + fonts into the egui context. Called once at app
/// startup; everything else picks up the colors via `style` and the font
/// via `FontFamily::{Proportional, Monospace}`.
pub fn install(ctx: &egui::Context) {
    install_fonts(ctx);
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = BG_WINDOW;
    visuals.panel_fill = BG_PANEL;
    visuals.extreme_bg_color = BG_WINDOW;
    visuals.faint_bg_color = BG_CARD;
    visuals.code_bg_color = BG_CARD;
    visuals.override_text_color = Some(FG_NORMAL);
    visuals.hyperlink_color = LINE_FRAMETIME;
    visuals.selection.bg_fill = BG_CARD_HOVER;
    visuals.selection.stroke = Stroke::new(1.0, LINE_FRAMETIME);

    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.weak_bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, SEPARATOR);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, FG_NORMAL);

    visuals.widgets.inactive.bg_fill = BG_CARD;
    visuals.widgets.inactive.weak_bg_fill = BG_CARD;
    visuals.widgets.hovered.bg_fill = BG_CARD_HOVER;
    visuals.widgets.hovered.weak_bg_fill = BG_CARD_HOVER;
    visuals.widgets.active.bg_fill = BG_CARD_HOVER;
    visuals.widgets.active.weak_bg_fill = BG_CARD_HOVER;

    visuals.window_rounding = Rounding::same(8.0);
    visuals.menu_rounding = Rounding::same(6.0);

    ctx.set_visuals(visuals);

    // Slightly tighter spacing than default — the HUD packs a lot per row.
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
    style.spacing.window_margin = egui::Margin::ZERO;
    // HUD is read-only — disable label selection globally so dragging
    // anywhere over the metrics doesn't highlight text or steal focus.
    style.interaction.selectable_labels = false;
    ctx.set_style(style);
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts
        .font_data
        .insert("jbmono".into(), FontData::from_static(FONT_REGULAR));
    fonts
        .font_data
        .insert("jbmono-medium".into(), FontData::from_static(FONT_MEDIUM));
    // Make our font the primary for both families. Mono everywhere fits a
    // numeric instrument panel and keeps metric columns aligned.
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "jbmono".into());
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .insert(0, "jbmono".into());
    // Register the medium variant under a custom family so RichText::strong
    // picks it up via style.text_styles below.
    fonts
        .families
        .entry(FontFamily::Name("jbmono-medium".into()))
        .or_default()
        .insert(0, "jbmono-medium".into());
    ctx.set_fonts(fonts);
}

/// Colour for an fps metric value. ≥ 120 fps green, ≥ 60 amber, lower red,
/// 0 (no data) muted.
pub fn fps_color(fps: f64) -> Color32 {
    if fps <= 0.0 {
        FG_WEAK
    } else if fps >= 120.0 {
        OK_GREEN
    } else if fps >= 60.0 {
        WARN_AMBER
    } else {
        BAD_RED
    }
}

/// Colour for a latency-in-µs metric. <16 ms (60 Hz frame) green,
/// <33 ms amber, ≥33 ms red, 0 (no sample) muted.
pub fn latency_color(us: u64) -> Color32 {
    if us == 0 {
        FG_WEAK
    } else if us < 16_000 {
        OK_GREEN
    } else if us < 33_000 {
        WARN_AMBER
    } else {
        BAD_RED
    }
}
