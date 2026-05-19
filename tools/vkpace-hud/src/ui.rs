//! egui HUD application.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use eframe::egui::{self, Color32, Frame, Margin, RichText, Rounding, Stroke};
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::state::SharedState;
use crate::stats;
use crate::theme;

/// Time window plotted on the X axis (60 s rolling history).
const WINDOW_NS: u64 = 60_000_000_000;
/// Plot bin width — 500 ms keeps the line stable at typical fps.
const BIN_NS: u64 = 500_000_000;
/// Live-stats window for the top-strip numbers.
const STATS_WINDOW_NS: u64 = 1_000_000_000;

pub struct HudApp {
    state: Arc<SharedState>,
}

impl HudApp {
    pub fn new(state: Arc<SharedState>) -> Self {
        Self { state }
    }
}

impl eframe::App for HudApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(16));

        let connected = self.state.connected.load(Ordering::Acquire);
        let now_ns = self.state.latest_ts().unwrap_or(0);
        let snapshot = if now_ns > 0 {
            self.state.snapshot_since(now_ns.saturating_sub(WINDOW_NS))
        } else {
            Vec::new()
        };
        let live = stats::live_stats(&snapshot, now_ns, STATS_WINDOW_NS);
        let drops: u64 = stats::frame_gaps(&snapshot)
            .iter()
            .map(|p| p[1] as u64)
            .sum();

        top_strip(ctx, connected, &live, drops);
        plots(ctx, &snapshot, now_ns, live.samples);
    }

    fn on_exit(&mut self) {
        self.state.stop.store(true, Ordering::Release);
    }
}

// ── Top strip ─────────────────────────────────────────────────────────

fn top_strip(ctx: &egui::Context, connected: bool, live: &stats::LiveStats, drops: u64) {
    egui::TopBottomPanel::top("nums")
        .frame(
            Frame::none()
                .fill(theme::BG_PANEL)
                .inner_margin(Margin::symmetric(16.0, 14.0))
                .stroke(Stroke::new(1.0, theme::SEPARATOR)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                metric_tile(
                    ui,
                    "fps",
                    format!("{:.1}", live.fps),
                    theme::fps_color(live.fps),
                );
                metric_tile(
                    ui,
                    "p50",
                    us_label(live.p50_us),
                    theme::latency_color(live.p50_us),
                );
                metric_tile(
                    ui,
                    "p99",
                    us_label(live.p99_us),
                    theme::latency_color(live.p99_us),
                );
                metric_tile(
                    ui,
                    "max",
                    us_label(live.max_us),
                    theme::latency_color(live.max_us),
                );
                metric_tile(
                    ui,
                    "samples",
                    live.samples.to_string(),
                    if live.samples == 0 {
                        theme::FG_WEAK
                    } else {
                        theme::FG_NORMAL
                    },
                );
                metric_tile(
                    ui,
                    "drops/60s",
                    drops.to_string(),
                    if drops == 0 {
                        theme::OK_GREEN
                    } else {
                        theme::WARN_AMBER
                    },
                );

                // Push the connection pill to the right edge.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    connection_pill(ui, connected);
                });
            });
        });
}

fn metric_tile(ui: &mut egui::Ui, label: &str, value: String, value_color: Color32) {
    Frame::none()
        .fill(theme::BG_CARD)
        .rounding(Rounding::same(6.0))
        .inner_margin(Margin::symmetric(12.0, 8.0))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(value)
                        .size(24.0)
                        .color(value_color)
                        .strong(),
                );
                ui.label(
                    RichText::new(label)
                        .size(10.0)
                        .color(theme::FG_WEAK)
                        .text_style(egui::TextStyle::Small),
                );
            });
        });
}

fn connection_pill(ui: &mut egui::Ui, connected: bool) {
    let (text, color, bg) = if connected {
        (
            "connected",
            theme::OK_GREEN,
            Color32::from_rgba_unmultiplied(96, 220, 140, 36),
        )
    } else {
        (
            "disconnected",
            theme::BAD_RED,
            Color32::from_rgba_unmultiplied(255, 110, 110, 36),
        )
    };
    Frame::none()
        .fill(bg)
        .rounding(Rounding::same(12.0))
        .inner_margin(Margin::symmetric(10.0, 4.0))
        .stroke(Stroke::new(1.0, color))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 4.0, color);
                ui.label(RichText::new(text).color(color).size(12.0).strong());
            });
        });
}

// ── Plots ─────────────────────────────────────────────────────────────

fn plots(ctx: &egui::Context, snapshot: &[crate::state::Record], now_ns: u64, samples: usize) {
    egui::CentralPanel::default()
        .frame(
            Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(Margin::same(10.0)),
        )
        .show(ctx, |ui| {
            let total = ui.available_height();
            // Three panels stacked; leave 8 px of breathing room between.
            let plot_h = (total / 3.0 - 12.0).max(80.0);

            plot_card(ui, "fps", plot_h, |inner| {
                let pts = stats::bin_records(snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                    b.len() as f64 * 1_000_000_000.0 / BIN_NS as f64
                });
                Plot::new("fps_plot")
                    .height(inner.available_height())
                    .legend(Legend::default().background_alpha(0.0))
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .show_x(false)
                    .show_y(true)
                    .show(inner, |plot_ui| {
                        plot_ui.line(
                            Line::new(PlotPoints::from(pts))
                                .color(theme::LINE_FPS)
                                .width(2.0)
                                .name("fps"),
                        );
                    });
            });

            plot_card(ui, "click-to-photon (µs)", plot_h, |inner| {
                if samples == 0 {
                    no_latency_hint(inner);
                    return;
                }
                let p50_points = stats::bin_records(snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                    bin_percentile(b, 0.50)
                });
                let p99_points = stats::bin_records(snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                    bin_percentile(b, 0.99)
                });
                let max_points = stats::bin_records(snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                    b.iter().map(|r| r.latency_us as f64).fold(0.0, f64::max)
                });
                Plot::new("latency_plot")
                    .height(inner.available_height())
                    .legend(Legend::default().background_alpha(0.0))
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .show(inner, |plot_ui| {
                        plot_ui.line(
                            Line::new(PlotPoints::from(p50_points))
                                .color(theme::LINE_P50)
                                .width(1.5)
                                .name("p50"),
                        );
                        plot_ui.line(
                            Line::new(PlotPoints::from(p99_points))
                                .color(theme::LINE_P99)
                                .width(1.5)
                                .name("p99"),
                        );
                        plot_ui.line(
                            Line::new(PlotPoints::from(max_points))
                                .color(theme::LINE_MAX)
                                .width(1.2)
                                .name("max"),
                        );
                    });
            });

            plot_card(ui, "frame-time (ms)", plot_h, |inner| {
                let pts = stats::bin_records(snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                    let secs_per_bin = BIN_NS as f64 / 1e9;
                    secs_per_bin * 1000.0 / b.len() as f64
                });
                Plot::new("frametime_plot")
                    .height(inner.available_height())
                    .legend(Legend::default().background_alpha(0.0))
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .show(inner, |plot_ui| {
                        plot_ui.line(
                            Line::new(PlotPoints::from(pts))
                                .color(theme::LINE_FRAMETIME)
                                .width(2.0)
                                .name("frame-time"),
                        );
                    });
            });
        });
}

fn plot_card(ui: &mut egui::Ui, title: &str, h: f32, body: impl FnOnce(&mut egui::Ui)) {
    Frame::none()
        .fill(theme::BG_PANEL)
        .rounding(Rounding::same(8.0))
        .stroke(Stroke::new(1.0, theme::SEPARATOR))
        .inner_margin(Margin::symmetric(10.0, 8.0))
        .show(ui, |ui| {
            ui.set_height(h);
            ui.label(
                RichText::new(title)
                    .color(theme::FG_WEAK)
                    .size(11.0)
                    .strong(),
            );
            ui.add_space(2.0);
            body(ui);
        });
    ui.add_space(4.0);
}

fn no_latency_hint(ui: &mut egui::Ui) {
    let h = ui.available_height();
    ui.vertical_centered(|ui| {
        ui.add_space((h / 2.0 - 10.0).max(0.0));
        ui.label(
            RichText::new("no latency samples")
                .color(theme::FG_NORMAL)
                .size(13.0)
                .strong(),
        );
        ui.label(
            RichText::new("app is not calling vkSetLatencyMarkerNV (Reflex)")
                .color(theme::FG_WEAK)
                .size(11.0)
                .italics(),
        );
    });
}

// ── Formatters ────────────────────────────────────────────────────────

fn us_label(us: u64) -> String {
    if us == 0 {
        "—".into()
    } else if us >= 1_000 {
        format!("{:.2} ms", us as f64 / 1_000.0)
    } else {
        format!("{us} µs")
    }
}

/// Percentile within a single bin's slice of records, ignoring zero
/// samples (no input-marker → no meaningful latency).
fn bin_percentile(bucket: &[&crate::state::Record], q: f64) -> f64 {
    let mut v: Vec<u64> = bucket
        .iter()
        .map(|r| r.latency_us)
        .filter(|&x| x > 0)
        .collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable();
    let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
    v[idx.min(v.len() - 1)] as f64
}
