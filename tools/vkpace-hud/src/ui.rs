//! egui HUD application.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints};

use crate::state::SharedState;
use crate::stats;

/// Time window plotted on the X axis (60 s rolling history).
const WINDOW_NS: u64 = 60_000_000_000;
/// Plot bin width — 100 ms per point keeps the line smooth without
/// drowning the renderer in geometry.
const BIN_NS: u64 = 100_000_000;
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
        // ~60 Hz refresh; cheap because we hold no GPU work between frames.
        ctx.request_repaint_after(Duration::from_millis(16));

        let connected = self.state.connected.load(Ordering::Acquire);
        let now_ns = self.state.latest_ts().unwrap_or(0);
        let snapshot = if now_ns > 0 {
            self.state.snapshot_since(now_ns.saturating_sub(WINDOW_NS))
        } else {
            Vec::new()
        };
        let live = stats::live_stats(&snapshot, now_ns, STATS_WINDOW_NS);

        egui::TopBottomPanel::top("nums")
            .min_height(64.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    big_metric(ui, "fps", format!("{:.1}", live.fps));
                    ui.separator();
                    big_metric(ui, "p50", us_label(live.p50_us));
                    ui.separator();
                    big_metric(ui, "p99", us_label(live.p99_us));
                    ui.separator();
                    big_metric(ui, "max", us_label(live.max_us));
                    ui.separator();
                    big_metric(ui, "samples", live.samples.to_string());
                    ui.add_space(ui.available_width() - 120.0);
                    connection_pill(ui, connected);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let total = ui.available_height();
            let plot_h = (total / 3.0 - 8.0).max(80.0);

            // Plot 1 — fps over time.
            let fps_points = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                if b.is_empty() {
                    0.0
                } else {
                    b.len() as f64 * 1_000_000_000.0 / BIN_NS as f64
                }
            });
            Plot::new("fps_plot")
                .height(plot_h)
                .legend(Legend::default())
                .allow_zoom(false)
                .allow_drag(false)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(PlotPoints::from(fps_points)).name("fps"));
                });

            // Plot 2 — latency p50 / p99 / max over time.
            let p50_points = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                bin_percentile(b, 0.50)
            });
            let p99_points = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                bin_percentile(b, 0.99)
            });
            let max_points = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                b.iter().map(|r| r.latency_us as f64).fold(0.0, f64::max)
            });
            Plot::new("latency_plot")
                .height(plot_h)
                .legend(Legend::default())
                .allow_zoom(false)
                .allow_drag(false)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(PlotPoints::from(p50_points)).name("p50 µs"));
                    plot_ui.line(Line::new(PlotPoints::from(p99_points)).name("p99 µs"));
                    plot_ui.line(Line::new(PlotPoints::from(max_points)).name("max µs"));
                });

            // Plot 3 — frame-index gaps as a bar chart.
            let bars: Vec<Bar> = stats::frame_gaps(&snapshot)
                .into_iter()
                .map(|p| Bar::new(p[0], p[1]).width(1.0))
                .collect();
            Plot::new("gaps_plot")
                .height(plot_h)
                .legend(Legend::default())
                .allow_zoom(false)
                .allow_drag(false)
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(BarChart::new(bars).name("dropped"));
                });
        });
    }

    fn on_exit(&mut self) {
        self.state.stop.store(true, Ordering::Release);
    }
}

fn big_metric(ui: &mut egui::Ui, label: &str, value: String) {
    ui.vertical(|ui| {
        ui.label(egui::RichText::new(value).size(28.0).strong());
        ui.label(egui::RichText::new(label).size(11.0).weak());
    });
}

fn connection_pill(ui: &mut egui::Ui, connected: bool) {
    let (text, color) = if connected {
        ("● connected", egui::Color32::from_rgb(96, 200, 96))
    } else {
        ("● disconnected", egui::Color32::from_rgb(200, 96, 96))
    };
    ui.label(egui::RichText::new(text).color(color).size(13.0));
}

fn us_label(us: u64) -> String {
    if us >= 1_000 {
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
