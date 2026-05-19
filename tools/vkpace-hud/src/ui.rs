//! egui HUD application.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::state::SharedState;
use crate::stats;

/// Time window plotted on the X axis (60 s rolling history).
const WINDOW_NS: u64 = 60_000_000_000;
/// Plot bin width. 100 ms bins at 144 fps held only ~14 samples each, so
/// boundary alignment alone produced ±10% reading swings (visible as the
/// noisy zig-zag in the first live capture). 500 ms gives ~70 samples per
/// bin at 144 fps — stable to within ~1 sample of ground truth — and 120
/// points across the 60 s window, which is plenty of horizontal
/// resolution at typical HUD sizes.
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

        // Drop count over the whole 60 s plotted window (cheap; ≤ 4096 records).
        let drops: u64 = stats::frame_gaps(&snapshot)
            .iter()
            .map(|p| p[1] as u64)
            .sum();

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
                    ui.separator();
                    big_metric(ui, "drops/60s", drops.to_string());
                    ui.add_space(ui.available_width() - 130.0);
                    connection_pill(ui, connected);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let total = ui.available_height();
            let plot_h = (total / 3.0 - 8.0).max(80.0);

            // Plot 1 — fps over time. `bin_records` already drops empties,
            // so this never sees a zero-length bucket.
            let fps_points = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                b.len() as f64 * 1_000_000_000.0 / BIN_NS as f64
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
            // Empty when the app doesn't call vkSetLatencyMarkerNV (vkcube,
            // most demo apps). Replace with a hint so the user knows it's
            // not broken — only Reflex-aware apps emit input markers.
            ui.vertical(|ui| {
                ui.set_height(plot_h);
                if live.samples == 0 {
                    ui.add_space(plot_h / 2.0 - 8.0);
                    ui.label(
                        egui::RichText::new(
                            "no latency samples — app is not emitting Reflex \
                             markers (vkSetLatencyMarkerNV). Frame-time plot \
                             below works regardless.",
                        )
                        .italics()
                        .weak(),
                    );
                } else {
                    let p50_points =
                        stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                            bin_percentile(b, 0.50)
                        });
                    let p99_points =
                        stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                            bin_percentile(b, 0.99)
                        });
                    let max_points =
                        stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
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
                }
            });

            // Plot 3 — frame-time (ms) from present-to-present ts_ns deltas.
            // Works for every app, including ones without Reflex markers.
            // We report the *median* delta per bin, not the mean: one
            // scheduler hiccup (a 30 ms frame mixed in with seventy 7 ms
            // frames) drags the mean visibly upward, but leaves the median
            // unchanged. The median tracks the steady-state cadence — what
            // a user actually wants to see for "is my pacing stable".
            let frametime_points =
                stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |bucket| {
                    if bucket.len() < 2 {
                        return 0.0;
                    }
                    let mut deltas: Vec<u64> = Vec::with_capacity(bucket.len() - 1);
                    let mut prev = bucket[0].ts_ns;
                    for r in &bucket[1..] {
                        if r.ts_ns > prev {
                            deltas.push(r.ts_ns - prev);
                        }
                        prev = r.ts_ns;
                    }
                    if deltas.is_empty() {
                        return 0.0;
                    }
                    deltas.sort_unstable();
                    let mid = deltas[deltas.len() / 2];
                    mid as f64 / 1_000_000.0
                });
            Plot::new("frametime_plot")
                .height(plot_h)
                .legend(Legend::default())
                .allow_zoom(false)
                .allow_drag(false)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(PlotPoints::from(frametime_points)).name("frame-time ms"),
                    );
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
        ("connected", egui::Color32::from_rgb(96, 200, 96))
    } else {
        ("disconnected", egui::Color32::from_rgb(200, 96, 96))
    };
    ui.horizontal(|ui| {
        // Paint a solid dot via the painter — default eframe fonts don't
        // ship `●` (U+25CF), so a glyph would render as the .notdef box.
        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 5.0, color);
        ui.label(egui::RichText::new(text).color(color).size(13.0));
    });
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
