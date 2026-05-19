//! egui HUD application.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use eframe::egui::{self, Color32, Frame, Label, Margin, RichText, Rounding, Stroke, Vec2b};

/// Drop-in for `ui.label(text)` that never lets the user select the
/// glyphs. We belt-and-suspender the global
/// `style.interaction.selectable_labels = false` because the style flag
/// has been seen to get re-enabled on the first paint depending on the
/// backend; per-widget `.selectable(false)` is the authoritative override.
fn ns_label(ui: &mut egui::Ui, text: impl Into<egui::WidgetText>) {
    ui.add(Label::new(text).selectable(false));
}
use egui_plot::{Legend, Line, Plot, PlotBounds, PlotPoints};

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
    /// Smoothed Y-axis upper bounds. egui_plot auto-fits per-frame, which
    /// makes the line appear to jump as bounds tighten/loosen with each
    /// new sample. We compute our own bounds, EMA-smooth them, then pin
    /// the plot to `[0, smoothed_max]`, so motion only comes from the
    /// data itself.
    fps_y_max: f64,
    frametime_y_max: f64,
    latency_y_max: f64,
    /// Cached aggregates. `update()` recomputes them only when
    /// `now_ns / RECOMPUTE_INTERVAL_NS` advances — during a window resize
    /// the compositor floods the event loop and forces `update()` to run
    /// at compositor rate (60-240 Hz on Wayland). Without caching, each
    /// of those calls re-clones the 4096-record snapshot and runs
    /// bin_records 5 times, which is the actual source of the laggy
    /// resize *and* the GPU-contention drop in game fps. With caching
    /// the heavy work stays bounded at ~10 Hz regardless of repaint rate.
    cache: PlotCache,
}

/// Recompute aggregates at most this often. 100 ms is a fifth of BIN_NS,
/// fast enough that the human eye can't see the staleness (eye persistence
/// is ~30 ms but for a plot this slow it's fine) and slow enough that the
/// cost is negligible during resize.
const RECOMPUTE_INTERVAL_NS: u64 = 100_000_000;

#[derive(Default)]
struct PlotCache {
    last_now_ns: u64,
    live: stats::LiveStats,
    drops: u64,
    samples: usize,
    fps_pts: Vec<[f64; 2]>,
    p50_pts: Vec<[f64; 2]>,
    p99_pts: Vec<[f64; 2]>,
    max_pts: Vec<[f64; 2]>,
    frametime_pts: Vec<[f64; 2]>,
}

impl HudApp {
    pub fn new(state: Arc<SharedState>) -> Self {
        Self {
            state,
            fps_y_max: 160.0,
            frametime_y_max: 33.0,
            latency_y_max: 30_000.0,
            cache: PlotCache::default(),
        }
    }
}

/// EMA-smooth `prev` toward `target`, clamped to `[floor, ceiling]`. Slow
/// alpha so the axis doesn't wobble even when the underlying max shifts.
fn smooth_bound(prev: f64, target: f64, floor: f64, ceiling: f64) -> f64 {
    let alpha = 0.05;
    let next = prev * (1.0 - alpha) + target * alpha;
    next.clamp(floor, ceiling)
}

impl eframe::App for HudApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Slow fallback heartbeat (4 Hz). Real refresh is reader-driven
        // via Context::request_repaint, so the UI is responsive when data
        // arrives without burning GPU cycles competing with the game when
        // it doesn't.
        ctx.request_repaint_after(Duration::from_millis(250));

        let connected = self.state.connected.load(Ordering::Acquire);
        let now_ns = self.state.latest_ts().unwrap_or(0);

        // Only recompute aggregates when the bin-quantized `now` advances.
        // Resize floods us with cheap renders that read these caches; the
        // heavy work stays clamped to ~10 Hz.
        let now_bucket = now_ns / RECOMPUTE_INTERVAL_NS;
        let last_bucket = self.cache.last_now_ns / RECOMPUTE_INTERVAL_NS;
        if now_ns > 0 && now_bucket != last_bucket {
            self.recompute_cache(now_ns);
        }

        // Smooth Y bounds — done outside the cache because the EMA needs
        // to step every frame for the visual to feel responsive on first
        // launch. The arithmetic is trivial (three f64 ops) so it's
        // cheap even at compositor rate.
        let live = &self.cache.live;
        let fps_target = (live.fps * 1.25).max(160.0);
        let frametime_target = (live.fps.max(30.0).recip() * 1000.0 * 2.0).max(20.0);
        let latency_target = ((live.max_us as f64) * 1.4).max(8_000.0);
        self.fps_y_max = smooth_bound(self.fps_y_max, fps_target, 60.0, 600.0);
        self.frametime_y_max = smooth_bound(self.frametime_y_max, frametime_target, 16.0, 200.0);
        self.latency_y_max = smooth_bound(self.latency_y_max, latency_target, 8_000.0, 200_000.0);

        top_strip(ctx, connected, &self.cache.live, self.cache.drops);
        plots_cached(
            ctx,
            &self.cache,
            self.fps_y_max,
            self.frametime_y_max,
            self.latency_y_max,
        );
    }

    fn on_exit(&mut self) {
        self.state.stop.store(true, Ordering::Release);
    }
}

impl HudApp {
    /// Recompute snapshot + stats + plot series. Runs at most every
    /// `RECOMPUTE_INTERVAL_NS`; everything else reads from cache.
    fn recompute_cache(&mut self, now_ns: u64) {
        let snapshot = self
            .state
            .snapshot_since(now_ns.saturating_sub(WINDOW_NS));
        let live = stats::live_stats(&snapshot, now_ns, STATS_WINDOW_NS);
        let drops: u64 = stats::frame_gaps(&snapshot)
            .iter()
            .map(|p| p[1] as u64)
            .sum();

        let fps_pts: Vec<[f64; 2]> = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
            if b.is_current && b.width_ns < 100_000_000 {
                return f64::NAN;
            }
            if !b.is_current && b.records.len() < 4 {
                return f64::NAN;
            }
            b.records.len() as f64 * 1_000_000_000.0 / b.width_ns as f64
        })
        .into_iter()
        .filter(|p| p[1].is_finite())
        .collect();

        let frametime_pts: Vec<[f64; 2]> =
            stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                if b.is_current && b.width_ns < 100_000_000 {
                    return f64::NAN;
                }
                if !b.is_current && b.records.len() < 4 {
                    return f64::NAN;
                }
                let secs = b.width_ns as f64 / 1e9;
                secs * 1000.0 / b.records.len() as f64
            })
            .into_iter()
            .filter(|p| p[1].is_finite())
            .collect();

        let (p50_pts, p99_pts, max_pts) = if live.samples == 0 {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            let p50 = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                bin_percentile(&b.records, 0.50)
            });
            let p99 = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                bin_percentile(&b.records, 0.99)
            });
            let mx = stats::bin_records(&snapshot, now_ns, WINDOW_NS, BIN_NS, |b| {
                b.records
                    .iter()
                    .map(|r| r.latency_us as f64)
                    .fold(0.0, f64::max)
            });
            (p50, p99, mx)
        };

        self.cache = PlotCache {
            last_now_ns: now_ns,
            samples: live.samples,
            live,
            drops,
            fps_pts,
            p50_pts,
            p99_pts,
            max_pts,
            frametime_pts,
        };
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
                ns_label(
                    ui,
                    RichText::new(value)
                        .size(24.0)
                        .color(value_color)
                        .strong(),
                );
                ns_label(
                    ui,
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
                ns_label(ui, RichText::new(text).color(color).size(12.0).strong());
            });
        });
}

// ── Plots ─────────────────────────────────────────────────────────────

fn plots_cached(
    ctx: &egui::Context,
    cache: &PlotCache,
    fps_y_max: f64,
    frametime_y_max: f64,
    latency_y_max: f64,
) {
    egui::CentralPanel::default()
        .frame(
            Frame::none()
                .fill(theme::BG_WINDOW)
                .inner_margin(Margin::same(10.0)),
        )
        .show(ctx, |ui| {
            let total = ui.available_height();
            // Per-card chrome bumped to 52 — the previous 38 covered the
            // Frame margins + title + spacer but didn't leave any room for
            // the plot's bottom X-axis labels, so the last card's labels
            // got clipped by the window edge. The plot itself now reserves
            // an extra ~12 px at the bottom via its inner ui margin
            // (`PLOT_BODY_MARGIN`).
            const CARD_CHROME: f32 = 52.0;
            let plot_h = ((total - 3.0 * CARD_CHROME) / 3.0).max(80.0);

            plot_card(ui, "fps", plot_h, |inner| {
                Plot::new("fps_plot")
                    .height(inner.available_height())
                    .legend(Legend::default().background_alpha(0.0))
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .auto_bounds(Vec2b::FALSE)
                    .show_x(false)
                    .show_y(true)
                    .show(inner, |plot_ui| {
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [-60.0, 0.0],
                            [0.0, fps_y_max],
                        ));
                        plot_ui.set_auto_bounds(Vec2b::FALSE);
                        plot_ui.line(
                            Line::new(PlotPoints::from(cache.fps_pts.clone()))
                                .color(theme::LINE_FPS)
                                .width(2.0)
                                .name("fps"),
                        );
                    });
            });

            plot_card(ui, "click-to-photon (µs)", plot_h, |inner| {
                if cache.samples == 0 {
                    no_latency_hint(inner);
                    return;
                }
                Plot::new("latency_plot")
                    .height(inner.available_height())
                    .legend(Legend::default().background_alpha(0.0))
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .auto_bounds(Vec2b::FALSE)
                    .show(inner, |plot_ui| {
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [-60.0, 0.0],
                            [0.0, latency_y_max],
                        ));
                        plot_ui.set_auto_bounds(Vec2b::FALSE);
                        plot_ui.line(
                            Line::new(PlotPoints::from(cache.p50_pts.clone()))
                                .color(theme::LINE_P50)
                                .width(1.5)
                                .name("p50"),
                        );
                        plot_ui.line(
                            Line::new(PlotPoints::from(cache.p99_pts.clone()))
                                .color(theme::LINE_P99)
                                .width(1.5)
                                .name("p99"),
                        );
                        plot_ui.line(
                            Line::new(PlotPoints::from(cache.max_pts.clone()))
                                .color(theme::LINE_MAX)
                                .width(1.2)
                                .name("max"),
                        );
                    });
            });

            plot_card(ui, "frame-time (ms)", plot_h, |inner| {
                Plot::new("frametime_plot")
                    .height(inner.available_height())
                    .legend(Legend::default().background_alpha(0.0))
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .auto_bounds(Vec2b::FALSE)
                    .show(inner, |plot_ui| {
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [-60.0, 0.0],
                            [0.0, frametime_y_max],
                        ));
                        plot_ui.set_auto_bounds(Vec2b::FALSE);
                        plot_ui.line(
                            Line::new(PlotPoints::from(cache.frametime_pts.clone()))
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
        // Extra bottom padding so the plot's own X-axis labels never crash
        // into the card's bottom edge; extra left padding so the leftmost
        // Y-axis number ("0", "-60") doesn't collide with the X "-60".
        .inner_margin(Margin {
            left: 14.0,
            right: 10.0,
            top: 8.0,
            bottom: 14.0,
        })
        .show(ui, |ui| {
            ui.set_height(h);
            ns_label(
                ui,
                RichText::new(title)
                    .color(theme::FG_WEAK)
                    .size(11.0)
                    .strong(),
            );
            ui.add_space(2.0);
            body(ui);
        });
    ui.add_space(6.0);
}

fn no_latency_hint(ui: &mut egui::Ui) {
    let h = ui.available_height();
    ui.vertical_centered(|ui| {
        ui.add_space((h / 2.0 - 10.0).max(0.0));
        ns_label(
            ui,
            RichText::new("no latency samples")
                .color(theme::FG_NORMAL)
                .size(13.0)
                .strong(),
        );
        // No italic — we only ship the upright JetBrains Mono cuts, and
        // egui synthesises italic via shear which looks bad on mono.
        ns_label(
            ui,
            RichText::new("app is not calling vkSetLatencyMarkerNV (Reflex)")
                .color(theme::FG_WEAK)
                .size(11.0),
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
