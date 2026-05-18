//! GPU↔CPU timestamp clock backed by `VK_KHR_calibrated_timestamps`.
//!
//! A background worker re-anchors the host/device timestamp pair every
//! [`CALIBRATION_PERIOD`]; reads use whatever the latest anchor is. The
//! worker wakes on a `Condvar`, so shutdown is immediate when the parent
//! `DeviceClock` is dropped.

use ash::vk;
use parking_lot::{Condvar, Mutex, RwLock};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::dispatch::DeviceTable;

const CALIBRATION_PERIOD: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Default)]
struct Anchor {
    host_ns: u64,
    device_ticks: u64,
}

struct Inner {
    device: vk::Device,
    fns: Arc<DeviceTable>,
    period_ns: f64,
    anchor: RwLock<Anchor>,
    stop: AtomicBool,
    wake: (Mutex<()>, Condvar),
}

impl Inner {
    /// # Safety
    /// Caller guarantees that the device handle and function pointers are still valid.
    unsafe fn refresh_anchor(&self) -> Result<(), vk::Result> {
        let Some(get_ts) = self.fns.get_calibrated_timestamps_khr else {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        };
        let infos = [
            vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE),
            vk::CalibratedTimestampInfoKHR::default()
                .time_domain(vk::TimeDomainKHR::CLOCK_MONOTONIC),
        ];
        let mut values = [0u64; 2];
        let mut max_deviation = 0u64;
        let r = unsafe {
            get_ts(
                self.device,
                infos.len() as u32,
                infos.as_ptr(),
                values.as_mut_ptr(),
                &mut max_deviation,
            )
        };
        if r != vk::Result::SUCCESS {
            return Err(r);
        }
        tracing::trace!(
            host_ns = values[1],
            device_ticks = values[0],
            max_deviation,
            "calibrated-timestamps refresh"
        );
        *self.anchor.write() = Anchor {
            device_ticks: values[0],
            host_ns: values[1],
        };
        Ok(())
    }
}

pub struct DeviceClock {
    inner: Arc<Inner>,
    worker: Option<JoinHandle<()>>,
}

impl DeviceClock {
    /// Returns `None` if `VK_KHR_calibrated_timestamps` isn't loaded.
    pub fn new(device: vk::Device, fns: Arc<DeviceTable>, timestamp_period: f32) -> Option<Self> {
        fns.get_calibrated_timestamps_khr?;
        let inner = Arc::new(Inner {
            device,
            fns,
            period_ns: timestamp_period as f64,
            anchor: RwLock::new(Anchor::default()),
            stop: AtomicBool::new(false),
            wake: (Mutex::new(()), Condvar::new()),
        });
        // Best-effort initial anchor.
        let _ = unsafe { inner.refresh_anchor() };

        let worker = {
            let inner = inner.clone();
            thread::Builder::new()
                .name("vkpace-clock".into())
                .spawn(move || calibration_loop(inner))
                .ok()
        };

        Some(Self { inner, worker })
    }

    pub fn now() -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
        (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
    }

    /// Convert raw GPU ticks into host monotonic nanoseconds.
    pub fn ticks_to_host_ns(&self, ticks: u64) -> u64 {
        let anchor = *self.inner.anchor.read();
        let diff = ticks as i128 - anchor.device_ticks as i128;
        let diff_ns = (diff as f64 * self.inner.period_ns).round() as i128;
        (anchor.host_ns as i128 + diff_ns).max(0) as u64
    }
}

impl Drop for DeviceClock {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        self.inner.wake.1.notify_all();
        if let Some(j) = self.worker.take() {
            let _ = j.join();
        }
    }
}

fn calibration_loop(inner: Arc<Inner>) {
    while !inner.stop.load(Ordering::Acquire) {
        let mut g = inner.wake.0.lock();
        let _ = inner.wake.1.wait_for(&mut g, CALIBRATION_PERIOD);
        drop(g);
        if inner.stop.load(Ordering::Acquire) {
            break;
        }
        if let Err(e) = unsafe { inner.refresh_anchor() } {
            tracing::warn!(?e, "calibrated-timestamps refresh failed");
        }
    }
}
