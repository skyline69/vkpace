//! Per-queue pool of timestamp query handles.
//!
//! Each call to [`TimestampPool::acquire`] returns a [`Handle`] holding two
//! pre-recorded command buffers: the first writes a `TOP_OF_PIPE` timestamp,
//! the second writes a `BOTTOM_OF_PIPE` timestamp. Wrap a queue submission
//! with `[start_buffer(), ...user cmds..., end_buffer()]` and the GPU stamps
//! frame start/end into a query pool that we read out asynchronously.
//!
//! Free-index tracking is a `Bitset` (constant-time alloc + free).  Each
//! [`Handle`] caches its two `VkCommandBuffer` pointers locally so the hot
//! path (`start_buffer`/`end_buffer`) never takes the chunks mutex.

use ash::vk;
use crossbeam_queue::SegQueue;
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::clock::DeviceClock;
use crate::dispatch::DeviceTable;

const CHUNK_SIZE: u32 = 512;
const _: () = assert!(CHUNK_SIZE.is_multiple_of(2));
const BITSET_WORDS: usize = (CHUNK_SIZE as usize).div_ceil(64);

/// O(1) free-pair allocator. Each pair of timestamp queries occupies
/// indices `{2k, 2k+1}`; we represent each pair by its even index `2k`, so a
/// `BITSET_WORDS`-word bitset covers all `CHUNK_SIZE/2` pairs.
pub struct Bitset {
    /// Bit `i` set ⇒ pair index `2*i` is free.
    words: [u64; BITSET_WORDS],
    hint: usize,
}

impl Bitset {
    pub fn full() -> Self {
        let total_pairs = (CHUNK_SIZE / 2) as usize;
        let mut words = [0u64; BITSET_WORDS];
        for i in 0..total_pairs {
            words[i / 64] |= 1u64 << (i % 64);
        }
        Self { words, hint: 0 }
    }

    pub fn acquire_pair(&mut self) -> Option<u32> {
        let start = self.hint.min(self.words.len().saturating_sub(1));
        for offset in 0..self.words.len() {
            let idx = (start + offset) % self.words.len();
            if self.words[idx] != 0 {
                let bit = self.words[idx].trailing_zeros() as usize;
                self.words[idx] &= !(1u64 << bit);
                self.hint = idx;
                let pair_index = idx * 64 + bit;
                return Some(pair_index as u32 * 2);
            }
        }
        None
    }

    pub fn release_pair(&mut self, query_index: u32) {
        let pair = (query_index / 2) as usize;
        self.words[pair / 64] |= 1u64 << (pair % 64);
        self.hint = self.hint.min(pair / 64);
    }

    pub fn any_free(&self) -> bool {
        self.words.iter().any(|w| *w != 0)
    }
}

struct QueryChunk {
    query_pool: vk::QueryPool,
    command_buffers: Vec<vk::CommandBuffer>,
    free: Bitset,
}

impl QueryChunk {
    /// # Safety
    /// `device` must be valid for the lifetime of the returned chunk; the
    /// caller is responsible for invoking [`QueryChunk::destroy`] before the
    /// device or command pool is destroyed.
    unsafe fn new(
        device: vk::Device,
        command_pool: vk::CommandPool,
        fns: &DeviceTable,
    ) -> Result<Self, vk::Result> {
        let qpci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(CHUNK_SIZE);
        let mut query_pool = vk::QueryPool::null();
        let r =
            unsafe { (fns.create_query_pool)(device, &qpci, std::ptr::null(), &mut query_pool) };
        if r != vk::Result::SUCCESS {
            return Err(r);
        }

        let cbai = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(CHUNK_SIZE);
        let mut command_buffers = vec![vk::CommandBuffer::null(); CHUNK_SIZE as usize];
        let r =
            unsafe { (fns.allocate_command_buffers)(device, &cbai, command_buffers.as_mut_ptr()) };
        if r != vk::Result::SUCCESS {
            unsafe { (fns.destroy_query_pool)(device, query_pool, std::ptr::null()) };
            return Err(r);
        }

        Ok(Self {
            query_pool,
            command_buffers,
            free: Bitset::full(),
        })
    }

    unsafe fn destroy(
        &mut self,
        device: vk::Device,
        command_pool: vk::CommandPool,
        fns: &DeviceTable,
    ) {
        if !self.command_buffers.is_empty() {
            unsafe {
                (fns.free_command_buffers)(
                    device,
                    command_pool,
                    self.command_buffers.len() as u32,
                    self.command_buffers.as_ptr(),
                )
            };
            self.command_buffers.clear();
        }
        if self.query_pool != vk::QueryPool::null() {
            unsafe { (fns.destroy_query_pool)(device, self.query_pool, std::ptr::null()) };
            self.query_pool = vk::QueryPool::null();
        }
    }
}

pub struct Handle {
    pool: Arc<PoolInner>,
    chunk_idx: usize,
    pub query_index: u32,
    pub start_cb: vk::CommandBuffer,
    pub end_cb: vk::CommandBuffer,
    pub was_submitted: AtomicBool,
}

impl Handle {
    #[inline]
    pub fn start_buffer(&self) -> vk::CommandBuffer {
        self.start_cb
    }

    #[inline]
    pub fn end_buffer(&self) -> vk::CommandBuffer {
        self.end_cb
    }

    pub fn await_end_ns(&self, clock: &DeviceClock) -> Option<u64> {
        self.pool
            .await_ticks(self.chunk_idx, self.query_index + 1)
            .map(|t| clock.ticks_to_host_ns(t))
    }

    /// Non-blocking probe — returns `true` iff the end-of-frame timestamp
    /// has been written by the GPU. Used by the swapchain monitor to bound
    /// its wait without ever calling into `GetQueryPoolResults(WAIT)`.
    pub fn has_end(&self) -> bool {
        self.pool.has_ticks(self.chunk_idx, self.query_index + 1)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // Lock-free push: hot path on user thread does no mutex acquisition.
        self.pool.expiring.push(ExpiringHandle {
            chunk_idx: self.chunk_idx,
            query_index: self.query_index,
            was_submitted: self.was_submitted.load(Ordering::Relaxed),
        });
        // Wake reaper. cv lock is uncontended — reaper sleeps on it.
        let _g = self.pool.wake.0.lock();
        self.pool.wake.1.notify_one();
    }
}

struct ExpiringHandle {
    chunk_idx: usize,
    query_index: u32,
    was_submitted: bool,
}

struct PoolInner {
    device: vk::Device,
    command_pool: vk::CommandPool,
    fns: Arc<DeviceTable>,
    chunks: Mutex<Vec<QueryChunk>>,
    expiring: SegQueue<ExpiringHandle>,
    /// Used only to park the reaper thread between drains — `expiring` is
    /// the actual queue, accessed lock-free.
    wake: (Mutex<()>, Condvar),
    stop: AtomicBool,
}

impl PoolInner {
    fn query_pool_for(&self, chunk_idx: usize) -> Option<vk::QueryPool> {
        let chunks = self.chunks.lock();
        chunks.get(chunk_idx).map(|c| c.query_pool)
    }

    fn await_ticks(&self, chunk_idx: usize, query: u32) -> Option<u64> {
        let query_pool = self.query_pool_for(chunk_idx)?;
        let mut result = [0u64; 2];
        let r = unsafe {
            (self.fns.get_query_pool_results)(
                self.device,
                query_pool,
                query,
                1,
                std::mem::size_of_val(&result),
                result.as_mut_ptr().cast(),
                std::mem::size_of_val(&result) as u64,
                vk::QueryResultFlags::TYPE_64
                    | vk::QueryResultFlags::WITH_AVAILABILITY
                    | vk::QueryResultFlags::WAIT,
            )
        };
        if r != vk::Result::SUCCESS || result[1] == 0 {
            return None;
        }
        Some(result[0])
    }

    /// Non-blocking availability check. Returns true if the GPU has finished
    /// writing the timestamp; false if not yet ready (or on any error).
    fn has_ticks(&self, chunk_idx: usize, query: u32) -> bool {
        let Some(query_pool) = self.query_pool_for(chunk_idx) else {
            return false;
        };
        let mut result = [0u64; 2];
        let r = unsafe {
            (self.fns.get_query_pool_results)(
                self.device,
                query_pool,
                query,
                1,
                std::mem::size_of_val(&result),
                result.as_mut_ptr().cast(),
                std::mem::size_of_val(&result) as u64,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        r == vk::Result::SUCCESS && result[1] != 0
    }
}

pub struct TimestampPool {
    inner: Arc<PoolInner>,
    reaper: Option<JoinHandle<()>>,
}

impl TimestampPool {
    pub fn new(device: vk::Device, command_pool: vk::CommandPool, fns: Arc<DeviceTable>) -> Self {
        let inner = Arc::new(PoolInner {
            device,
            command_pool,
            fns,
            chunks: Mutex::new(Vec::new()),
            expiring: SegQueue::new(),
            wake: (Mutex::new(()), Condvar::new()),
            stop: AtomicBool::new(false),
        });
        let reaper = {
            let inner = inner.clone();
            thread::Builder::new()
                .name("vkpace-reaper".into())
                .spawn(move || run_reaper(inner))
                .ok()
        };
        Self { inner, reaper }
    }

    pub fn acquire(self: &Arc<Self>) -> Option<Arc<Handle>> {
        let (chunk_idx, query_index, start_cb, end_cb) = self.find_or_alloc()?;
        unsafe { self.record(chunk_idx, query_index, start_cb, end_cb).ok()? };
        Some(Arc::new(Handle {
            pool: self.inner.clone(),
            chunk_idx,
            query_index,
            start_cb,
            end_cb,
            was_submitted: AtomicBool::new(false),
        }))
    }

    fn find_or_alloc(&self) -> Option<(usize, u32, vk::CommandBuffer, vk::CommandBuffer)> {
        let mut chunks = self.inner.chunks.lock();

        if let Some((idx, chunk)) = chunks
            .iter_mut()
            .enumerate()
            .find(|(_, c)| c.free.any_free())
        {
            let query = chunk.free.acquire_pair().unwrap();
            let start_cb = chunk.command_buffers[query as usize];
            let end_cb = chunk.command_buffers[query as usize + 1];
            return Some((idx, query, start_cb, end_cb));
        }

        let mut chunk = unsafe {
            QueryChunk::new(self.inner.device, self.inner.command_pool, &self.inner.fns).ok()?
        };
        let query = chunk.free.acquire_pair().unwrap();
        let start_cb = chunk.command_buffers[query as usize];
        let end_cb = chunk.command_buffers[query as usize + 1];
        let idx = chunks.len();
        chunks.push(chunk);
        Some((idx, query, start_cb, end_cb))
    }

    unsafe fn record(
        &self,
        chunk_idx: usize,
        query_index: u32,
        cb_start: vk::CommandBuffer,
        cb_end: vk::CommandBuffer,
    ) -> Result<(), vk::Result> {
        let fns = &self.inner.fns;
        let chunks = self.inner.chunks.lock();
        let query_pool = chunks[chunk_idx].query_pool;
        drop(chunks);

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let Some(write_ts) = fns.cmd_write_timestamp2 else {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        };
        let Some(reset_qp) = fns.reset_query_pool else {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
        };

        for (offset, stage, cb) in [
            (0u32, vk::PipelineStageFlags2::TOP_OF_PIPE, cb_start),
            (1u32, vk::PipelineStageFlags2::BOTTOM_OF_PIPE, cb_end),
        ] {
            let index = query_index + offset;
            unsafe {
                reset_qp(self.inner.device, query_pool, index, 1);
                let r = (fns.reset_command_buffer)(cb, vk::CommandBufferResetFlags::empty());
                if r != vk::Result::SUCCESS {
                    return Err(r);
                }
                let r = (fns.begin_command_buffer)(cb, &begin);
                if r != vk::Result::SUCCESS {
                    return Err(r);
                }
                write_ts(cb, stage, query_pool, index);
                let r = (fns.end_command_buffer)(cb);
                if r != vk::Result::SUCCESS {
                    return Err(r);
                }
            }
        }
        Ok(())
    }
}

impl Drop for TimestampPool {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        self.inner.wake.1.notify_all();
        if let Some(j) = self.reaper.take() {
            let _ = j.join();
        }
        // try_lock_for: if GPU is wedged + reaper is stuck blocking on
        // GetQueryPoolResults(WAIT), prefer leaking over hanging the host.
        match self.inner.chunks.try_lock_for(Duration::from_secs(2)) {
            Some(mut chunks) => {
                for chunk in chunks.iter_mut() {
                    unsafe {
                        chunk.destroy(self.inner.device, self.inner.command_pool, &self.inner.fns)
                    };
                }
                chunks.clear();
            }
            None => tracing::error!(
                "TimestampPool: chunks mutex stuck on drop — leaking query/CB resources"
            ),
        }
    }
}

fn run_reaper(inner: Arc<PoolInner>) {
    loop {
        // Drain everything currently pending without sleeping.
        while let Some(item) = inner.expiring.pop() {
            process_expiring(&inner, item);
        }
        // Nothing left — park on Condvar until a drop notifies or stop fires.
        if inner.stop.load(Ordering::Acquire) {
            // Drain anything raced in after we set stop.
            while let Some(item) = inner.expiring.pop() {
                process_expiring(&inner, item);
            }
            return;
        }
        let mut g = inner.wake.0.lock();
        // Re-check the queue under the lock to avoid a missed-wakeup race.
        if inner.expiring.is_empty() && !inner.stop.load(Ordering::Acquire) {
            inner.wake.1.wait(&mut g);
        }
    }
}

fn process_expiring(inner: &PoolInner, item: ExpiringHandle) {
    if item.was_submitted {
        let mut result = [0u64; 2];
        let chunks = inner.chunks.lock();
        let query_pool = chunks[item.chunk_idx].query_pool;
        drop(chunks);
        let _ = unsafe {
            (inner.fns.get_query_pool_results)(
                inner.device,
                query_pool,
                item.query_index + 1,
                1,
                std::mem::size_of_val(&result),
                result.as_mut_ptr().cast(),
                std::mem::size_of_val(&result) as u64,
                vk::QueryResultFlags::TYPE_64
                    | vk::QueryResultFlags::WITH_AVAILABILITY
                    | vk::QueryResultFlags::WAIT,
            )
        };
    }
    let mut chunks = inner.chunks.lock();
    if let Some(chunk) = chunks.get_mut(item.chunk_idx) {
        chunk.free.release_pair(item.query_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset_alloc_release_cycle() {
        let mut bs = Bitset::full();
        assert!(bs.any_free());
        let mut taken = Vec::new();
        while let Some(q) = bs.acquire_pair() {
            assert_eq!(q % 2, 0);
            taken.push(q);
        }
        assert_eq!(taken.len(), (CHUNK_SIZE / 2) as usize);
        assert!(!bs.any_free());
        for &q in &taken[..4] {
            bs.release_pair(q);
        }
        let mut second = Vec::new();
        while let Some(q) = bs.acquire_pair() {
            second.push(q);
        }
        let mut a = taken[..4].to_vec();
        let mut b = second.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn bitset_returns_even_indices_only() {
        let mut bs = Bitset::full();
        for _ in 0..10 {
            assert_eq!(bs.acquire_pair().unwrap() % 2, 0);
        }
    }
}
