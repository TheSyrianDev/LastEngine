//! Paged KV block allocator (vLLM-style free queue + ref counts). Pure Rust, no GPU.
//! Free list: uncached blocks go to the front (LIFO reuse), blocks with a prefix hash go to the back (FIFO eviction).
//! Ported by concept from vLLM kv_cache_utils.py / block_pool.py (see docs/vllm-port-guide.md).
use std::collections::VecDeque;

pub type BlockId = u32;

#[derive(Debug, PartialEq)]
pub enum FreeError { DoubleFree(BlockId) }

pub struct BlockAllocator {
    ref_cnt: Vec<u32>,
    cached: Vec<bool>,
    free: VecDeque<BlockId>,
    in_free: Vec<bool>,
}

impl BlockAllocator {
    pub fn new(total: usize) -> Self {
        Self { ref_cnt: vec![0; total], cached: vec![false; total],
               free: (0..total as BlockId).collect(), in_free: vec![true; total] }
    }
    pub fn total(&self) -> usize { self.ref_cnt.len() }
    pub fn num_free(&self) -> usize { self.free.len() }
    pub fn ref_count(&self, b: BlockId) -> u32 { self.ref_cnt[b as usize] }

    /// Take `n` blocks from the front of the free queue, ref_cnt=1. None if not enough free (all or nothing).
    pub fn allocate(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if self.free.len() < n { return None; }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let b = self.free.pop_front().unwrap();
            self.in_free[b as usize] = false;
            self.cached[b as usize] = false; // evicting any stale prefix-cache entry
            self.ref_cnt[b as usize] = 1;
            out.push(b);
        }
        Some(out)
    }

    /// Mark a block as holding a registered prefix hash (goes to the back of the queue when freed).
    pub fn mark_cached(&mut self, b: BlockId) { self.cached[b as usize] = true; }

    /// Share an existing block (prefix cache hit). Removes it from the free queue if it was there.
    pub fn touch(&mut self, b: BlockId) {
        let i = b as usize;
        if self.in_free[i] {
            self.free.retain(|&x| x != b);
            self.in_free[i] = false;
        }
        self.ref_cnt[i] += 1;
    }

    /// Drop one reference. At zero: uncached -> front (LIFO), cached -> back (FIFO).
    pub fn free_block(&mut self, b: BlockId) -> Result<(), FreeError> {
        let i = b as usize;
        if self.ref_cnt[i] == 0 { return Err(FreeError::DoubleFree(b)); }
        self.ref_cnt[i] -= 1;
        if self.ref_cnt[i] == 0 {
            self.in_free[i] = true;
            if self.cached[i] { self.free.push_back(b); } else { self.free.push_front(b); }
        }
        Ok(())
    }

    /// Free a request's blocks in reverse order, so the tail is evicted first (vLLM behaviour).
    pub fn free_seq(&mut self, blocks: &[BlockId]) -> Result<(), FreeError> {
        for &b in blocks.iter().rev() { self.free_block(b)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustion_is_all_or_nothing() {
        let mut a = BlockAllocator::new(4);
        assert_eq!(a.allocate(5), None);
        assert_eq!(a.num_free(), 4);
        let x = a.allocate(3).unwrap();
        assert_eq!(a.allocate(2), None);
        assert_eq!(a.num_free(), 1);
        a.free_seq(&x).unwrap();
        assert_eq!(a.num_free(), 4);
    }

    #[test]
    fn uncached_is_lifo_cached_is_fifo() {
        let mut a = BlockAllocator::new(4);
        let v = a.allocate(4).unwrap(); // 0,1,2,3
        a.mark_cached(v[0]);
        a.free_block(v[0]).unwrap(); // cached -> back
        a.free_block(v[1]).unwrap(); // uncached -> front
        a.free_block(v[2]).unwrap(); // uncached -> front (before v[1])
        assert_eq!(a.allocate(1).unwrap(), vec![v[2]]);
        assert_eq!(a.allocate(1).unwrap(), vec![v[1]]);
        assert_eq!(a.allocate(1).unwrap(), vec![v[0]]); // cached one at the back, evicted last
        assert_eq!(a.allocate(1), None); // v[3] was never freed: still held, so the pool is exhausted
    }

    #[test]
    fn ref_count_and_touch() {
        let mut a = BlockAllocator::new(2);
        let b = a.allocate(1).unwrap()[0];
        a.touch(b);
        assert_eq!(a.ref_count(b), 2);
        a.free_block(b).unwrap();
        assert_eq!(a.num_free(), 1);
        a.free_block(b).unwrap();
        assert_eq!(a.num_free(), 2);
    }

    #[test]
    fn touch_removes_from_free_queue() {
        let mut a = BlockAllocator::new(2);
        let b = a.allocate(1).unwrap()[0];
        a.mark_cached(b);
        a.free_block(b).unwrap();
        a.touch(b);
        assert_eq!(a.num_free(), 1);
        assert_eq!(a.ref_count(b), 1);
    }

    #[test]
    fn double_free_is_error() {
        let mut a = BlockAllocator::new(1);
        let b = a.allocate(1).unwrap()[0];
        a.free_block(b).unwrap();
        assert_eq!(a.free_block(b), Err(FreeError::DoubleFree(b)));
    }

    #[test]
    fn random_churn_conserves_blocks() {
        let mut a = BlockAllocator::new(32);
        let mut seed: u64 = 12345;
        let mut live: Vec<Vec<BlockId>> = vec![];
        for _ in 0..5000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if (seed >> 33) % 2 == 0 {
                if let Some(v) = a.allocate(((seed >> 40) % 5 + 1) as usize) { live.push(v); }
            } else if !live.is_empty() {
                let i = ((seed >> 20) as usize) % live.len();
                let v = live.swap_remove(i);
                a.free_seq(&v).unwrap();
            }
            let held: usize = live.iter().map(|v| v.len()).sum();
            assert_eq!(a.num_free() + held, 32);
        }
    }
}
