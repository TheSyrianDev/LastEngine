//! Per-sequence block table: logical token position -> physical KV slot.
//! slot(pos) = block[pos / BS] * BS + pos % BS. Pure Rust, no GPU.
//! Ported by concept from vLLM worker/gpu/block_table.py (see docs/vllm-port-guide.md).
use crate::blockalloc::{BlockAllocator, BlockId};

pub const BS: usize = 16;

pub struct BlockTable { pub blocks: Vec<BlockId> }

impl BlockTable {
    pub fn new() -> Self { Self { blocks: Vec::new() } }
    /// Grow to hold `n_tokens` positions, allocating blocks on demand. False if the pool is exhausted (nothing taken).
    pub fn ensure(&mut self, a: &mut BlockAllocator, n_tokens: usize) -> bool {
        let need = n_tokens.div_ceil(BS);
        if need <= self.blocks.len() { return true; }
        match a.allocate(need - self.blocks.len()) {
            Some(mut v) => { self.blocks.append(&mut v); true }
            None => false,
        }
    }
    pub fn slot(&self, pos: usize) -> usize { self.blocks[pos / BS] as usize * BS + pos % BS }
    pub fn release(&mut self, a: &mut BlockAllocator) {
        let _ = a.free_seq(&self.blocks);
        self.blocks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shuffled physical blocks: pool hands out ids in a scrambled order; logical positions must still map 1:1 onto distinct slots.
    #[test]
    fn shuffled_blocks_are_a_bijection() {
        let mut a = BlockAllocator::new(8);
        let _ = a.allocate(3); // burn some ids so the next seq gets non-contiguous ones
        let mut t = BlockTable::new();
        assert!(t.ensure(&mut a, 40)); // 3 blocks
        assert_eq!(t.blocks.len(), 3);
        let mut seen = std::collections::HashSet::new();
        for p in 0..40 { assert!(seen.insert(t.slot(p))); }
        assert_eq!(seen.len(), 40);
    }

    #[test]
    fn ensure_is_all_or_nothing_and_grows() {
        let mut a = BlockAllocator::new(2);
        let mut t = BlockTable::new();
        assert!(t.ensure(&mut a, 16));
        assert_eq!(t.blocks.len(), 1);
        assert!(!t.ensure(&mut a, 16 * 3));
        assert_eq!(t.blocks.len(), 1);
        assert!(t.ensure(&mut a, 32));
        assert_eq!(t.blocks.len(), 2);
    }

    #[test]
    fn release_returns_all_blocks() {
        let mut a = BlockAllocator::new(4);
        let mut t = BlockTable::new();
        assert!(t.ensure(&mut a, 60));
        t.release(&mut a);
        assert_eq!(a.num_free(), 4);
    }
}
