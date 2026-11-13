//! Slot scheduler for phase 2 continuous batching (docs/concurrency-plan.md step 4).
//! Pure bookkeeping, no GPU: maps waiting requests onto free KV slots in FIFO order.

use std::collections::VecDeque;

/// Admission result for one request.
#[derive(Debug, PartialEq, Eq)]
pub enum Admit { Slot(usize), Queued(usize) }

#[derive(Debug)]
pub struct Scheduler {
    slots: Vec<Option<u64>>,      // slot -> request id occupying it
    waiting: VecDeque<u64>,       // FIFO of request ids without a slot
}

impl Scheduler {
    pub fn new(n_slots: usize) -> Self {
        assert!(n_slots > 0, "scheduler needs at least one slot");
        Scheduler { slots: vec![None; n_slots], waiting: VecDeque::new() }
    }

    /// Give request `id` a free slot, or queue it at position `waiting.len()` (0-based).
    pub fn admit(&mut self, id: u64) -> Admit {
        if let Some(s) = self.slots.iter().position(|x| x.is_none()) {
            self.slots[s] = Some(id);
            Admit::Slot(s)
        } else {
            self.waiting.push_back(id);
            Admit::Queued(self.waiting.len() - 1)
        }
    }

    /// Free slot `s` (request finished or cancelled). Returns the id promoted into it, if any.
    pub fn retire(&mut self, s: usize) -> Option<u64> {
        assert!(s < self.slots.len(), "slot {} out of range", s);
        let promoted = self.waiting.pop_front();
        self.slots[s] = promoted;
        promoted
    }

    /// Remove a queued request that was cancelled before getting a slot. Returns true if found.
    pub fn cancel_waiting(&mut self, id: u64) -> bool {
        if let Some(i) = self.waiting.iter().position(|&x| x == id) { self.waiting.remove(i); true } else { false }
    }

    pub fn active(&self) -> Vec<(usize, u64)> {
        self.slots.iter().enumerate().filter_map(|(s, x)| x.map(|id| (s, id))).collect()
    }

    pub fn waiting_len(&self) -> usize { self.waiting.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_free_slots_then_queues_fifo() {
        let mut s = Scheduler::new(2);
        assert_eq!(s.admit(1), Admit::Slot(0));
        assert_eq!(s.admit(2), Admit::Slot(1));
        assert_eq!(s.admit(3), Admit::Queued(0));
        assert_eq!(s.admit(4), Admit::Queued(1));
        assert_eq!(s.waiting_len(), 2);
    }

    #[test]
    fn retire_promotes_oldest_waiting_into_freed_slot() {
        let mut s = Scheduler::new(1);
        s.admit(10);
        s.admit(11);
        s.admit(12);
        assert_eq!(s.retire(0), Some(11));
        assert_eq!(s.active(), vec![(0, 11)]);
        assert_eq!(s.retire(0), Some(12));
        assert_eq!(s.retire(0), None);
        assert!(s.active().is_empty());
    }

    #[test]
    fn cancel_removes_only_waiting_request() {
        let mut s = Scheduler::new(1);
        s.admit(1);
        s.admit(2);
        s.admit(3);
        assert!(s.cancel_waiting(2));
        assert!(!s.cancel_waiting(1));
        assert_eq!(s.retire(0), Some(3));
    }
}
