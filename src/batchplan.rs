//! Batched-decode step planner for phase 2 step 4 (docs/concurrency-plan.md). Pure logic, no GPU.
//! Turns the set of active sequences into the (tokens, slots, positions) arrays that
//! `DenseGpu::decode_batch` takes, and decides which sequences finish after each sampled token.

/// One active sequence: its KV slot, the position of the next token to write, the token to feed next,
/// and generation bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seq {
    pub slot: usize,
    pub pos: usize,
    pub next_tok: u32,
    pub generated: usize,
    pub max_tokens: usize,
}

/// Inputs for one `decode_batch` call, one entry per row, in the order given by `rows`.
#[derive(Debug, PartialEq, Eq)]
pub struct StepInput { pub toks: Vec<u32>, pub slots: Vec<usize>, pub poss: Vec<usize>, pub rows: Vec<usize> }

#[derive(Debug, PartialEq, Eq)]
pub enum Finish { Length, Eos, Stop }

/// Build the batch for the next step from the active sequences (index = row id in `seqs`).
/// Sequences that reached `ctx_len` are skipped (the scheduler must retire them first).
pub fn plan_step(seqs: &[Seq], ctx_len: usize, max_rows: usize) -> StepInput {
    let mut out = StepInput { toks: vec![], slots: vec![], poss: vec![], rows: vec![] };
    for (i, s) in seqs.iter().enumerate() {
        if out.rows.len() == max_rows { break; }
        if s.pos >= ctx_len { continue; }
        out.toks.push(s.next_tok); out.slots.push(s.slot); out.poss.push(s.pos); out.rows.push(i);
    }
    out
}

/// After sampling `tok` for sequence `s`, advance it and report whether it finished.
/// Returns Some(reason) when the sequence should be retired (its slot freed).
pub fn advance(s: &mut Seq, tok: u32, eos: &[u32], ctx_len: usize) -> Option<Finish> {
    s.generated += 1;
    s.pos += 1;
    s.next_tok = tok;
    if eos.contains(&tok) { return Some(Finish::Eos); }
    if s.generated >= s.max_tokens { return Some(Finish::Length); }
    if s.pos >= ctx_len { return Some(Finish::Length); }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    fn seq(slot: usize, pos: usize, tok: u32, max: usize) -> Seq { Seq { slot, pos, next_tok: tok, generated: 0, max_tokens: max } }

    #[test]
    fn plan_collects_rows_in_order_and_respects_max_rows() {
        let seqs = vec![seq(0, 10, 5, 8), seq(1, 3, 6, 8), seq(2, 7, 7, 8)];
        let p = plan_step(&seqs, 4096, 2);
        assert_eq!(p.toks, vec![5, 6]);
        assert_eq!(p.slots, vec![0, 1]);
        assert_eq!(p.poss, vec![10, 3]);
        assert_eq!(p.rows, vec![0, 1]);
    }

    #[test]
    fn plan_skips_full_context() {
        let seqs = vec![seq(0, 4096, 1, 8), seq(1, 100, 2, 8)];
        let p = plan_step(&seqs, 4096, 8);
        assert_eq!(p.rows, vec![1]);
        assert_eq!(p.poss, vec![100]);
    }

    #[test]
    fn advance_stops_on_eos_and_length() {
        let mut s = seq(0, 10, 0, 2);
        assert_eq!(advance(&mut s, 42, &[2], 4096), None);
        assert_eq!((s.pos, s.generated, s.next_tok), (11, 1, 42));
        assert_eq!(advance(&mut s, 2, &[2], 4096), Some(Finish::Eos));
        let mut t = seq(1, 10, 0, 1);
        assert_eq!(advance(&mut t, 9, &[2], 4096), Some(Finish::Length));
    }

    #[test]
    fn advance_stops_at_context_end() {
        let mut s = seq(0, 4094, 0, 100);
        assert_eq!(advance(&mut s, 9, &[], 4096), None);
        assert_eq!(advance(&mut s, 9, &[], 4096), Some(Finish::Length));
    }
}
