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

/// Drive one batched-decode loop over a fixed set of requests, with a caller-supplied step function.
/// `step(plan) -> sampled tokens` (one per row of the plan, same order). Finished sequences are
/// reported through `done(id, reason)`. Returns the number of steps taken. Pure logic: the GPU
/// backend supplies `step` as a `decode_batch` + sampler call.
pub fn run_loop<F, D>(seqs: &mut Vec<(u64, Seq)>, eos: &[u32], ctx_len: usize, max_rows: usize, mut step: F, mut done: D) -> usize
where F: FnMut(&StepInput) -> Vec<u32>, D: FnMut(u64, Finish) {
    let mut steps = 0;
    while !seqs.is_empty() {
        let view: Vec<Seq> = seqs.iter().map(|(_, s)| s.clone()).collect();
        let plan = plan_step(&view, ctx_len, max_rows);
        if plan.rows.is_empty() { break; }
        let toks = step(&plan);
        assert_eq!(toks.len(), plan.rows.len(), "step must return one token per row");
        steps += 1;
        let mut retire = vec![];
        for (k, &row) in plan.rows.iter().enumerate() {
            if let Some(r) = advance(&mut seqs[row].1, toks[k], eos, ctx_len) { retire.push((row, r)); }
        }
        // Remove from the back so indices stay valid.
        retire.sort_by(|a, b| b.0.cmp(&a.0));
        for (row, r) in retire { let (id, _) = seqs.remove(row); done(id, r); }
    }
    steps
}

#[cfg(test)]
mod loop_tests {
    use super::*;
    fn seq(slot: usize, pos: usize, tok: u32, max: usize) -> Seq { Seq { slot, pos, next_tok: tok, generated: 0, max_tokens: max } }

    #[test]
    fn loop_runs_all_sequences_to_their_limits() {
        let mut seqs = vec![(1u64, seq(0, 10, 5, 3)), (2u64, seq(1, 20, 6, 5))];
        let mut finished = vec![];
        let steps = run_loop(&mut seqs, &[], 4096, 8, |p| p.toks.iter().map(|t| t + 1).collect(), |id, r| finished.push((id, r)));
        assert_eq!(steps, 5);
        assert_eq!(finished, vec![(1, Finish::Length), (2, Finish::Length)]);
        assert!(seqs.is_empty());
    }

    #[test]
    fn loop_retires_on_eos_and_keeps_others() {
        let mut seqs = vec![(1u64, seq(0, 10, 0, 100)), (2u64, seq(1, 10, 0, 100))];
        let mut finished = vec![];
        // Sequence 1 emits EOS (2) immediately; sequence 2 keeps going until its length limit.
        let mut calls = 0;
        run_loop(&mut seqs, &[2], 4096, 8, |p| { calls += 1; p.slots.iter().map(|&sl| if sl == 0 { 2 } else { 7 }).collect() }, |id, r| finished.push((id, r)));
        // Seq 1 finishes at the first step; seq 2 continues alone for its full 100 tokens.
        assert_eq!(finished, vec![(1, Finish::Eos), (2, Finish::Length)]);
        assert_eq!(calls, 100);
        assert!(seqs.is_empty());
    }
}

/// Greedy (argmax) token per row from `decode_batch` logits. Ties go to the lowest index, like `local::sample` at temperature 0.
/// Sampling with temperature > 0 is not in the batched path yet; callers must route such requests to the single-sequence path.
pub fn greedy_rows(logits: &[Vec<f32>]) -> Vec<u32> {
    logits.iter().map(|row| {
        let mut best = 0usize;
        for (i, &v) in row.iter().enumerate() { if v > row[best] { best = i; } }
        best as u32
    }).collect()
}

#[cfg(test)]
mod greedy_tests {
    use super::*;
    #[test]
    fn argmax_per_row_lowest_index_on_tie() {
        let rows = vec![vec![0.1, 3.0, 2.0], vec![5.0, 5.0, 1.0], vec![-1.0, -0.5, -2.0]];
        assert_eq!(greedy_rows(&rows), vec![1, 0, 1]);
    }
}

/// Interim cap on rows per batched step. Rows 7 and 8 diverge from solo decode (known bug, see docs/concurrency-plan.md); keep at 6 until root-caused.
pub const BATCH_ROW_CAP: usize = 6;

/// Routing rule for the batched path: only deterministic plain-text requests may use it.
/// Sampled requests (temperature > 0, or any penalty/top-k/top-p/min-p/seed), tool requests and
/// thinking-enabled requests stay on the single-sequence `generate` path until the batched loop
/// supports them. Mirrors the constraints documented in docs/concurrency-plan.md.
pub fn batched_eligible(temperature: f32, top_p: f32, top_k: usize, min_p: f32, presence_penalty: f32, repeat_penalty: f32, seed: Option<u64>, has_tools: bool, thinking: bool) -> bool {
    temperature <= 0.0 && (top_p == 0.0 || top_p == 1.0) && top_k == 0 && min_p == 0.0
        && presence_penalty == 0.0 && (repeat_penalty == 0.0 || repeat_penalty == 1.0) && seed.is_none()
        && !has_tools && !thinking
}

#[cfg(test)]
mod route_tests {
    use super::*;
    #[test]
    fn only_plain_greedy_is_batched() {
        assert!(batched_eligible(0.0, 1.0, 0, 0.0, 0.0, 1.0, None, false, false));
        assert!(batched_eligible(0.0, 0.0, 0, 0.0, 0.0, 0.0, None, false, false));
        assert!(!batched_eligible(0.7, 1.0, 0, 0.0, 0.0, 1.0, None, false, false));
        assert!(!batched_eligible(0.0, 1.0, 40, 0.0, 0.0, 1.0, None, false, false));
        assert!(!batched_eligible(0.0, 1.0, 0, 0.0, 0.0, 1.0, Some(7), false, false));
        assert!(!batched_eligible(0.0, 1.0, 0, 0.0, 0.0, 1.0, None, true, false));
        assert!(!batched_eligible(0.0, 1.0, 0, 0.0, 0.0, 1.0, None, false, true));
    }
}

/// One step of the batched loop: plan, call `step`, advance every planned row, and return the ids of
/// sequences that finished (with reason) and removed from `seqs`. Returns None when nothing was planned.
/// Used by the server worker, which admits new requests between steps (see `run_loop` for a fixed set).
pub fn step_once<F>(seqs: &mut Vec<(u64, Seq)>, eos: &[u32], ctx_len: usize, max_rows: usize, step: &mut F) -> Option<Vec<(u64, Finish)>>
where F: FnMut(&StepInput) -> Vec<u32> {
    let view: Vec<Seq> = seqs.iter().map(|(_, s)| s.clone()).collect();
    let plan = plan_step(&view, ctx_len, max_rows);
    if plan.rows.is_empty() { return None; }
    let toks = step(&plan);
    assert_eq!(toks.len(), plan.rows.len(), "step must return one token per row");
    let mut fin = vec![];
    let mut rm = vec![];
    for (k, &row) in plan.rows.iter().enumerate() {
        if let Some(r) = advance(&mut seqs[row].1, toks[k], eos, ctx_len) { rm.push((row, r)); }
    }
    rm.sort_by(|a, b| b.0.cmp(&a.0));
    for (row, r) in rm { let (id, _) = seqs.remove(row); fin.push((id, r)); }
    Some(fin)
}

#[cfg(test)]
mod step_tests {
    use super::*;
    fn seq(slot: usize, pos: usize, tok: u32, max: usize) -> Seq { Seq { slot, pos, next_tok: tok, generated: 0, max_tokens: max } }

    #[test]
    fn admission_between_steps_joins_the_batch() {
        let mut seqs = vec![(1u64, seq(0, 10, 5, 4))];
        let mut step = |p: &StepInput| p.toks.iter().map(|t| t + 1).collect::<Vec<u32>>();
        assert_eq!(step_once(&mut seqs, &[], 4096, 8, &mut step), Some(vec![]));
        // A new request is admitted into slot 1 mid-run.
        seqs.push((2, seq(1, 20, 9, 2)));
        assert_eq!(step_once(&mut seqs, &[], 4096, 8, &mut step), Some(vec![]));
        let f = step_once(&mut seqs, &[], 4096, 8, &mut step).unwrap();
        assert_eq!(f, vec![(2, Finish::Length)]);
        assert_eq!(seqs.len(), 1);
        let f = step_once(&mut seqs, &[], 4096, 8, &mut step).unwrap();
        assert_eq!(f, vec![(1, Finish::Length)]);
        assert!(step_once(&mut seqs, &[], 4096, 8, &mut step).is_none());
    }
}

/// vLLM-style token budget step (pure): decodes first (1 token each, running order), then admits waiting prefills
/// while budget remains. `allocate` returns false when the KV pool cannot hold a request (it is skipped, not dropped).
pub fn budget_step(running: &[usize], waiting: &[(usize, usize)], budget: usize, mut allocate: impl FnMut(usize, usize) -> bool) -> (Vec<usize>, Vec<usize>) {
    let mut used = 0usize;
    let mut decodes = vec![];
    for &id in running { if used + 1 <= budget { decodes.push(id); used += 1; } }
    let mut admits = vec![];
    for &(id, ntok) in waiting {
        if used >= budget { break; }
        let take = ntok.min(budget - used);
        if !allocate(id, take) { break; } // FIFO fairness: stop at the first request that does not fit
        admits.push(id); used += take;
    }
    (decodes, admits)
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn budget_never_exceeded_and_decodes_first() {
        let (d, a) = budget_step(&[1, 2, 3], &[(10, 100), (11, 5)], 8, |_, _| true);
        assert_eq!(d, vec![1, 2, 3]);
        assert_eq!(a, vec![10]); // 5 budget left after decodes, 10 takes all 5, 11 gets none
    }
    #[test]
    fn fifo_stops_at_first_unallocatable() {
        let (_, a) = budget_step(&[], &[(1, 4), (2, 4), (3, 4)], 100, |id, _| id != 2);
        assert_eq!(a, vec![1]); // 2 fails, 3 must not jump the queue
    }
}

/// Chunked prefill (pure): split a prompt of `len` tokens into chunks of at most `chunk` tokens, each chunk
/// starting at the previous end. Chunks cover [0, len) exactly once, in order.
pub fn prefill_chunks(len: usize, chunk: usize) -> Vec<(usize, usize)> {
    assert!(chunk > 0, "chunk must be > 0");
    let mut out = vec![]; let mut s = 0;
    while s < len { let e = (s + chunk).min(len); out.push((s, e)); s = e; }
    out
}

#[cfg(test)]
mod chunk_tests {
    use super::*;
    #[test]
    fn chunks_cover_prompt_exactly() {
        let c = prefill_chunks(1000, 256);
        assert_eq!(c.first(), Some(&(0, 256)));
        assert_eq!(c.last(), Some(&(768, 1000)));
        assert_eq!(c.iter().map(|(a, b)| b - a).sum::<usize>(), 1000);
        for w in c.windows(2) { assert_eq!(w[0].1, w[1].0); }
    }
    #[test]
    fn empty_and_exact_multiple() {
        assert!(prefill_chunks(0, 8).is_empty());
        assert_eq!(prefill_chunks(16, 8), vec![(0, 8), (8, 16)]);
    }
}
