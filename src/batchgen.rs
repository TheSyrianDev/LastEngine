//! Batched greedy generation over DenseGpu slots (phase 2 step 3 refactor). Shared by the CLI gates and, later, the server.
//! Each prompt is prefilled into its own KV slot, then all live sequences decode together via batchplan::run_loop.
use crate::batchplan::{self, Seq};
use crate::gpu::dense::DenseGpu;

/// Greedy-generate `max_tokens` (or until eos) for each prompt. Requires prompts.len() <= gp.slots().
/// Returns the generated token ids per prompt, in prompt order.
pub fn generate(gp: &mut DenseGpu, prompts: &[Vec<u32>], max_tokens: usize, eos: &[u32]) -> Vec<Vec<u32>> {
    assert!(gp.slots() >= prompts.len(), "need LE_SLOTS>={}", prompts.len());
    let ctx = gp.ctx_len();
    gp.reset();
    let mut seqs: Vec<(u64, Seq)> = vec![];
    for (i, ids) in prompts.iter().enumerate() {
        gp.set_slot(i);
        let lg = gp.prefill(ids);
        seqs.push((i as u64, Seq { slot: i, pos: ids.len(), next_tok: batchplan::greedy_rows(&[lg])[0], generated: 0, max_tokens }));
    }
    let mut outs: Vec<Vec<u32>> = vec![vec![]; prompts.len()];
    // Prefill already produced the first token per sequence; it is emitted by the loop as next_tok.
    batchplan::run_loop(&mut seqs, eos, ctx, gp.slots(), |plan| {
        for (k, &tok) in plan.toks.iter().enumerate() { outs[plan.slots[k]].push(tok); }
        let logits = gp.decode_batch(&plan.toks, &plan.slots, &plan.poss);
        batchplan::greedy_rows(&logits)
    }, |_, _| {});
    outs
}

/// Streaming single-request greedy generation (S2): prefill into slot 0, then emit each token via `on`.
/// `on` returns false to stop early. Returns the number of tokens emitted (eos is emitted, then generation stops).
pub fn generate_stream(gp: &mut DenseGpu, ids: &[u32], max_tokens: usize, eos: &[u32], on: &mut dyn FnMut(u32) -> bool) -> usize {
    gp.reset(); gp.set_slot(0);
    let mut lg = gp.prefill(ids);
    let mut n = 0;
    while n < max_tokens {
        let t = batchplan::greedy_rows(&[lg.clone()])[0];
        n += 1;
        let keep = on(t);
        if !keep || eos.contains(&t) || n >= max_tokens { break; }
        lg = gp.forward(t);
    }
    n
}
