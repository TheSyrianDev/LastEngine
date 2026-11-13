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

/// Per-request sampler for the batch path (S5): temperature + top_p over the top-64 logits, xorshift RNG owned by the request.
pub struct Picker { pub temperature: f32, pub top_p: f32, pub rng: u64 }
impl Picker {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Picker { Picker { temperature, top_p, rng: seed.wrapping_mul(0x9E3779B97F4A7C15) | 1 } }
    fn next(&mut self) -> f32 { let s = &mut self.rng; *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; ((*s >> 40) as f32) / (1u64 << 24) as f32 }
    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 { return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32; }
        let mut ix: Vec<u32> = (0..logits.len() as u32).collect();
        let k = 64.min(logits.len());
        ix.select_nth_unstable_by(k - 1, |a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
        let mut top: Vec<(u32, f32)> = ix[..k].iter().map(|&i| (i, logits[i as usize] / self.temperature)).collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mx = top[0].1;
        let mut p: Vec<f32> = top.iter().map(|t| (t.1 - mx).exp()).collect();
        let z: f32 = p.iter().sum(); p.iter_mut().for_each(|v| *v /= z);
        let mut keep = p.len(); let mut cum = 0.0;
        for (i, v) in p.iter().enumerate() { cum += v; if self.top_p > 0.0 && self.top_p < 1.0 && cum >= self.top_p { keep = i + 1; break; } }
        let z: f32 = p[..keep].iter().sum();
        let mut r = self.next() * z;
        for i in 0..keep { r -= p[i]; if r <= 0.0 { return top[i].0; } }
        top[keep - 1].0
    }
}

/// Streaming single-request generation with a sampler (S5). Greedy when picker.temperature <= 0.
pub fn generate_stream_sampled(gp: &mut DenseGpu, ids: &[u32], max_tokens: usize, eos: &[u32], pk: &mut Picker, on: &mut dyn FnMut(u32) -> bool) -> usize {
    gp.reset(); gp.set_slot(0);
    let mut lg = gp.prefill(ids);
    let mut n = 0;
    while n < max_tokens {
        let t = pk.pick(&lg);
        n += 1;
        let keep = on(t);
        if !keep || eos.contains(&t) || n >= max_tokens { break; }
        lg = gp.forward(t);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn picker_greedy_and_seeded() {
        let l = vec![0.1f32, 3.0, 0.2, 2.9];
        assert_eq!(Picker::new(0.0, 1.0, 1).pick(&l), 1);
        let mut a = Picker::new(0.8, 0.9, 7); let mut b = Picker::new(0.8, 0.9, 7);
        let xs: Vec<u32> = (0..20).map(|_| a.pick(&l)).collect();
        let ys: Vec<u32> = (0..20).map(|_| b.pick(&l)).collect();
        assert_eq!(xs, ys);
        assert!(xs.iter().all(|&t| t < 4));
    }
}
