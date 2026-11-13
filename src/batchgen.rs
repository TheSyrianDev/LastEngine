//! Batched greedy generation over DenseGpu slots (phase 2 step 3 refactor). Shared by the CLI gates and, later, the server.
//! Each prompt is prefilled into its own KV slot, then all live sequences decode together via batchplan::run_loop.
use crate::batchplan::{self, budget_step, Seq};
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

/// Interleaved serving loop (step 1): many requests share the GPU. Each round spends one token budget:
/// decodes of running sequences first (batchplan::budget_step), then prefill chunks of at most `chunk` tokens
/// for requests still ingesting their prompt. A long prompt therefore no longer stalls running decodes.
/// Greedy output is identical to generate_stream because decode_batch rows are batch-invariant (gate: tools/batch_invariance_gate.sh).
pub fn serve_interleaved(gp: &mut DenseGpu, rx: std::sync::mpsc::Receiver<crate::reqsched::Request>, eos: &[u32], chunk: usize, budget: usize) {
    use crate::reqsched::{Request, Token};
    struct Act { req: Request, ids: Vec<u32>, done: usize, slot: usize, pos: usize, next: Option<u32>, n: usize, max: usize, pk: Picker }
    let nslots = gp.slots(); let ctx = gp.ctx_len();
    let mut waiting: std::collections::VecDeque<Request> = Default::default();
    let mut act: Vec<Act> = vec![];
    let mut seed: u64 = 1;
    loop {
        if act.is_empty() && waiting.is_empty() { match rx.recv() { Ok(r) => waiting.push_back(r), Err(_) => return } }
        while let Ok(r) = rx.try_recv() { waiting.push_back(r); }
        // Admit into free slots.
        while act.len() < nslots {
            let Some(req) = waiting.pop_front() else { break };
            let max_pos = ctx.saturating_sub(1);
            let ids = if req.prompt_ids.len() > max_pos { req.prompt_ids[req.prompt_ids.len() - max_pos..].to_vec() } else { req.prompt_ids.clone() };
            let slot = (0..nslots).find(|s| !act.iter().any(|a| a.slot == *s)).unwrap();
            seed = seed.wrapping_add(1);
            let pk = Picker::new(req.sampling.temperature, req.sampling.top_p, seed);
            let max = req.max_tokens.min(max_pos.saturating_sub(ids.len()));
            if ids.is_empty() || max == 0 { let _ = req.tx.send(Token::Done { n: 0 }); continue; }
            act.push(Act { req, ids, done: 0, slot, pos: 0, next: None, n: 0, max, pk });
        }
        let mut finished: Vec<usize> = vec![];
        // Decode phase: emit each decoding sequence's pending token, then feed the survivors as one batch.
        let decoding: Vec<usize> = (0..act.len()).filter(|&i| act[i].next.is_some()).collect();
        let prefilling: Vec<usize> = (0..act.len()).filter(|&i| act[i].next.is_none()).collect();
        let mut feed: Vec<(usize, u32)> = vec![];
        for &i in &decoding {
            let a = &mut act[i]; let t = a.next.unwrap();
            if eos.contains(&t) { finished.push(i); continue; }
            a.n += 1;
            if a.req.tx.send(Token::Tok(t)).is_err() { finished.push(i); continue; }
            if a.n >= a.max { finished.push(i); continue; }
            feed.push((i, t));
        }
        // Budget: decodes spend one token each; the rest goes to prefill chunks in admission order.
        let (_, admits) = budget_step(&feed.iter().map(|x| x.0).collect::<Vec<_>>(), &prefilling.iter().map(|&i| (i, (act[i].ids.len() - act[i].done).min(chunk))).collect::<Vec<_>>(), budget, |_, _| true);
        if !feed.is_empty() {
            let toks: Vec<u32> = feed.iter().map(|x| x.1).collect();
            let slots: Vec<usize> = feed.iter().map(|x| act[x.0].slot).collect();
            let poss: Vec<usize> = feed.iter().map(|x| act[x.0].pos).collect();
            let logits = gp.decode_batch(&toks, &slots, &poss);
            for (k, &(i, _)) in feed.iter().enumerate() { let a = &mut act[i]; a.pos += 1; a.next = Some(a.pk.pick(&logits[k])); }
        }
        let mut left = budget.saturating_sub(feed.len());
        for &i in &admits {
            let a = &mut act[i];
            let take = (a.ids.len() - a.done).min(chunk).min(left);
            if take == 0 { break; }
            left -= take;
            let end = a.done + take; let last = end == a.ids.len();
            let lg = gp.prefill_at(a.slot, a.pos, &a.ids[a.done..end], last);
            a.pos += take; a.done = end;
            if last { a.next = Some(a.pk.pick(&lg)); }
        }
        finished.sort_unstable(); finished.dedup();
        for &i in finished.iter().rev() { let a = act.remove(i); let _ = a.req.tx.send(Token::Done { n: a.n }); }
    }
}
