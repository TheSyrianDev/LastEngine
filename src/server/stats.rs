//! Run statistics: per-request records and totals, exposed at /stats and /stats/requests.
use serde::Serialize;
use std::{collections::VecDeque, sync::Mutex, time::Instant};

#[derive(Serialize, Clone)]
pub struct ReqStat {
    pub id: u64,
    pub started_unix_ms: u64,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub prefill_tok_s: f64,
    pub time_to_first_token_ms: f64,
    pub decode_ms: f64,
    pub decode_tok_s: f64,
    pub total_ms: f64,
    /// false when the client disconnected before generation finished
    pub completed: bool,
}

#[derive(Default)]
struct Inner {
    next_id: u64,
    recent: VecDeque<ReqStat>,
    requests: u64,
    prompt_tokens: u64,
    cached_tokens: u64,
    completion_tokens: u64,
    prefill_ms: f64,
    decode_ms: f64,
    active: u64,
}

pub struct Stats {
    started: Instant,
    inner: Mutex<Inner>,
}

const KEEP: usize = 200;

impl Stats {
    pub fn new() -> Stats { Stats { started: Instant::now(), inner: Mutex::new(Inner::default()) } }

    pub fn begin(&self) -> u64 {
        let mut g = self.inner.lock().unwrap();
        g.next_id += 1;
        g.active += 1;
        g.next_id
    }

    pub fn finish(&self, r: ReqStat) {
        let mut g = self.inner.lock().unwrap();
        g.active = g.active.saturating_sub(1);
        g.requests += 1;
        g.prompt_tokens += r.prompt_tokens as u64;
        g.cached_tokens += r.cached_tokens as u64;
        g.completion_tokens += r.completion_tokens as u64;
        g.prefill_ms += r.prefill_ms;
        g.decode_ms += r.decode_ms;
        if g.recent.len() == KEEP { g.recent.pop_front(); }
        g.recent.push_back(r);
    }

    pub fn summary(&self, model: serde_json::Value) -> serde_json::Value {
        let g = self.inner.lock().unwrap();
        let rate = |tok: u64, ms: f64| if ms > 0.0 { tok as f64 / (ms / 1000.0) } else { 0.0 };
        let last = g.recent.back().cloned();
        let n = g.recent.len().max(1) as f64;
        let avg_ttft = g.recent.iter().map(|r| r.time_to_first_token_ms).sum::<f64>() / n;
        serde_json::json!({
            "model": model,
            "uptime_s": self.started.elapsed().as_secs_f64(),
            "active_requests": g.active,
            "totals": {
                "requests": g.requests,
                "prompt_tokens": g.prompt_tokens,
                "cached_tokens": g.cached_tokens,
                "completion_tokens": g.completion_tokens,
                "prefill_tok_s": rate(g.prompt_tokens - g.cached_tokens.min(g.prompt_tokens), g.prefill_ms),
                "decode_tok_s": rate(g.completion_tokens, g.decode_ms),
                "avg_time_to_first_token_ms": avg_ttft,
            },
            "last_request": last,
        })
    }

    pub fn recent(&self, limit: usize) -> Vec<ReqStat> {
        let g = self.inner.lock().unwrap();
        g.recent.iter().rev().take(limit).cloned().collect()
    }
}
