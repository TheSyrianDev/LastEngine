//! Inference backend interface the server talks to. Real engines implement `Backend`; `Stub` echoes for protocol tests.
use std::sync::Arc;

pub struct GenRequest {
    pub prompt: String,
    pub max_tokens: usize,
}

pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    /// Time the backend spent on prompt processing before the first token (ms).
    pub prefill_ms: f64,
}

pub trait Backend: Send + Sync {
    fn model_id(&self) -> String;
    fn n_ctx(&self) -> usize;
    fn n_params(&self) -> u64;
    /// Streams text pieces to `emit`; `emit` returns false when the client is gone (stop generating).
    fn generate(&self, req: &GenRequest, emit: &mut dyn FnMut(&str) -> bool) -> Usage;
}

pub struct Stub;

impl Backend for Stub {
    fn model_id(&self) -> String { "stub".into() }
    fn n_ctx(&self) -> usize { 4096 }
    fn n_params(&self) -> u64 { 0 }
    fn generate(&self, req: &GenRequest, emit: &mut dyn FnMut(&str) -> bool) -> Usage {
        let words: Vec<&str> = req.prompt.split_whitespace().collect();
        let mut n = 0;
        for w in words.iter().rev().take(req.max_tokens) {
            if !emit(&format!("{w} ")) { break; }
            n += 1;
        }
        Usage { prompt_tokens: words.len(), completion_tokens: n, cached_tokens: 0, prefill_ms: 0.0 }
    }
}

pub type Shared = Arc<dyn Backend>;
