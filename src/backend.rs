//! Inference backend interface the server talks to. Real engines implement `Backend`; `Stub` echoes for protocol tests.
use serde_json::Value;
use std::sync::Arc;

#[derive(Default, Clone)]
pub struct Sampling { pub temperature: f32, pub top_p: f32, pub top_k: usize, pub min_p: f32, pub presence_penalty: f32, pub repeat_penalty: f32, pub seed: Option<u64> }

pub struct GenRequest {
    pub messages: Vec<Value>,
    pub tools: Option<Value>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub enable_thinking: bool,
}

pub struct ToolCall { pub name: String, pub arguments: String }

pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    /// Time the backend spent on prompt processing before the first token (ms).
    pub prefill_ms: f64,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: &'static str,
}

pub enum Delta { Content(String), Reasoning(String) }

/// A client-facing error (HTTP status + OpenAI-style type/code). Returned by `Backend::check` before generation starts.
pub struct ApiError { pub status: u16, pub code: &'static str, pub message: String }

pub trait Backend: Send + Sync {
    fn model_id(&self) -> String;
    fn n_ctx(&self) -> usize;
    fn n_params(&self) -> u64;
    fn supports_tools(&self) -> bool { false }
    /// Streams deltas to `emit`; `emit` returns false when the client is gone (stop generating).
    fn generate(&self, req: &GenRequest, emit: &mut dyn FnMut(Delta) -> bool) -> Usage;
    /// Pre-flight validation run before generation: template errors, prompts that do not fit the context window.
    fn check(&self, _req: &GenRequest) -> Result<(), ApiError> { Ok(()) }
    /// Save the current context (tokens + model state) to `path`; returns (tokens, bytes).
    fn slot_save(&self, _path: &std::path::Path) -> Result<(usize, usize), String> { Err("slots not supported".into()) }
    fn slot_restore(&self, _path: &std::path::Path) -> Result<(usize, usize), String> { Err("slots not supported".into()) }
}

pub struct Stub;

impl Backend for Stub {
    fn model_id(&self) -> String { "stub".into() }
    fn n_ctx(&self) -> usize { 4096 }
    fn n_params(&self) -> u64 { 0 }
    fn generate(&self, req: &GenRequest, emit: &mut dyn FnMut(Delta) -> bool) -> Usage {
        let prompt = req.messages.iter().filter_map(|x| x["content"].as_str()).collect::<Vec<_>>().join("\n");
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let mut n = 0;
        for w in words.iter().rev().take(req.max_tokens) {
            if !emit(Delta::Content(format!("{w} "))) { break; }
            n += 1;
        }
        Usage { prompt_tokens: words.len(), completion_tokens: n, cached_tokens: 0, prefill_ms: 0.0, tool_calls: vec![], finish_reason: "stop" }
    }
}

pub type Shared = Arc<dyn Backend>;
