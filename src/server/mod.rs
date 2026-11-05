//! OpenAI-compatible HTTP surface (contract taken from what LastHarness/llama-swap use).
use crate::backend::{GenRequest, Shared, Stub};
mod stats;
use stats::{ReqStat, Stats};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use crate::cli::ServeArgs;
use axum::{
    extract::{Query, State},
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{convert::Infallible, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

pub async fn run(a: ServeArgs) -> Result<(), String> {
    let be: Shared = if a.stub || a.model.is_none() {
        Arc::new(Stub)
    } else {
        return Err("real backends are not implemented yet; use --stub".into());
    };
    let st = Arc::new(App { be, stats: Stats::new() });
    let app = Router::new()
        .route("/stats", get(stats_summary))
        .route("/stats/requests", get(stats_requests))
        .route("/v1/models", get(models))
        .route("/props", get(props))
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/v1/chat/completions", post(chat))
        .with_state(st);
    let l = tokio::net::TcpListener::bind(&a.addr).await.map_err(|e| format!("bind {}: {e}", a.addr))?;
    eprintln!("lastengine listening on {}", a.addr);
    axum::serve(l, app).await.map_err(|e| e.to_string())
}

pub struct App {
    be: Shared,
    stats: Stats,
}

fn model_info(be: &Shared) -> Value {
    json!({"id": be.model_id(), "n_ctx": be.n_ctx(), "n_params": be.n_params(), "engine": "lastengine", "version": env!("CARGO_PKG_VERSION")})
}

async fn stats_summary(State(a): State<Arc<App>>) -> Json<Value> {
    Json(a.stats.summary(model_info(&a.be)))
}

async fn stats_requests(State(a): State<Arc<App>>, Query(q): Query<std::collections::HashMap<String, String>>) -> Json<Value> {
    let n = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50);
    Json(json!({"requests": a.stats.recent(n)}))
}

async fn models(State(a): State<Arc<App>>) -> Json<Value> {
    let be = &a.be;
    Json(json!({"object":"list","data":[{"id":be.model_id(),"object":"model","owned_by":"lastengine","meta":{"n_params":be.n_params()}}]}))
}

async fn props(State(a): State<Arc<App>>) -> Json<Value> {
    let be = &a.be;
    Json(json!({
        "default_generation_settings": {"n_ctx": be.n_ctx()},
        "chat_template_caps": {"supports_tools": false, "supports_reasoning_effort": false},
        "modalities": {"vision": false}
    }))
}

fn prompt_of(body: &Value) -> String {
    body["messages"].as_array().map(|m| {
        m.iter().filter_map(|x| x["content"].as_str()).collect::<Vec<_>>().join("\n")
    }).unwrap_or_default()
}

async fn chat(State(app): State<Arc<App>>, Json(body): Json<Value>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let req = GenRequest {
        prompt: prompt_of(&body),
        max_tokens: body["max_tokens"].as_u64().unwrap_or(256) as usize,
    };
    let stream = body["stream"].as_bool().unwrap_or(false);
    let include_usage = body["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let id = "chatcmpl-lastengine";
    let be = app.be.clone();
    let model = be.model_id();
    let rid = app.stats.begin();
    let t0 = Instant::now();
    let started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    if !stream {
        let app2 = app.clone();
        let h = tokio::task::spawn_blocking(move || {
            let mut text = String::new();
            let mut ttft = None;
            let u = be.generate(&req, &mut |s| { ttft.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1000.0); text.push_str(s); true });
            let st = record(rid, started_unix_ms, t0, ttft, &u, true);
            app2.stats.finish(st.clone());
            (text, u, st)
        });
        let (text, u, st) = h.await.unwrap();
        return Json(json!({
            "id": id, "object": "chat.completion", "model": model,
            "choices": [{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
            "usage": {"prompt_tokens":u.prompt_tokens,"completion_tokens":u.completion_tokens,
                      "prompt_tokens_details":{"cached_tokens":u.cached_tokens}},
            "timings": timings(&st)
        })).into_response();
    }
    let (tx, rx) = mpsc::channel::<Event>(64);
    tokio::task::spawn_blocking(move || {
        let ev = |v: Value| Event::default().data(v.to_string());
        let chunk = |delta: Value, fin: Value| json!({"id":id,"object":"chat.completion.chunk","model":model,
            "choices":[{"index":0,"delta":delta,"finish_reason":fin}]});
        if tx.blocking_send(ev(chunk(json!({"role":"assistant"}), Value::Null))).is_err() { return; }
        // emit returns false when the receiver is dropped (client closed the socket): generation stops.
        let mut ttft = None;
        let mut alive = true;
        let u = be.generate(&req, &mut |s| {
            ttft.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1000.0);
            alive = tx.blocking_send(ev(chunk(json!({"content":s}), Value::Null))).is_ok();
            alive
        });
        let st = record(rid, started_unix_ms, t0, ttft, &u, alive);
        app.stats.finish(st.clone());
        let _ = tx.blocking_send(ev(chunk(json!({}), json!("stop"))));
        if include_usage {
            let _ = tx.blocking_send(ev(json!({"id":id,"object":"chat.completion.chunk","model":model,"choices":[],
                "usage":{"prompt_tokens":u.prompt_tokens,"completion_tokens":u.completion_tokens,
                         "prompt_tokens_details":{"cached_tokens":u.cached_tokens}},
                "timings": timings(&st)})));
        }
        let _ = tx.blocking_send(Event::default().data("[DONE]"));
    });
    Sse::new(ReceiverStream::new(rx).map(Ok::<_, Infallible>)).into_response()
}

fn record(id: u64, started_unix_ms: u64, t0: Instant, ttft: Option<f64>, u: &crate::backend::Usage, completed: bool) -> ReqStat {
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ttft = ttft.unwrap_or(total_ms);
    let new_prompt = u.prompt_tokens.saturating_sub(u.cached_tokens);
    let prefill_ms = if u.prefill_ms > 0.0 { u.prefill_ms } else { ttft };
    let decode_ms = (total_ms - ttft).max(0.0);
    let rate = |n: usize, ms: f64| if ms > 0.0 { n as f64 / (ms / 1000.0) } else { 0.0 };
    ReqStat {
        id, started_unix_ms,
        prompt_tokens: u.prompt_tokens, cached_tokens: u.cached_tokens, completion_tokens: u.completion_tokens,
        prefill_ms, prefill_tok_s: rate(new_prompt, prefill_ms),
        time_to_first_token_ms: ttft, decode_ms,
        // tokens after the first are produced during decode_ms
        decode_tok_s: rate(u.completion_tokens.saturating_sub(1), decode_ms),
        total_ms, completed,
    }
}

/// llama.cpp-style field names so existing tooling can read them.
fn timings(s: &ReqStat) -> Value {
    json!({
        "prompt_n": s.prompt_tokens.saturating_sub(s.cached_tokens), "prompt_ms": s.prefill_ms, "prompt_per_second": s.prefill_tok_s,
        "predicted_n": s.completion_tokens, "predicted_ms": s.decode_ms, "predicted_per_second": s.decode_tok_s,
        "cache_n": s.cached_tokens, "time_to_first_token_ms": s.time_to_first_token_ms,
    })
}
