//! OpenAI-compatible HTTP surface (contract taken from what LastHarness/llama-swap use).
use crate::backend::{Delta, GenRequest, Sampling, Shared, Stub};
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
    // Bind first so clients get 503 (not connection refused) while the model loads.
    let l = tokio::net::TcpListener::bind(&a.addr).await.map_err(|e| format!("bind {}: {e}", a.addr))?;
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let loading = Router::new().fallback(|| async { (axum::http::StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status":"loading model"}))) });
    let h = tokio::spawn(async move { let _ = axum::serve(l, loading).with_graceful_shutdown(async { let _ = rx.await; }).await; });
    let be: Shared = if a.stub || a.model.is_none() {
        Arc::new(Stub)
    } else {
        let path = a.model.clone().unwrap();
        if let Some(r) = &a.rope_scaling { std::env::set_var("LE_ROPE", r); }
        if let Some(r) = a.rope_scale { std::env::set_var("LE_ROPE_SCALE", r.to_string()); }
        if let Some(r) = a.yarn_orig_ctx { std::env::set_var("LE_YARN_ORIG", r.to_string()); }
        if let Some(c) = a.chunk { std::env::set_var("LE_TMAX", c.to_string()); }
        if let Some(k) = &a.kv { std::env::set_var("LE_KV", k); }
        let (ctx, margin) = (a.ctx.unwrap_or(8192), a.vram_margin.unwrap_or(128));
        let tf = a.chat_template_file.clone();
        let l = tokio::task::spawn_blocking(move || crate::local::Local::load(&path, ctx, margin, tf.as_deref())).await.map_err(|e| e.to_string())??;
        Arc::new(l)
    };
    let st = Arc::new(App { be, slot_dir: a.slot_save_path.clone().map(std::path::PathBuf::from), stats: Stats::new() });
    let app = Router::new()
        .route("/", get(|| async { axum::response::Html(include_str!("ui.html")) }))
        .route("/stats", get(stats_summary))
        .route("/stats/requests", get(stats_requests))
        .route("/v1/models", get(models))
        .route("/props", get(props))
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/v1/chat/completions", post(chat))
        .route("/slots/{id}", post(slot_action))
        .with_state(st);
    let _ = tx.send(());
    let _ = h.await;
    let l = tokio::net::TcpListener::bind(&a.addr).await.map_err(|e| format!("bind {}: {e}", a.addr))?;
    eprintln!("lastengine listening on {}", a.addr);
    axum::serve(l, app).await.map_err(|e| e.to_string())
}

pub struct App {
    be: Shared,
    slot_dir: Option<std::path::PathBuf>,
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
        "chat_template_caps": {"supports_tools": a.be.supports_tools(), "supports_reasoning_effort": false},
        "modalities": {"vision": false}
    }))
}

fn sampling_of(b: &Value) -> Sampling {
    let f = |k: &str, d: f32| b[k].as_f64().map(|v| v as f32).unwrap_or(d);
    Sampling { temperature: f("temperature", 0.7), top_p: f("top_p", 0.95), top_k: b["top_k"].as_u64().unwrap_or(40) as usize, min_p: f("min_p", 0.0),
               presence_penalty: f("presence_penalty", 0.0), repeat_penalty: f("repeat_penalty", 1.0), seed: b["seed"].as_u64() }
}

fn tool_calls_json(u: &crate::backend::Usage) -> Value {
    Value::Array(u.tool_calls.iter().enumerate().map(|(i, c)| json!({"index":i,"id":format!("call_{i}"),"type":"function","function":{"name":c.name,"arguments":c.arguments}})).collect())
}

async fn chat(State(app): State<Arc<App>>, Json(body): Json<Value>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let req = GenRequest {
        messages: body["messages"].as_array().cloned().unwrap_or_default(),
        tools: body.get("tools").filter(|t| t.as_array().map_or(false, |a| !a.is_empty())).cloned(),
        max_tokens: body["max_tokens"].as_u64().or(body["max_completion_tokens"].as_u64()).unwrap_or(4096) as usize,
        sampling: sampling_of(&body),
        enable_thinking: body["chat_template_kwargs"]["enable_thinking"].as_bool().unwrap_or(true),
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
            let (mut text, mut reasoning) = (String::new(), String::new());
            let mut ttft = None;
            let u = be.generate(&req, &mut |d| { ttft.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1000.0); match d { Delta::Content(s) => text.push_str(&s), Delta::Reasoning(s) => reasoning.push_str(&s) } true });
            let st = record(rid, started_unix_ms, t0, ttft, &u, true);
            app2.stats.finish(st.clone());
            (text, reasoning, u, st)
        });
        let (text, reasoning, u, st) = h.await.unwrap();
        return Json(json!({
            "id": id, "object": "chat.completion", "model": model,
            "choices": [{"index":0,"message":{"role":"assistant","content":text,"reasoning_content":reasoning,"tool_calls":tool_calls_json(&u)},"finish_reason":u.finish_reason}],
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
        let u = be.generate(&req, &mut |d| {
            ttft.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1000.0);
            let delta = match d { Delta::Content(s) => json!({"content":s}), Delta::Reasoning(s) => json!({"reasoning_content":s}) };
            alive = tx.blocking_send(ev(chunk(delta, Value::Null))).is_ok();
            alive
        });
        if !u.tool_calls.is_empty() { let _ = tx.blocking_send(ev(chunk(json!({"tool_calls":tool_calls_json(&u)}), Value::Null))); }
        let st = record(rid, started_unix_ms, t0, ttft, &u, alive);
        app.stats.finish(st.clone());
        let _ = tx.blocking_send(ev(chunk(json!({}), json!(u.finish_reason))));
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

async fn slot_action(State(app): State<Arc<App>>, axum::extract::Path(id): axum::extract::Path<u32>,
                     axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>, Json(body): Json<Value>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let err = |code: u16, m: &str| (axum::http::StatusCode::from_u16(code).unwrap(), Json(json!({"error": {"code": code, "message": m, "type": "invalid_request_error"}}))).into_response();
    let Some(dir) = app.slot_dir.clone() else { return err(501, "slot save path not configured (--slot-save-path)") };
    let Some(name) = body.get("filename").and_then(|v| v.as_str()).map(|s| s.to_string()) else { return err(400, "filename required") };
    if name.is_empty() || name.contains('/') || name.contains("..") || name.contains('\\') { return err(400, "invalid filename"); }
    let path = dir.join(&name);
    let action = q.get("action").cloned().unwrap_or_default();
    let be = app.be.clone();
    let t0 = std::time::Instant::now();
    let r = tokio::task::spawn_blocking(move || match action.as_str() {
        "save" => be.slot_save(&path).map(|r| (true, r)),
        "restore" => be.slot_restore(&path).map(|r| (false, r)),
        _ => Err("action must be save or restore".into()),
    }).await.unwrap();
    match r {
        Ok((save, (n, bytes))) => {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if save { Json(json!({"id_slot": id, "filename": name, "n_saved": n, "n_written": bytes, "timings": {"save_ms": ms}})).into_response() }
            else { Json(json!({"id_slot": id, "filename": name, "n_restored": n, "n_read": bytes, "timings": {"restore_ms": ms}})).into_response() }
        }
        Err(e) => err(400, &e),
    }
}
