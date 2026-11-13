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
    let max_queue = std::env::var("LE_MAX_QUEUE").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize);
    let st = Arc::new(App { be, slot_dir: a.slot_save_path.clone().map(std::path::PathBuf::from), stats: Stats::new(), pending: std::sync::atomic::AtomicUsize::new(0), max_queue });
    let app = Router::new()
        .route("/", get(web_index))
        .route("/index.html", get(web_index))
        .route("/favicon.svg", get(|| async { ([("content-type","image/svg+xml")], include_str!("../../assets/webui/favicon.svg")) }))
        .route("/_app/version.json", get(|| async { ([("content-type","application/json")], include_str!("../../assets/webui/_app/version.json")) }))
        .route("/_app/immutable/bundle.B83VQjAo.js", get(|| async { ([("content-type","text/javascript"),("cache-control","max-age=3600")], include_str!("../../assets/webui/_app/immutable/bundle.B83VQjAo.js")) }))
        .route("/_app/immutable/assets/bundle.DN2gXIUA.css", get(|| async { ([("content-type","text/css"),("cache-control","max-age=3600")], include_str!("../../assets/webui/_app/immutable/assets/bundle.DN2gXIUA.css")) }))
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

async fn web_index() -> axum::response::Html<&'static str> { axum::response::Html(include_str!("../../assets/webui/index.html")) }

pub struct App {
    be: Shared,
    slot_dir: Option<std::path::PathBuf>,
    stats: Stats,
    // Requests admitted (waiting for or running a generation). Phase 1 of docs/concurrency-plan.md: one active
    // generation, a bounded queue behind it, 429 when full.
    pending: std::sync::atomic::AtomicUsize,
    max_queue: usize,
}

/// Releases one admitted-request slot when the generation task finishes (or is dropped).
struct QueueGuard(Arc<App>);
impl Drop for QueueGuard { fn drop(&mut self) { self.0.pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst); } }

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
    Json(json!({"object":"list","data":[{"id":be.model_id(),"object":"model","created":0,"owned_by":"lastengine","meta":{"n_params":be.n_params()}}]}))
}

async fn props(State(a): State<Arc<App>>) -> Json<Value> {
    let be = &a.be;
    Json(json!({
        "total_slots": 1, "model_alias": be.model_id(), "model_path": be.model_id(), "build_info": format!("lastengine {}", env!("CARGO_PKG_VERSION")),
        "default_generation_settings": {"n_ctx": be.n_ctx(), "params": {"n_predict": -1, "temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.0}},
        "chat_template_caps": {"supports_tools": a.be.supports_tools(), "supports_reasoning_effort": false},
        "modalities": {"vision": false}
    }))
}

fn sampling_of(b: &Value) -> Sampling {
    let f = |k: &str, d: f32| b[k].as_f64().map(|v| v as f32).unwrap_or(d);
    Sampling { temperature: f("temperature", 0.7), top_p: f("top_p", 0.95), top_k: b["top_k"].as_u64().unwrap_or(40) as usize, min_p: f("min_p", 0.0),
               presence_penalty: f("presence_penalty", 0.0), repeat_penalty: f("repeat_penalty", 1.0), seed: b["seed"].as_u64() }
}


/// Holds back text that could still become a stop string; `push` returns (text safe to emit, stop hit).
struct StopFilter { stops: Vec<String>, buf: String }
impl StopFilter {
    fn new(stops: Vec<String>) -> Self { StopFilter { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), buf: String::new() } }
    fn push(&mut self, s: &str) -> (String, bool) {
        if self.stops.is_empty() { return (s.to_string(), false); }
        self.buf.push_str(s);
        if let Some(i) = self.stops.iter().filter_map(|st| self.buf.find(st.as_str())).min() {
            let out = self.buf[..i].to_string(); self.buf.clear(); return (out, true);
        }
        let mut hold = 0;
        for st in &self.stops {
            for k in (1..st.len()).rev() {
                if st.is_char_boundary(k) && self.buf.ends_with(&st[..k]) { hold = hold.max(k); break; }
            }
        }
        let cut = self.buf.len() - hold;
        let out = self.buf[..cut].to_string(); self.buf.drain(..cut);
        (out, false)
    }
    fn flush(&mut self) -> String { std::mem::take(&mut self.buf) }
}

/// Sets the flag when the request future is dropped (client closed the connection) unless disarmed.
struct CancelOnDrop(Arc<std::sync::atomic::AtomicBool>, bool);
impl Drop for CancelOnDrop { fn drop(&mut self) { if self.1 { self.0.store(true, std::sync::atomic::Ordering::Relaxed); } } }

fn stops_of(b: &Value) -> Vec<String> {
    match &b["stop"] { Value::String(s) => vec![s.clone()], Value::Array(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(), _ => vec![] }
}

fn tool_calls_json(u: &crate::backend::Usage) -> Value {
    Value::Array(u.tool_calls.iter().enumerate().map(|(i, c)| json!({"index":i,"id":format!("call_{i}"),"type":"function","function":{"name":c.name,"arguments":c.arguments}})).collect())
}

/// OpenAI-style error body: {"error":{"message","type","code","param"}} with the given HTTP status.
fn api_error(status: u16, typ: &str, code: &str, message: &str) -> axum::response::Response {
    use axum::response::IntoResponse;
    let st = axum::http::StatusCode::from_u16(status).unwrap_or(axum::http::StatusCode::BAD_REQUEST);
    (st, Json(json!({"error": {"message": message, "type": typ, "code": code, "param": null}}))).into_response()
}

async fn chat(State(app): State<Arc<App>>, raw: axum::body::Bytes) -> axum::response::Response {
    use axum::response::IntoResponse;
    let body: Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(e) => return api_error(400, "invalid_request_error", "invalid_json", &format!("request body is not valid JSON: {e}")),
    };
    if !body.is_object() { return api_error(400, "invalid_request_error", "invalid_body", "request body must be a JSON object"); }
    match body.get("messages").and_then(|m| m.as_array()) {
        Some(m) if !m.is_empty() => {}
        _ => return api_error(400, "invalid_request_error", "missing_messages", "'messages' must be a non-empty array"),
    }
    if let Some(t) = body["temperature"].as_f64() { if !(0.0..=2.0).contains(&t) { return api_error(400, "invalid_request_error", "invalid_temperature", "'temperature' must be between 0 and 2"); } }
    if let Some(p) = body["top_p"].as_f64() { if !(p > 0.0 && p <= 1.0) { return api_error(400, "invalid_request_error", "invalid_top_p", "'top_p' must be in (0, 1]"); } }
    if let Some(m) = body["max_tokens"].as_u64().or(body["max_completion_tokens"].as_u64()) { if m == 0 { return api_error(400, "invalid_request_error", "invalid_max_tokens", "'max_tokens' must be at least 1"); } }
    // tool_choice: "none" hides tools; a named function keeps only that tool, so the model can only call it.
    let tc = &body["tool_choice"];
    let all_tools = body.get("tools").and_then(|t| t.as_array()).cloned().unwrap_or_default();
    let tools_val: Option<Value> = if tc.as_str() == Some("none") { None } else {
        let want = tc["function"]["name"].as_str();
        let kept: Vec<Value> = all_tools.into_iter().filter(|t| want.map_or(true, |w| t["function"]["name"].as_str() == Some(w))).collect();
        if kept.is_empty() { None } else { Some(Value::Array(kept)) }
    };
    // reasoning_effort "none" turns thinking off; chat_template_kwargs.enable_thinking wins when given.
    let think_default = body["reasoning_effort"].as_str().map_or(true, |e| e != "none");
    let req = GenRequest {
        messages: body["messages"].as_array().cloned().unwrap_or_default(),
        tools: tools_val,
        max_tokens: body["max_tokens"].as_u64().or(body["max_completion_tokens"].as_u64()).unwrap_or(4096) as usize,
        sampling: sampling_of(&body),
        enable_thinking: body["chat_template_kwargs"]["enable_thinking"].as_bool().unwrap_or(think_default),
    };
    if let Err(e) = app.be.check(&req) { return api_error(e.status, "invalid_request_error", e.code, &e.message); }
    if app.pending.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= app.max_queue {
        app.pending.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        return api_error(429, "rate_limit_error", "server_busy", &format!("server is busy: {} requests already queued or running; retry later", app.max_queue));
    }
    let stream = body["stream"].as_bool().unwrap_or(false);
    let include_usage = body["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let id = "chatcmpl-lastengine";
    let be = app.be.clone();
    let model = be.model_id();
    let rid = app.stats.begin();
    let t0 = Instant::now();
    let started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let stops = stops_of(&body);
    if !stream {
        let app2 = app.clone();
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut guard = CancelOnDrop(cancel.clone(), true);
        let h = tokio::task::spawn_blocking(move || {
            let _q = QueueGuard(app2.clone());
            let (mut text, mut reasoning) = (String::new(), String::new());
            let mut ttft = None; let mut sf = StopFilter::new(stops.clone()); let mut sfr = StopFilter::new(stops); let mut hit = false;
            let mut u = be.generate(&req, &mut |d| {
                ttft.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1000.0);
                match d {
                    Delta::Content(s) => { let (o, h) = sf.push(&s); text.push_str(&o); if h { hit = true; return false; } }
                    Delta::Reasoning(s) => { let (o, h) = sfr.push(&s); reasoning.push_str(&o); if h { hit = true; return false; } }
                }
                !cancel.load(std::sync::atomic::Ordering::Relaxed)
            });
            if !hit { text.push_str(&sf.flush()); reasoning.push_str(&sfr.flush()); } else { u.finish_reason = "stop"; }
            let st = record(rid, started_unix_ms, t0, ttft, &u, true);
            app2.stats.finish(st.clone());
            (text, reasoning, u, st)
        });
        let (text, reasoning, u, st) = h.await.unwrap();
        guard.1 = false;
        return Json(json!({
            "id": id, "object": "chat.completion", "model": model,
            "choices": [{"index":0,"message":{"role":"assistant","content":text,"reasoning_content":reasoning,"tool_calls":if u.tool_calls.is_empty() { Value::Null } else { tool_calls_json(&u) }},"finish_reason":u.finish_reason}],
            "usage": {"prompt_tokens":u.prompt_tokens,"completion_tokens":u.completion_tokens,"total_tokens":u.prompt_tokens+u.completion_tokens,
                      "prompt_tokens_details":{"cached_tokens":u.cached_tokens}},
            "timings": timings(&st)
        })).into_response();
    }
    let (tx, rx) = mpsc::channel::<Event>(64);
    tokio::task::spawn_blocking(move || {
        let _q = QueueGuard(app.clone());
        let ev = |v: Value| Event::default().data(v.to_string());
        let chunk = |delta: Value, fin: Value| json!({"id":id,"object":"chat.completion.chunk","model":model,
            "choices":[{"index":0,"delta":delta,"finish_reason":fin}]});
        if tx.blocking_send(ev(chunk(json!({"role":"assistant"}), Value::Null))).is_err() { return; }
        // emit returns false when the receiver is dropped (client closed the socket): generation stops.
        let mut ttft = None;
        let mut alive = true;
        let mut sf = StopFilter::new(stops.clone()); let mut sfr = StopFilter::new(stops); let mut hit = false;
        let mut u = be.generate(&req, &mut |d| {
            ttft.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1000.0);
            let delta = match d {
                Delta::Content(s) => { let (o, h) = sf.push(&s); if h { hit = true; } if o.is_empty() { None } else { Some(json!({"content":o})) } }
                Delta::Reasoning(s) => { let (o, h) = sfr.push(&s); if h { hit = true; } if o.is_empty() { None } else { Some(json!({"reasoning_content":o})) } }
            };
            if let Some(delta) = delta { alive = tx.blocking_send(ev(chunk(delta, Value::Null))).is_ok(); }
            alive && !hit
        });
        if !hit {
            let rest = sf.flush(); if !rest.is_empty() { alive = tx.blocking_send(ev(chunk(json!({"content":rest}), Value::Null))).is_ok(); }
            let rr = sfr.flush(); if !rr.is_empty() { alive = tx.blocking_send(ev(chunk(json!({"reasoning_content":rr}), Value::Null))).is_ok(); }
        } else { u.finish_reason = "stop"; }
        if !u.tool_calls.is_empty() { let _ = tx.blocking_send(ev(chunk(json!({"tool_calls":tool_calls_json(&u)}), Value::Null))); }
        let st = record(rid, started_unix_ms, t0, ttft, &u, alive);
        app.stats.finish(st.clone());
        let _ = tx.blocking_send(ev(chunk(json!({}), json!(u.finish_reason))));
        if include_usage {
            let _ = tx.blocking_send(ev(json!({"id":id,"object":"chat.completion.chunk","model":model,"choices":[],
                "usage":{"prompt_tokens":u.prompt_tokens,"completion_tokens":u.completion_tokens,"total_tokens":u.prompt_tokens+u.completion_tokens,
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
