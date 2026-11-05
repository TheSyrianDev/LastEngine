//! OpenAI-compatible HTTP surface (contract taken from what LastHarness/llama-swap use).
use crate::backend::{GenRequest, Shared, Stub};
use crate::cli::ServeArgs;
use axum::{
    extract::State,
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
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/props", get(props))
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/v1/chat/completions", post(chat))
        .with_state(be);
    let l = tokio::net::TcpListener::bind(&a.addr).await.map_err(|e| format!("bind {}: {e}", a.addr))?;
    eprintln!("lastengine listening on {}", a.addr);
    axum::serve(l, app).await.map_err(|e| e.to_string())
}

async fn models(State(be): State<Shared>) -> Json<Value> {
    Json(json!({"object":"list","data":[{"id":be.model_id(),"object":"model","owned_by":"lastengine","meta":{"n_params":be.n_params()}}]}))
}

async fn props(State(be): State<Shared>) -> Json<Value> {
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

async fn chat(State(be): State<Shared>, Json(body): Json<Value>) -> axum::response::Response {
    use axum::response::IntoResponse;
    let req = GenRequest {
        prompt: prompt_of(&body),
        max_tokens: body["max_tokens"].as_u64().unwrap_or(256) as usize,
    };
    let stream = body["stream"].as_bool().unwrap_or(false);
    let include_usage = body["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let id = "chatcmpl-lastengine";
    let model = be.model_id();
    if !stream {
        let h = tokio::task::spawn_blocking(move || {
            let mut text = String::new();
            let u = be.generate(&req, &mut |s| { text.push_str(s); true });
            (text, u)
        });
        let (text, u) = h.await.unwrap();
        return Json(json!({
            "id": id, "object": "chat.completion", "model": model,
            "choices": [{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
            "usage": {"prompt_tokens":u.prompt_tokens,"completion_tokens":u.completion_tokens,
                      "prompt_tokens_details":{"cached_tokens":u.cached_tokens}}
        })).into_response();
    }
    let (tx, rx) = mpsc::channel::<Event>(64);
    tokio::task::spawn_blocking(move || {
        let ev = |v: Value| Event::default().data(v.to_string());
        let chunk = |delta: Value, fin: Value| json!({"id":id,"object":"chat.completion.chunk","model":model,
            "choices":[{"index":0,"delta":delta,"finish_reason":fin}]});
        if tx.blocking_send(ev(chunk(json!({"role":"assistant"}), Value::Null))).is_err() { return; }
        // emit returns false when the receiver is dropped (client closed the socket): generation stops.
        let u = be.generate(&req, &mut |s| tx.blocking_send(ev(chunk(json!({"content":s}), Value::Null))).is_ok());
        let _ = tx.blocking_send(ev(chunk(json!({}), json!("stop"))));
        if include_usage {
            let _ = tx.blocking_send(ev(json!({"id":id,"object":"chat.completion.chunk","model":model,"choices":[],
                "usage":{"prompt_tokens":u.prompt_tokens,"completion_tokens":u.completion_tokens,
                         "prompt_tokens_details":{"cached_tokens":u.cached_tokens}}})));
        }
        let _ = tx.blocking_send(Event::default().data("[DONE]"));
    });
    Sse::new(ReceiverStream::new(rx).map(Ok::<_, Infallible>)).into_response()
}
