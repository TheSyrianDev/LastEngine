//! Real backend: GGUF model on the CUDA engine, with prefix reuse, sampling and reasoning/tool-call splitting.
use crate::backend::*;
use crate::{cpu_ref::Model, gguf::Gguf, gpu::Gpu, template, tokenizer::Tokenizer};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

struct Inner { gpu: Gpu, hist: Vec<u32>, rng: u64 }
pub struct Local { inner: Mutex<Inner>, tok: Tokenizer, tmpl: String, name: String, n_ctx: usize, n_params: u64, bos: String, eos: String }

impl Local {
    pub fn load(path: &str, ctx: usize, margin: usize, tmpl_file: Option<&str>) -> Result<Local, String> {
        let g = Gguf::open(path).map_err(|e| format!("{e}"))?;
        let tok = Tokenizer::from_gguf(&g)?;
        let tmpl = match tmpl_file {
            Some(f) if f != "gguf" => std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?,
            _ => g.kv.get("tokenizer.chat_template").and_then(|v| v.as_str().map(String::from)).ok_or("model has no chat template")?,
        };
        let name = std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let n_params = g.tensors.iter().map(|t| t.dims.iter().product::<u64>()).sum();
        let m = Arc::new(Model::load(path, &g));
        let gpu = Gpu::new(m, ctx, margin);
        let s = |id: Option<u32>| id.map(|i| tok.tokens[i as usize].clone()).unwrap_or_default();
        let (bos, eos) = (s(tok.bos), s(tok.eos));
        Ok(Local { inner: Mutex::new(Inner { gpu, hist: vec![], rng: 0x9E3779B97F4A7C15 }), tok, tmpl, name, n_ctx: ctx, n_params, bos, eos })
    }
}

fn fix_messages(msgs: &[Value]) -> Vec<Value> {
    msgs.iter().map(|m| {
        let mut m = m.clone();
        if let Some(a) = m["content"].as_array() {
            let t: String = a.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("");
            m["content"] = json!(t);
        }
        if m["content"].is_null() { m["content"] = json!(""); }
        if let Some(tc) = m["tool_calls"].as_array_mut() {
            for c in tc { if let Some(s) = c["function"]["arguments"].as_str() { if let Ok(v) = serde_json::from_str::<Value>(s) { c["function"]["arguments"] = v; } } }
        }
        m
    }).collect()
}

fn next_rand(s: &mut u64) -> f32 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; ((*s >> 40) as f32) / (1u64 << 24) as f32 }

fn sample(logits: &mut [f32], sp: &Sampling, hist: &[u32], rng: &mut u64) -> u32 {
    if sp.repeat_penalty > 1.0 || sp.presence_penalty != 0.0 {
        let tail = &hist[hist.len().saturating_sub(256)..];
        for &t in tail { let l = &mut logits[t as usize]; if sp.repeat_penalty > 1.0 { *l = if *l > 0.0 { *l / sp.repeat_penalty } else { *l * sp.repeat_penalty }; } *l -= sp.presence_penalty; }
    }
    if sp.temperature <= 0.0 { return logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32; }
    let k = if sp.top_k == 0 { 64 } else { sp.top_k.min(256) };
    let mut ix: Vec<u32> = (0..logits.len() as u32).collect();
    ix.select_nth_unstable_by(k - 1, |a, b| logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap());
    let mut top: Vec<(u32, f32)> = ix[..k].iter().map(|&i| (i, logits[i as usize] / sp.temperature)).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mx = top[0].1;
    let mut p: Vec<f32> = top.iter().map(|t| (t.1 - mx).exp()).collect();
    let z: f32 = p.iter().sum(); p.iter_mut().for_each(|v| *v /= z);
    let mut keep = p.len(); let mut cum = 0.0;
    for (i, v) in p.iter().enumerate() { cum += v; if sp.top_p > 0.0 && sp.top_p < 1.0 && cum >= sp.top_p { keep = i + 1; break; } }
    if sp.min_p > 0.0 { let thr = p[0] * sp.min_p; keep = keep.min(p.iter().position(|v| *v < thr).unwrap_or(p.len())).max(1); }
    let z: f32 = p[..keep].iter().sum();
    let mut r = next_rand(rng) * z;
    for i in 0..keep { r -= p[i]; if r <= 0.0 { return top[i].0; } }
    top[keep - 1].0
}

/// Splits generated text into reasoning / content / tool-call text while streaming.
struct Splitter { buf: String, mode: u8, tool: String } // mode 0 content, 1 reasoning, 2 tool
impl Splitter {
    fn push(&mut self, s: &str, out: &mut Vec<Delta>) {
        self.buf.push_str(s);
        loop {
            let (marker, next): (&str, u8) = match self.mode { 1 => ("</think>", 0), 0 => ("<tool_call>", 2), _ => { self.tool.push_str(&std::mem::take(&mut self.buf)); return; } };
            if let Some(p) = self.buf.find(marker) {
                let head: String = self.buf[..p].to_string();
                if !head.is_empty() { out.push(if self.mode == 1 { Delta::Reasoning(head) } else { Delta::Content(head) }); }
                self.buf = self.buf[p + marker.len()..].trim_start_matches(if self.mode == 1 { "\n" } else { "" }).to_string();
                if next == 2 { self.tool.push_str("<tool_call>"); }
                self.mode = next; continue;
            }
            // hold back a possible partial marker at the end
            let mut hold = 0;
            for i in 1..marker.len().min(self.buf.len() + 1) { if self.buf.is_char_boundary(self.buf.len() - i) && marker.starts_with(&self.buf[self.buf.len() - i..]) { hold = i; } }
            let emit_to = self.buf.len() - hold;
            if emit_to > 0 { let t = self.buf[..emit_to].to_string(); out.push(if self.mode == 1 { Delta::Reasoning(t) } else { Delta::Content(t) }); self.buf = self.buf[emit_to..].to_string(); }
            return;
        }
    }
    fn finish(&mut self, out: &mut Vec<Delta>) {
        if self.mode == 2 { self.tool.push_str(&std::mem::take(&mut self.buf)); return; }
        let t = std::mem::take(&mut self.buf);
        if !t.is_empty() { out.push(if self.mode == 1 { Delta::Reasoning(t) } else { Delta::Content(t) }); }
    }
}

fn parse_tools(s: &str) -> Vec<ToolCall> {
    let mut v = vec![];
    for blk in s.split("<tool_call>").skip(1) {
        let Some(a) = blk.find("<function=") else { continue };
        let rest = &blk[a + 10..];
        let Some(e) = rest.find('>') else { continue };
        let name = rest[..e].trim().to_string();
        let mut args = serde_json::Map::new();
        for p in rest[e..].split("<parameter=").skip(1) {
            let Some(e2) = p.find('>') else { continue };
            let key = p[..e2].to_string();
            let body = &p[e2 + 1..];
            let val = body.split("</parameter>").next().unwrap_or("");
            let val = val.strip_prefix('\n').unwrap_or(val); let val = val.strip_suffix('\n').unwrap_or(val);
            args.insert(key, serde_json::from_str::<Value>(val).ok().filter(|v| !v.is_string()).unwrap_or_else(|| json!(val)));
        }
        v.push(ToolCall { name, arguments: Value::Object(args).to_string() });
    }
    v
}

impl Backend for Local {
    fn model_id(&self) -> String { self.name.clone() }
    fn n_ctx(&self) -> usize { self.n_ctx }
    fn n_params(&self) -> u64 { self.n_params }
    fn supports_tools(&self) -> bool { self.tmpl.contains("tool_call") }
    fn generate(&self, req: &GenRequest, emit: &mut dyn FnMut(Delta) -> bool) -> Usage {
        let t0 = std::time::Instant::now();
        let msgs = fix_messages(&req.messages);
        let prompt = match template::render(&self.tmpl, &msgs, req.tools.as_ref(), true, &self.bos, &self.eos) {
            Ok(p) => p, Err(e) => { eprintln!("template error: {e}"); String::new() }
        };
        let ids = self.tok.encode(&prompt);
        let mut g = self.inner.lock().unwrap();
        let g = &mut *g;
        let max_pos = self.n_ctx.saturating_sub(1);
        let ids = if ids.len() > max_pos { ids[ids.len() - max_pos..].to_vec() } else { ids };
        // Recurrent state cannot rewind: reuse only when the cached history is a prefix of the new prompt.
        let cached = if !g.hist.is_empty() && ids.starts_with(&g.hist) && g.hist.len() < ids.len() { g.hist.len() } else { g.gpu.reset(); g.hist.clear(); 0 };
        let mut logits = g.gpu.prefill(&ids[cached..]);
        g.hist = ids.clone();
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let start_in_think = prompt.trim_end().ends_with("<think>");
        let mut sp = Splitter { buf: String::new(), mode: if start_in_think && req.enable_thinking { 1 } else { 0 }, tool: String::new() };
        if let Some(s) = req.sampling.seed { g.rng = s.wrapping_mul(0x9E3779B97F4A7C15) | 1; }
        let mut bytes: Vec<u8> = vec![];
        let mut n = 0; let mut alive = true; let mut reason = "length";
        let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| self.tok.tokens.iter().position(|x| x == t).map(|i| i as u32)).collect();
        let budget = req.max_tokens.min(max_pos.saturating_sub(ids.len()));
        while n < budget {
            let t = sample(&mut logits, &req.sampling, &g.hist, &mut g.rng);
            if eos.contains(&t) { reason = "stop"; break; }
            n += 1;
            bytes.extend(self.tok.decode_bytes(&[t]));
            let valid = match std::str::from_utf8(&bytes) { Ok(_) => bytes.len(), Err(e) => e.valid_up_to() };
            if valid > 0 {
                let text = String::from_utf8_lossy(&bytes[..valid]).to_string(); bytes.drain(..valid);
                let mut out = vec![]; sp.push(&text, &mut out);
                for d in out { if !emit(d) { alive = false; } }
                if !alive { break; }
            }
            if n >= budget { break; }
            logits = g.gpu.forward(t); g.hist.push(t);
        }
        // hist holds exactly the tokens fed to the model.
        let mut out = vec![]; sp.finish(&mut out);
        for d in out { if alive { alive = emit(d); } }
        let tool_calls = parse_tools(&sp.tool);
        if !tool_calls.is_empty() && reason == "stop" { reason = "tool_calls"; }
        Usage { prompt_tokens: ids.len(), completion_tokens: n, cached_tokens: cached, prefill_ms, tool_calls, finish_reason: reason }
    }
}
