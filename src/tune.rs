//! Measured auto-tune: run short prefill+decode trials in child processes, cache the winner per (gpu, model shape, ctx).
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::process::Command;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Entry { pub kv: String, pub chunk: usize, pub prefill_tps: f32, pub decode_tps: f32 }

fn path() -> std::path::PathBuf {
    let base = std::env::var("XDG_CACHE_HOME").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache"));
    base.join("lastengine").join("tune.json")
}
fn load() -> BTreeMap<String, Entry> { std::fs::read_to_string(path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default() }
pub fn lookup(key: &str) -> Option<Entry> { load().get(key).cloned() }
pub fn gpu_name() -> String {
    Command::new("nvidia-smi").args(["--query-gpu=name", "--format=csv,noheader"]).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn trial(exe: &str, model: &str, ctx: usize, kv: &str, chunk: usize, prompt: &str) -> Option<(f32, f32, Option<String>)> {
    let o = Command::new(exe).args(["gpu", model, prompt, "1"]).env("LE_CTX", ctx.to_string()).env("LE_KV", kv).env("LE_TMAX", chunk.to_string()).env("LE_NOTUNE", "1").env("LE_N", "48").output().ok()?;
    let out = String::from_utf8_lossy(&o.stdout).to_string(); let err = String::from_utf8_lossy(&o.stderr).to_string();
    let key = err.lines().find_map(|l| l.strip_prefix("tunekey ")).map(|s| s.to_string());
    let pf = out.lines().find(|l| l.starts_with("prefill "))?; let w: Vec<&str> = pf.split_whitespace().collect();
    let (n, secs): (f32, f32) = (w.get(1)?.parse().ok()?, w.get(4)?.trim_end_matches('s').parse().ok()?);
    let dc = out.lines().find(|l| l.starts_with("decode "))?; let i = dc.find("= ")?; let dtps: f32 = dc[i + 2..].split_whitespace().next()?.parse().ok()?;
    Some((n / secs, dtps, key))
}

pub fn run(model: &str, ctx: usize) {
    let exe = std::env::current_exe().expect("exe").to_string_lossy().to_string();
    let words = "the cat sat on a mat and the dog ran to the red house by the river while clouds moved over stone hills";
    let prompt = std::iter::repeat(words).take(110).collect::<Vec<_>>().join(" "); // ~2200 tokens
    let kvs: &[&str] = if ctx <= 8192 { &["f16", "q8"] } else { &["q8", "tcq3"] };
    let (mut best, mut key): (Option<(f64, Entry)>, Option<String>) = (None, None);
    for kv in kvs { for chunk in [1024usize, 2048, 4096] {
        match trial(&exe, model, ctx, kv, chunk, &prompt) {
            Some((pp, dc, k)) => {
                if key.is_none() { key = k; }
                // wall time of a typical request: 2000-token prompt + 256 generated tokens
                let t = 2000.0 / pp as f64 + 256.0 / dc as f64;
                eprintln!("tune: kv {kv:5} chunk {chunk:4}: prefill {pp:6.1} tok/s  decode {dc:5.1} tok/s  request {t:.2}s");
                if best.as_ref().map_or(true, |b| t < b.0) { best = Some((t, Entry { kv: kv.to_string(), chunk, prefill_tps: pp, decode_tps: dc })); }
            }
            None => eprintln!("tune: kv {kv} chunk {chunk}: failed (likely out of memory), skipped"),
        }
    } }
    let (Some((_, e)), Some(k)) = (best, key) else { eprintln!("tune: no trial succeeded"); std::process::exit(1) };
    let mut m = load(); m.insert(k.clone(), e.clone());
    let p = path(); std::fs::create_dir_all(p.parent().unwrap()).ok();
    std::fs::write(&p, serde_json::to_string_pretty(&m).unwrap()).expect("write cache");
    println!("{}", serde_json::json!({"key": k, "kv": e.kv, "chunk": e.chunk, "prefill_tps": e.prefill_tps, "decode_tps": e.decode_tps, "cache": p}));
}
