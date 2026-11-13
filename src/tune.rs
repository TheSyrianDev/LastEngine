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

/// Per-device profile (docs/autotune-plan.md, schema 1). Written by `tune`, read by `serve` (wiring pending).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Profile {
    pub schema: u32,
    pub engine: String,
    pub device: String,
    pub model_sha: String,
    pub ctx: usize,
    pub settings: BTreeMap<String, String>,
    pub prefill_tps: f32,
    pub decode_tps: f32,
    pub tuned_at: String,
}

pub const PROFILE_SCHEMA: u32 = 1;

/// Path: $XDG_CACHE_HOME/lastengine/profiles/<device>-<model_sha>-<ctx>.json (device sanitised to [a-z0-9-]).
pub fn profile_path(device: &str, model_sha: &str, ctx: usize) -> std::path::PathBuf {
    let dev: String = device.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let base = std::env::var("XDG_CACHE_HOME").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache"));
    base.join("lastengine").join("profiles").join(format!("{dev}-{model_sha}-{ctx}.json"))
}

pub fn write_profile(p: &std::path::Path, prof: &Profile) -> std::io::Result<()> {
    if let Some(d) = p.parent() { std::fs::create_dir_all(d)?; }
    std::fs::write(p, serde_json::to_string_pretty(prof).unwrap())
}

/// Load and validate. Returns None (with a reason) if the file is missing, unparsable, or from another schema/engine/model.
pub fn read_profile(p: &std::path::Path, engine: &str, model_sha: &str) -> Result<Profile, String> {
    let t = std::fs::read_to_string(p).map_err(|e| format!("profile missing: {e}"))?;
    let prof: Profile = serde_json::from_str(&t).map_err(|e| format!("profile unparsable: {e}"))?;
    if prof.schema != PROFILE_SCHEMA { return Err(format!("profile schema {} != {}", prof.schema, PROFILE_SCHEMA)); }
    if prof.engine != engine { return Err(format!("profile engine {} != {}", prof.engine, engine)); }
    if prof.model_sha != model_sha { return Err("profile model_sha mismatch".into()); }
    Ok(prof)
}

#[cfg(test)]
mod profile_tests {
    use super::*;
    fn sample() -> Profile {
        let mut s = BTreeMap::new(); s.insert("chunk".into(), "512".into());
        Profile { schema: 1, engine: "0.1.0".into(), device: "RTX 3050".into(), model_sha: "abc123".into(), ctx: 8192, settings: s, prefill_tps: 2000.0, decode_tps: 93.0, tuned_at: "2026-11-13T00:00:00Z".into() }
    }
    #[test]
    fn roundtrip_and_validate() {
        let d = std::env::temp_dir().join(format!("le-prof-{}", std::process::id()));
        let p = d.join("p.json");
        write_profile(&p, &sample()).unwrap();
        assert_eq!(read_profile(&p, "0.1.0", "abc123").unwrap(), sample());
        let _ = std::fs::remove_dir_all(&d);
    }
    #[test]
    fn rejects_mismatch_and_missing() {
        let d = std::env::temp_dir().join(format!("le-prof2-{}", std::process::id()));
        let p = d.join("p.json");
        write_profile(&p, &sample()).unwrap();
        assert!(read_profile(&p, "0.2.0", "abc123").unwrap_err().contains("engine"));
        assert!(read_profile(&p, "0.1.0", "zzz").unwrap_err().contains("model_sha"));
        let _ = std::fs::remove_dir_all(&d);
        assert!(read_profile(&p, "0.1.0", "abc123").unwrap_err().contains("missing"));
    }
    #[test]
    fn path_sanitises_device() {
        let p = profile_path("NVIDIA GeForce RTX 3050 Laptop", "abc", 8192);
        assert!(p.to_string_lossy().ends_with("nvidia-geforce-rtx-3050-laptop-abc-8192.json"));
    }
}

/// Cheap model identity: FNV-1a over the first 1 MiB of the file plus its size. Not cryptographic; only for cache keys.
pub fn model_sha(path: &str) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let mut buf = vec![0u8; 1 << 20];
    let n = f.read(&mut buf)?;
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in &buf[..n] { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    Ok(format!("{:016x}{:x}", h, len))
}

/// Serve-side load: returns the applied settings (env vars that were unset before). Never overrides explicit env.
/// LE_PROFILE=off disables. Any failure is reported as a reason and serving proceeds with defaults.
pub fn apply_profile_env(model: &str, ctx: usize) -> Result<Option<Profile>, String> {
    if std::env::var("LE_PROFILE").map_or(false, |v| v == "off") { let _ = LOADED.set(("".into(), None, "LE_PROFILE=off".into())); return Ok(None); }
    let sha = model_sha(model).map_err(|e| format!("model sha: {e}"))?;
    let dev = gpu_name();
    let p = profile_path(&dev, &sha, ctx);
    let prof = match read_profile(&p, env!("CARGO_PKG_VERSION"), &sha) {
        Ok(pr) => pr,
        Err(why) => { eprintln!("profile: not applied ({why})"); let _ = LOADED.set((p.display().to_string(), None, why)); return Ok(None); }
    };
    for (k, v) in &prof.settings {
        if std::env::var_os(k).is_none() { std::env::set_var(k, v); }
    }
    eprintln!("profile: applied {} setting(s) from {}", prof.settings.len(), p.display());
    let _ = LOADED.set((p.display().to_string(), Some(prof.clone()), "applied".into()));
    Ok(Some(prof))
}

#[cfg(test)]
mod apply_tests {
    use super::*;
    #[test]
    fn model_sha_changes_with_content() {
        let d = std::env::temp_dir().join(format!("le-sha-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("m.bin");
        std::fs::write(&f, b"abc").unwrap();
        let a = model_sha(f.to_str().unwrap()).unwrap();
        std::fs::write(&f, b"abd").unwrap();
        let b = model_sha(f.to_str().unwrap()).unwrap();
        assert_ne!(a, b);
        let _ = std::fs::remove_dir_all(&d);
    }
}

/// What serve loaded at startup (for GET /v1/profile). Set once by apply_profile_env.
pub static LOADED: std::sync::OnceLock<(String, Option<Profile>, String)> = std::sync::OnceLock::new();

/// Status JSON for GET /v1/profile: path, whether applied, and the profile contents if any.
pub fn status_json() -> serde_json::Value {
    match LOADED.get() {
        Some((path, prof, why)) => serde_json::json!({"path": path, "applied": prof.is_some(), "reason": why, "profile": prof}),
        None => serde_json::json!({"path": null, "applied": false, "reason": "serve did not check a profile", "profile": null}),
    }
}
