//! Hardware probe: `lastengine probe` prints one JSON object. Fields are stable; unknown values are null.
use serde_json::{json, Value};
use std::process::Command;

fn smi(q: &str) -> Option<Vec<String>> {
    let o = Command::new("nvidia-smi").args([format!("--query-gpu={q}"), "--format=csv,noheader,nounits".into()]).output().ok()?;
    if !o.status.success() { return None; }
    Some(String::from_utf8_lossy(&o.stdout).lines().next()?.split(',').map(|x| x.trim().to_string()).collect())
}
fn meminfo(key: &str) -> Option<u64> {
    let t = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = t.lines().find(|l| l.starts_with(key))?;
    Some(l.split_whitespace().nth(1)?.parse::<u64>().ok()? / 1024)
}

pub fn probe_json() -> Value {
    let gpu = smi("name,memory.total,memory.free,compute_cap,driver_version");
    let mut backends = vec![];
    let g = match &gpu {
        Some(v) if v.len() == 5 => {
            let cc = v[3].clone();
            let ok = cc.split('.').next().and_then(|m| m.parse::<u32>().ok()).map_or(false, |m| m >= 8);
            if ok { backends.push("cuda"); }
            json!({"name": v[0], "vram_total_mib": v[1].parse::<u64>().ok(), "vram_free_mib": v[2].parse::<u64>().ok(), "compute_capability": cc, "cuda_driver_version": v[4], "supported": ok})
        }
        _ => Value::Null,
    };
    json!({
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "cpu_threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "ram_total_mib": meminfo("MemTotal:"), "ram_available_mib": meminfo("MemAvailable:"),
        "gpu": g, "backends": backends, "engine_version": env!("CARGO_PKG_VERSION"),
    })
}
