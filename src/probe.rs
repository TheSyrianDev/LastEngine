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
    build_probe(smi("name,memory.total,memory.free,compute_cap,driver_version"), meminfo("MemTotal:"), meminfo("MemAvailable:"), std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

/// Pure JSON builder so the field mapping can be tested without hardware.
pub fn build_probe(gpu: Option<Vec<String>>, ram_total_mib: Option<u64>, ram_available_mib: Option<u64>, cpu_threads: usize) -> Value {
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
        "cpu_threads": cpu_threads,
        "ram_total_mib": ram_total_mib, "ram_available_mib": ram_available_mib,
        "gpu": g, "backends": backends, "engine_version": env!("CARGO_PKG_VERSION"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(xs: &[&str]) -> Option<Vec<String>> { Some(xs.iter().map(|x| x.to_string()).collect()) }

    #[test]
    fn ampere_gpu_reports_cuda() {
        let j = build_probe(v(&["NVIDIA RTX 3050", "4096", "3800", "8.6", "615.78"]), Some(23750), Some(15000), 12);
        assert_eq!(j["backends"][0], "cuda");
        assert_eq!(j["gpu"]["supported"], true);
        assert_eq!(j["gpu"]["vram_total_mib"], 4096);
        assert_eq!(j["gpu"]["vram_free_mib"], 3800);
        assert_eq!(j["ram_total_mib"], 23750);
        assert_eq!(j["cpu_threads"], 12);
    }

    #[test]
    fn pre_ampere_gpu_has_no_backend() {
        let j = build_probe(v(&["GTX 1060", "6144", "6000", "6.1", "550.0"]), None, None, 4);
        assert!(j["backends"].as_array().unwrap().is_empty());
        assert_eq!(j["gpu"]["supported"], false);
    }

    #[test]
    fn no_gpu_and_no_ram_are_null_not_missing() {
        let j = build_probe(None, None, None, 2);
        assert!(j["gpu"].is_null());
        assert!(j["ram_total_mib"].is_null());
        assert!(j["backends"].as_array().unwrap().is_empty());
        assert!(j.get("engine_version").is_some());
    }

    #[test]
    fn malformed_smi_row_is_treated_as_no_gpu() {
        let j = build_probe(v(&["only", "three", "fields"]), None, None, 1);
        assert!(j["gpu"].is_null());
    }
}
