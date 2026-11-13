# Auto-tune plan

Status: plan, 2026-11-13. CPU-side parts first; GPU tuning waits for the machine lock.

## Current state
- `lastengine tune --model M --ctx N` (src/tune.rs) runs short prefill+decode trials in child
  processes over kv/chunk settings, caches the winner in `$XDG_CACHE_HOME/lastengine/tune.json`
  keyed by (gpu name, model shape, ctx).
- `lastengine probe` prints hardware JSON (GPU name, CC, driver, VRAM, CPU threads).
- `serve` does not yet read the cache automatically.

## Goal
Per-device profile written once, loaded by `serve`, exposed over the API so Last AI can show
"tuned for your GPU" and trigger a re-tune. Reference: magnitudedev/magnitude (self-tunes per
device; ideas only, keep attribution if any code is ported).

## Profile JSON (schema v1)
```json
{"schema":1,"engine":"0.1.0","device":{"name":"...","cc":"8.6","vram_mib":4096,"driver":"..."},
 "model":{"file":"...","sha_prefix":"...","n_layer":42,"n_kv_head":2,"head_dim":128},
 "ctx":8192,"settings":{"kv":"f16","chunk":512,"slots":1,"mmq_tile":"128r","dn_gqa":true,"dn_sp":32,"batch_rows":6},
 "measured":{"prefill_tps":0,"decode_tps":0,"trials":0},"tuned_at":"<iso8601>"}
```
Stored at `$XDG_CACHE_HOME/lastengine/profiles/<device>-<model-sha>-<ctx>.json`.

## Grid (small, measured)
- kv: f16 (only, until K8V4 lands)
- chunk: 256, 512, 1024
- mmq tile: LE_MMQR 0/1
- dn_gqa/dn_sp: off, on-SP32
- slots: 1 (batch routes stay off until gate passes)
Each trial: warmup + 2 timed runs, median; abort on OOM and record it.

## Steps
1. CPU: profile read/write + schema check + `serve` auto-load (env override LE_PROFILE=off). No GPU.
2. CPU: `tune --dry-run` prints grid size and VRAM estimate. No GPU.
3. GPU (lock): run tune on MiniCPM5-2B-Q4_K_M, commit the profile JSON under bench/profiles/.
4. API: GET /v1/profile (device, settings, measured, tuned_at); POST /v1/tune (async, returns job id; GET job status). Coordinate shape with web-view-desktop-pivot.
5. Idle VRAM free: flag `--unload-idle SECS` drops weights after idle, reloads on next request. Off by default.

## Gates
- tune never changes default output; profile only picks speed settings that pass dense_cmp gate.
- A profile whose schema/engine/model sha does not match is ignored, with a log line.
