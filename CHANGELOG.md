# Changelog

## v0.1.0

### Speed
- Dense prefill: gate and up projections fused into one GEMM/matvec with `swiglu_gu` (commit 3cf3631). MiniCPM5 4K prefill: no measurable change (within 1%).
- Dense prefill attention v2 (`dn_flash_128b`, commit 459b18d): 64-key tiles, Q in registers, `ldmatrix.trans` for V. MiniCPM5 Q4_K_M 4K prefill 1808 → 1972 tok/s (+9%, r=3, prompts within 8 tokens). Decode unchanged. `LE_DN_FLASH2=0` selects the old kernel.
- Sage-A1.5 (MoE, `-ncmoe 99`), same day, r=3, ctx 20000, prompts within 8 tokens:

| Build | pp4096 tok/s | pp16384 tok/s | tg128 tok/s |
|---|---|---|---|
| llama.cpp b12182 | 275 | 227 | 21.1 |
| Last Engine 0.1.0 | 826 | 678 | ~34 |

llama.cpp single runs earlier the same day were higher (326 / 314 / 23.5); against those, Last Engine is ~2.5x / ~2.2x / ~1.45x.

### Known gaps
- MiniCPM5 dense still trails llama.cpp b12182 on prefill (about 1972 vs 3376 tok/s at 4K) and decode (about 80 vs 102). The Q4_K GEMM and dense attention are the remaining targets.

### Tooling
- `lastengine probe` prints hardware and backend JSON.
- `LE_PROF=1` per-kernel prefill timing.
- `LE_MMQR=1` opt-in 128-row GEMM tile (no default change).

### Platform
- Linux CUDA build tested on RTX 3050 Laptop (sm_86).
- Windows build is compiled in CI but unverified on a GPU.
