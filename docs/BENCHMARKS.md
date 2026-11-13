# Matched benchmark vs llama.cpp (RTX 3050 Laptop 4 GB, 2026-11-06)

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB, sm_86), 12 CPU threads, driver 615.78.08.
llama.cpp: version 0.8.0-dev (build 1064, commit 0893ec9), `llama-bench -p N -n 128 -ngl 99 -fa 1 -r 2` (`-ncmoe 99` for the MoE model: all experts on CPU).
LastEngine: this repo, `lastengine serve --ctx N+400`, f16 KV, one chat request with an N-token prompt (prefix cache cold), 128 generated tokens, temperature 0, thinking off; numbers from the server's `timings`.
Tool: `tools/bench_vs_llamacpp.py`. One run per cell for LastEngine, mean of 2 for llama.cpp. LastEngine streams experts from RAM through a VRAM slot cache on the MoE model; `-ncmoe 99` is the closest llama.cpp setting that fits the card.

## Sage-A1.5 APEX MiniPlus (35B-A3B MoE, 14 GiB)
| prompt | LastEngine prefill | llama.cpp prefill | LastEngine decode | llama.cpp decode |
|---|---|---|---|---|
| 512 | 400 tok/s | 335 | 32.8 tok/s | 26.9 |
| 4096 | 817 | 336 | 31.7 | 27.4 |
| 16384 | 639 | 321 | 29.3 | 26.0 |

## MiniCPM5-2B Q4_K_M (dense, fits in VRAM) — LastEngine is SLOWER here
| prompt | LastEngine prefill | llama.cpp prefill | LastEngine decode | llama.cpp decode |
|---|---|---|---|---|
| 512 | 995 tok/s | 3472 | 70.4 tok/s | 95.0 |
| 4096 | 271 | 3082 | 29.8 | 94.6 |
| 16384 | not run (attention cost grows quadratically) | | | |

Cause: the dense attention kernels (`dn_attn` in src/cuda/dense.cu) are a simple one-block-per-head loop with a warp reduction per key; no flash-style tiling or split-K. Fixing this is the next dense-model task.

## MiniCPM5-2B Q4_K_M after flash prefill + split-K decode (2026-11-07, LastEngine only)
llama.cpp could not be re-run: its build directory (~/AI/engines/llama-mainline) is no longer on disk, so no matched llama.cpp numbers exist for this section.
| prompt | prefill tok/s | decode tok/s (128 tokens) |
|---|---|---|
| ~4K (4071) | 1540 | 71.1 |
| ~16K (16359) | 1071 | 44.5 |
Earlier llama.cpp numbers in this file (prefill 3400/2890 at 512/4K, decode ~90) are from the earlier session and should be re-measured before any claim.

## Dense split-K tuning (2026-11-07, LastEngine only, MiniCPM5 Q4_K_M, 128 generated tokens)
Default split count changed from span/64 to a fixed 16 (`LE_DN_SP` still overrides).
| split count | 4K decode tok/s | 16K decode tok/s |
|---|---|---|
| 4 | 74.6 | |
| 8 | 75.1 | |
| 16 (new default) | 76.2 | 49.4 |
| 32 | | 48.2 |
| 64 (old default) | 71.3 | 46.5 |
| 128 | 64.7 | |
Prefill unchanged (~1530 tok/s at 4K, ~1060 at 16K). dense-cmp 15/15 argmax after the change.
Profile at 4K (nsys, decode-heavy run): mmq_12 46% of GPU time (prefill GEMMs), dn_flash_128 13%, matvec_12 11%, mmq_14 9%, dn_attn_split 6% (78 us per call, 2268 launches incl. prefill), matvec_14 5%.

## llama.cpp b12182 (2026-11-07, matched flags, single run each)
Binaries: ~/AI/engines/llama-prebuilt-b12182/full/llama-b12182/ (llama-bench, llama-server; version 0.9.0-dev build 12182 commit fd27b2034; CUDA 12.8 cudart and cublas libs from the cudart archive copied into the same folder; run with LD_LIBRARY_PATH set to that folder).
Command: llama-bench -m MODEL -p N -n 128 -ngl 99 -fa 1 -r 1 (MoE adds -ncmoe 99).
Hardware: RTX 3050 Laptop 4 GB, driver 615.78.08.

| model | pp512 | pp4096 | pp16384 | tg128 |
|---|---|---|---|---|
| MiniCPM5 Q4_K_M (dense) | 3796 | 3376 | 2374 | 102.3 (pp16384 run: 102.9) |
| Sage-A1.5 APEX MiniPlus, -ncmoe 99 | | 342 | 335 | 25.2 |

LastEngine on MiniCPM5 Q4_K_M, same card (serve, one chat request, 128 generated tokens, temperature 0; not llama-bench, so prompt text differs): 4K prefill 1541, decode 75.7 tok/s; 16K prefill 1060, decode 49.4 tok/s.
Sage MoE LastEngine numbers are from the earlier run against llama.cpp mainline (not re-run here): 4K prefill 817, decode 31.7 tok/s.
Caveat: one repetition per cell, so no error bars; the llama-bench rows show ±0.00 because only one run was done.

## mmq register prefetch (2026-11-07, MiniCPM5 Q4_K_M, serve, one chat request, 128 generated tokens, temperature 0)
Change: mmq_impl loads the next 4-unit step's weights and activations into registers before the current step's mma (src/cuda/kernels.cu, mmq_impl).
| prompt | before prefill / decode | after prefill / decode |
|---|---|---|
| 4K (4071 tok) | 1541 / 75.7 | 1582 / 77.1 |
| 16K (16359 tok) | 1060 / 49.4 | 1094 / 50.1 |
dense-cmp 15/15 argmax (MiniCPM5, 15 tokens, mean |dlogprob| 0.0032). Gain is about 3%; llama.cpp b12182 is 3376 / 2374 prefill and ~102 decode on the same card.

## Split-K env read cached (2026-11-07, MiniCPM5 Q4_K_M, serve --ctx 17000, 128 tokens, temp 0)
Change: LE_DN_SPLIT_MIN / LE_DN_SPLIT / LE_DN_SP are read once (dn_split_env, src/gpu/dense.rs) instead of per layer per token. Expected gain: host time only.
Measured: no measurable change. Three repeat runs gave 4K decode 74.8 / 72.2 / 73.3 tok/s and 4K prefill 1558 on the first (cold) run; a single run earlier in the session gave 77.1, so run-to-run spread is about ±3%.
dense-cmp 15/15 argmax (MiniCPM5, 15 tokens, mean |dlogprob| 0.0032).

## Device-side pos/nk (2026-11-07, MiniCPM5 Q4_K_M, serve --ctx 17000, 128 tokens, temp 0)
Change: decode position and KV length are uploaded once per token into a device buffer (DenseGpu.dpos); dn_prep, dn_cvt_half_d, dn_attn and dn_attn_split read them there, so kernel launch args no longer change per token. Prerequisite for CUDA graph capture; no speed change by itself.
Measured: 4K prefill 1572, decode 73.1 and 76.6 (two runs); 16K prefill 1088, decode 49.7. Same as before within run-to-run noise (4K decode 72-77).
dense-cmp 15/15 argmax: MiniCPM5 default (mean |dlogprob| 0.0032), LFM3 (0.0018), MiniCPM5 with LE_DN_SPLIT_MIN=4 (0.0045).

## Decode CUDA graph (2026-11-07, MiniCPM5 Q4_K_M, serve --ctx 17000, 128 tokens, temp 0)
Change: the decode layer loop (fn layers, src/gpu/dense.rs) is captured once per split-K pattern into a CUDA graph and replayed each token (cudarc begin_capture/end_capture, thread-local mode). The dense engine stream is now a non-blocking stream (the legacy default stream cannot capture). LE_DN_GRAPH=0 turns graphs off.
Same session, back to back:
- graph on: 4K prefill 1583, decode 80.4 and 80.5; 16K prefill 1092, decode 51.8
- graph off (LE_DN_GRAPH=0): 4K prefill 1578, decode 76.8 and 76.6; 16K prefill 1089, decode 50.1
So graphs give about +5% decode at 4K and +3% at 16K. Prefill unchanged.
dense-cmp 15/15 argmax: MiniCPM5 default (mean |dlogprob| 0.0032), LFM3 (0.0018), MiniCPM5 with LE_DN_SPLIT_MIN=4 forced split (0.0045).
llama.cpp b12182 (same card, same flags as above): 4K pp 3376, 16K pp 2374, tg ~102. Still behind.

## Dense prefill: 128-token MMQ tile (2026-11-07, MiniCPM5 Q4_K_M, serve --ctx 17000, 128 tokens, temp 0)
Change: mmq128_impl (src/cuda/kernels.cu) stages 128 tokens per block instead of 64, so each weight tile is dequantised half as often. Selected by LE_MMQ128 (default on; LE_MMQ128=0 gives the 64-token kernel). Dense path only; MoE unchanged.
Same session, back to back:
- LE_MMQ128=0: 4K prefill 1589, decode 80.7; 16K prefill 1093, decode 51.8
- LE_MMQ128=1 (default): 4K prefill 1752, decode 80.5; 16K prefill 1167, decode 51.7
So prefill +10% at 4K and +7% at 16K. Decode unchanged.
dense-cmp 15/15 argmax: MiniCPM5 (mean |dlogprob| 0.0032), LFM3 (0.0018).
llama.cpp b12182 (same card, same flags): 4K pp 3376, 16K pp 2374. Still behind, about 1.9x at 4K.

## MMQ128 launch bounds (2026-11-07, MiniCPM5 Q4_K_M, serve --ctx 17000, 128 tokens, temp 0, same session)
Change: `__launch_bounds__(256, 2)` on the mmq128 kernels (src/cuda/kernels.cu, MMQ128 macro). Lets the compiler use up to 2 blocks per SM.
Two binaries, run back to back with the same script (/tmp/claude-1000/bench_bin.sh):
- base (813bf9f): 4K prefill 1770 (first run), decode 80.8; 16K prefill 1174, decode 51.9
- launch bounds (256,2): 4K prefill 1854 (first run), decode 80.7; 16K prefill 1217, decode 51.8
Prefill +5% at 4K, +4% at 16K; decode unchanged. The second 4K prefill run in each build (about 70 tok/s) is a prefix-cache hit on the repeated prompt, not prefill; use the first run.
dense-cmp 15/15 argmax: MiniCPM5 (mean |dlogprob| 0.0032, max 0.0169), LFM3 (0.0018, max 0.0091).
llama.cpp b12182 (same card): 4K pp 3376, 16K pp 2374, tg ~102.

## Sage-A1.5 APEX MiniPlus, same-session rerun (2026-11-11, RTX 3050 Laptop 4 GB)
Model: ~/AI/Models/sage-a15/Sage-A1.5.APEX-I-MiniPlus-V2.1.gguf (14.17 GiB as read by llama-bench; llama-bench labels it qwen35moe 35B.A3B Q4_0).
Lock: ~/AI/.machine-busy held for the run. llama-swap had no models loaded (/running empty); the /unload call returned 405, so no unload was needed.

| engine | version | flags | pp4096 tok/s | pp16384 tok/s | decode tok/s |
|---|---|---|---|---|---|
| llama.cpp | b12182 (fd27b2034) | llama-bench -ngl 99 -fa 1 -ncmoe 99 -p 4096,16384 -n 128 -r 1 | 326.0 ± 0 | 313.8 ± 0 | tg128 23.5 |
| LastEngine | 40969d3 | serve --ctx 17000 (auto-tune: kv q8, chunk 2048, 992 expert slots, 12480 MiB pinned expert store) | 763.3 (prompt_n 3043) | 748.3 (prompt_n 12110) | 31.4–31.8 (4K), 29.0 (16K) |

Caveats (read before quoting):
- Token counts do not match. llama-bench processes exactly 4096 / 16384 tokens. LastEngine's prompts were built from characters and tokenized to 3043 / 12110, so its numbers are a rate on a shorter prompt. Not a matched-token comparison.
- One run each (llama-bench -r 1; LastEngine 2 requests at 4K, 1 at 16K). No error bars for llama.cpp.
- LastEngine's second 4K request had prompt_n 7 (prefix cache hit); only the first 4K request is a prefill measurement.
- The llama.cpp decode column is from llama-bench tg128 with no prompt context; LastEngine decode is measured after its prompt.
- Earlier LastEngine Sage numbers (817 prefill / 31.7 decode, not re-run) are superseded by this run for the 4K row.

Result: LastEngine beats llama.cpp b12182 on this MoE model in this session: about 2.3x on prefill and about 1.35x on decode. The MoE win holds, with the token-count caveat above.

## Sage fixed-token-count repeat (2026-11-11, LastEngine 8355d33, llama.cpp b12182 fd27b2034)
Same flags as above; llama-bench -r 3. LastEngine prompts calibrated to prompt_n 4096 and 16384 (chat template adds a few tokens; a unique random prefix per run defeats the prefix cache).

| engine | prompt | prefill tok/s | decode tok/s |
|---|---|---|---|
| llama.cpp b12182 | pp4096 | 151.35 ± 24.96 | tg128 16.45 ± 11.89 |
| llama.cpp b12182 | pp16384 | 202.30 ± 139.64 | (same run) |
| LastEngine | 4096 (prompt_n 4095/4147/4141) | 875.7 / 815.4 / 808.7 | 34.5 / 36.1 / 35.4 |
| LastEngine | 16384 (prompt_n 16320/16301/16238) | 663.8 / 662.4 / 673.6 | 26.7 / 29.0 / 0.0 (third value is a timing artifact, not a real 0) |

Caveats:
- The llama.cpp r=3 run is unusable: the spread (±25 to ±140 tok/s) is far above the single-run values (326 / 314 / 23.5) from earlier the same day. GPU was idle at 2 MiB before the run and at 0 % after. Cause not found; treat the llama.cpp numbers from this run as invalid and re-run before quoting.
- LastEngine prompts match the target token counts within about 1 %, so prefill is comparable in size. Its three 16K decode values also show run-to-run variance.
- LastEngine's 4K prefill is higher here than in the earlier single run (763 with 3043 tokens), probably because the calibrated prompts are longer and the earlier run was shorter; not a like-for-like comparison with that row.
