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
Binaries: /home/omar/AI/engines/llama-prebuilt-b12182/full/llama-b12182/ (llama-bench, llama-server; version 0.9.0-dev build 12182 commit fd27b2034; CUDA 12.8 cudart and cublas libs from the cudart archive copied into the same folder; run with LD_LIBRARY_PATH set to that folder).
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
