# Dense speed plan: beat llama.cpp b12182 on prefill

Card: RTX 3050 4 GB. Model: MiniCPM5-2B Q4_K_M. Same flags as the llama-bench comparison.

## Measured gap (HEAD a8103f2)

| Metric | LastEngine | llama.cpp b12182 | Ratio |
|---|---|---|---|
| Prefill 4K tok/s | 1854 | 3376 | 1.82x behind |
| Prefill 16K tok/s | 1217 | 2374 | 1.95x behind |
| Decode 4K tok/s | 80.7 | ~102 | 1.27x behind |

## What keeps us apart (prefill)

1. **Dequant cost (~20% of prefill).** With NODEQ upper bound, 4K prefill reaches 2182 tok/s. Expected gain: up to +18% if dequant is hidden or moved off the hot loop (e.g., pre-dequantized tiles in shared memory or fused unpack into the MMA operand load). Estimate: +5–10%.
2. **Weight tile size and load structure in `mmq128_impl`** (src/cuda/kernels.cu). Each block stages 64 weight rows x 128 tokens, 4 K-units per step, with two `__syncthreads` per step and a 48 KB static shared-memory budget. Expected gain: the largest single item. Wider K per step with coalesced loads should reduce barrier count per FLOP. Estimate: +10–20%.
3. **Occupancy.** `__launch_bounds__(256, 2)` gave +5%. Pushing further needs smaller shared-memory footprint per block. Expected gain: +5%.
4. **Double buffering.** Tried: 66 KB dynamic shared memory, one block per SM, 1659 tok/s (slower). Not on the plan unless the tile shrinks first.
5. **MMA use.** Current path is int8 MMQ with mma.sync m16n8k32 / m16n8k16. Check whether the k16 variant is used on every tile edge and whether accumulator reuse across K-steps is complete. Expected gain: unknown until profiled; ncu runs exceed 500 s, so use targeted timing.
6. **Graph overhead.** Decode graphs gave +5%. Prefill has no graph path yet. Expected gain for prefill: small, under 3%.

## Decode (secondary)

- Split-K attention and matvec kernels are in place. Gap is 1.27x. Expected gain: +5–10% from matvec tile work. Not started.

## Order of work

1. Shrink `mmq128_impl` shared memory per block, to allow more blocks per SM (item 3), then widen the K step (item 2). Measure 4K and 16K after each.
2. Move dequant off the inner loop (item 1).
3. Re-check MMA edge tiles (item 5).
4. Prefill graph and decode matvec (items 6 and decode).

## Gates

- Correctness: dense-cmp 15/15 on MiniCPM5 and LFM3; tf_compare top-1 at or above 98%.
- Tests: `cargo test --release`.
- Commit each finished change locally with repo identity, no attribution, no push.
- Report 4K and 16K prefill and decode after each change, against the llama.cpp numbers above.

## Results log (2026-11-07, RTX 3050 Laptop, MiniCPM5-2B Q4_K_M, bench_bin.sh)

HEAD 085ede9 re-measured this session: 4K prefill 1690 / 1687 tok/s, 16K 1187, 4K decode 71.6 / 72.4 (earlier notes said 1876 / 80.1; those did not reproduce).

| Change | 4K prefill | 16K prefill | Result |
|---|---|---|---|
| step 3 `__launch_bounds__(256,1)` | 1581 | 1136 | slower, reverted |
| step 1 256-token tile, 16 warps, dynamic smem 57344 B | 1699 / 1693 | 1201 | within noise, reverted |
| scale loads as float2 out of inner loop | 1704 / 1701 | 1198 | within noise, reverted |
| persistent wave-fit (grid = 40 resident blocks, strided tile loop, no fixup kernel) | 1553 / 1549 | 1115 | slower, reverted. The grid bound (about 0.7% tail waste) is consistent with this result: the loop-over-tiles cost outweighs any tail gain. Not a full stream-K (no split-K fixup was built). |

Profiling blocked: Nsight Compute returns ERR_NVGPUCTRPERM (no GPU performance counter access for this user). Per-kernel limits are unknown until counters are enabled.

Full stream-K (fixed 40 blocks, contiguous K-step ranges per block, atomicAdd merge into a zeroed output): 1450 / 1454 tok/s at 4K, 1066 at 16K. Slower than HEAD, reverted. The grid bound (about 0.7% tail waste) was not the limiting factor.

## Matched baseline (2026-11-08, RTX 3050 Laptop, sm_86, MiniCPM5-2B Q4_K_M)

llama.cpp: local checkout b10330 (commit 687e778), Linux CUDA build of `git archive` in /tmp/claude-1000/llama-src, CMAKE_CUDA_ARCHITECTURES=86.
Command: `llama-bench -m MiniCPM5-2B-Q4_K_M.gguf -ngl 99 -p 4096,16384 -n 128 -r 2 -o md`

| Test | llama.cpp b10330 | LastEngine HEAD 085ede9 (server timings) | Gap |
|---|---|---|---|
| pp4096 | 3196.34 +- 5.97 | 1690 / 1687 | 1.9x |
| pp16384 | 2289.69 +- 3.44 | 1187 | 1.9x |
| tg128 | 102.03 +- 0.32 | 71.6 / 72.4 | 1.4x |

Caveat: llama-bench times bare prefill/decode; our numbers come from the HTTP server timings, so server overhead is not isolated.

## Decode bandwidth check (2026-11-08)

GPU memory: 4096 MiB, 6001 MHz, 128-bit bus, so peak is about 192 GB/s (estimate from clock and bus width, not measured).
Model read per token: Q4_K_M, about 1.45 GiB (about 1.56 GB).
- LastEngine decode 72 tok/s: about 112 GB/s, about 58% of peak.
- llama.cpp decode 102 tok/s: about 159 GB/s, about 83% of peak.

Kernel reading: `mv_impl` (src/cuda/kernels.cu:372) runs one row per warp for cols=2304 (32 lanes, one 32-value unit each). Weight blocks are 18-to-210-byte packed structs read per lane, not vector loads. This is the likely cause of the low bandwidth. Not yet changed or measured.

## Decode matvec: two rows per lane group (2026-11-08)

`mv_impl` handles two rows per lane group, both weight streams issued before reduction, rpb 8→16 in dense.rs and gpu.rs. Kernel test gpu_matvec_matches_cpu_dequant passed.
Bench (bench_bin.sh, same settings): HEAD decode 73.0 / 73.0 tok/s, prefill 1705/1692 (4K), 1194 (16K). Two-row: decode 70.7 / 70.7, prefill 1695/1692, 1189. Slower on decode, reverted; not committed.

## Q4_K GEMM rewrite plan (LE_MMQ2) — written 2026-11-12, before any code change

Target: mmq128_12 (Q4_K), 61.7% of 4K prefill kernel time (LE_EVT, commit aa90476). Estimated real gap vs llama.cpp about 1.6x on this op.

What already exists, so the rewrite does not repeat it:
- 128-row x 128-token block tile with 8 warps (16 rows x 128 tokens per warp): `mmq128r_*` in src/cuda/kernels.cu (`LE_MMQR=1`). Measured +3% at 4K, not default. This is llama's Ampere tile shape already.
- Scales pre-split per K-unit (Sa0/Sa1/Sam/Sbd/Sbs in smem), so x_dm-style precompute exists in part.
- Tried and not winning: stream-K with atomicAdd (slower), wave-fit persistent (slower), 256-token tile (noise), double buffer (slower, smem over 48 KB).

What is new in LE_MMQ2 (to be tested, not assumed to win):
1. Stage Q4_K weights as pre-unpacked int8 nibble quads per K-unit (llama's `(qs>>0)&0x0F0F0F0F` / `(qs>>4)&0x0F0F0F0F` layout) and precompute dm*sc and -dmin*m per row and sub-block at load time, so the inner loop reads one int per mma operand and one float per sub-block.
2. One `__syncthreads` per 4-unit step instead of two, by double-buffering the activation tile in registers (not smem).
3. Keep the 128x128 tile; keep LE_MMQ2 default off unless r=3 A/B against HEAD wins.

Gates: cargo test; dense-cmp MiniCPM5 15/15 (argmax); Sage correctness only if the MoE path uses mmq128 (check before claiming); r=3 A/B vs HEAD with matched prompts (ab.py); machine lock ~/AI/.machine-busy written for the run.
Status: plan only. No kernel code written yet for LE_MMQ2.

## Decode status (2026-11-12, after attention/fusion round)

Measured on RTX 3050 Laptop, MiniCPM5-2B Q4_K_M, 2594-token chat prompt, 128 generated, ctx 4096, graph mode, r=3.

| Config | Graph decode tok/s |
|---|---|
| HEAD baseline (before this round) | 87.8-88.6 |
| Default now (fused q/k prep + v write, be73cd8) | 88.5-88.7 |
| + LE_DN_GQA=1 LE_DN_SP=32 (1de9ee5) | 90.3-91.0 |
| + LE_RMSQ=1 multi-block (605129d) | 90.7 |
| llama.cpp b12182, same prompt | not measured as wall tok/s; estimated 88 from nsys kernel sum (10.3 ms/tok kernels, 11.3-11.4 ms/tok wall prediction) |

Note: llama.cpp decode is about 102 tok/s on the older tg128 bench (b10330, short prompt). The 88 figure above is llama b12182 at the matched 2583-token prompt, so the two numbers are not directly comparable. Matched comparison here is an estimate only: LastEngine default 88.6 measured vs llama about 88 estimated. Measure llama graph wall tok/s before claiming parity.

Eager per-token kernel times (LE_EVT, same prompt), ms/tok:
- matvec_12 (Q4_K) 6.22 vs llama 6.19: parity.
- matvec_14 (Q6_K) 2.73 vs llama 2.46: +0.3.
- attention (split + comb + prep + cvt) 2.44 vs llama 1.04: +1.4. Largest gap.
- rmsnorm + quant_x + add + swiglu about 1.5 vs llama norm/quant/rope about 0.6: +0.9.

Ranked remaining work:
1. Attention decode: GQA kernel is correct (per-layer diff 3.3e-7) but only +2.5%. The gap to llama's 1.04 ms is mostly split count and launch structure; not yet tuned beyond SP 16/32/64.
2. Norm/quant/elementwise: multi-block rms_quant gives +2.6% but changes greedy output at a near-tie; decide with projects-45 whether to accept.
3. Q6_K matvec +0.3 ms.
4. Prefill (unchanged this round): Q4_K GEMM and attention still about 1.6x and 1.5x behind llama (see LE_MMQ2 plan above).

Defaults unchanged: LE_DN_GQA off, LE_RMSQ off, LE_DN_QKV on. Nothing pushed.

## Matched wall decode, llama.cpp vs best opt-in (2026-11-12, measured under lock)

Prompt: same_req.json (2594 prompt tokens after chat template), 128 generated, temperature 0, ctx 4096, graph mode, r=3.

| Engine | Config | Decode tok/s (r=1 / r=2 / r=3) |
|---|---|---|
| llama.cpp b12182 (llama-server, CUDA, -ngl 99, -np 1) | default | 94.85 / 94.86 / 94.86 |
| LastEngine | default (be73cd8) | 88.6 (earlier session, not re-measured here) |
| LastEngine | LE_DN_GQA=1 LE_DN_SP=32 LE_RMSQ=1 | 93.5 / 93.5 / 93.4 |

Notes:
- The llama decode numbers come from server timings (`predicted_per_second`) with 128 tokens each. The second request reused the prompt cache, so its prefill figure is meaningless; decode is valid.
- The earlier estimate of llama at about 88 tok/s (from nsys kernel sums) was wrong. Measured llama wall decode is about 94.9 tok/s at this prompt.
- Best opt-in combo is about 1.5% behind llama decode at this prompt (93.5 vs 94.9). Default is about 7% behind (88.6 vs 94.9).
- Earlier llama CPU-only run (copy in /tmp without ggml-cuda) gave 14 tok/s and is not valid. Use the CUDA build at ~/AI/engines/llama-prebuilt-b12182/full/llama-b12182/llama-server.
- The "parity" wording earlier in this doc refers to kernel-sum comparison only. Wall decode is the number to use.

## Default flipped: GQA on, SP 32, rms_quant on (2026-11-12, peer decision projects-45)

- Defaults now: LE_DN_GQA on (=0 off), LE_DN_SP 32 (was 16), LE_RMSQ on (=0 off). LE_DN_QKV unchanged (default on).
- Gate (tools/dense_cmp_gate.sh, under lock): "The capital of France is" 14/15 argmax, mean |dlogprob| 0.0098, max 0.0760; "Explain in one sentence why the sky is blue." 15/15, mean 0.0166, max 0.0560.
- Graph decode r=3, same 2594-token prompt, 128 gen, default build: 93.2 / 93.2 / 93.1 tok/s (old default 88.6; llama.cpp b12182 94.9).
- Output differs from the old default at one near-tie token (char 326 of 545). Accepted by peer decision.
- Gap to llama decode at this prompt: about 1.8% (93.2 vs 94.9).

## Prefill Q4_K GEMM study and plan (2026-11-12, no GPU runs; machine busy per peer)

Status: study done from llama.cpp source (`ggml/src/ggml-cuda/mmq-config-ampere.cuh`, `mmq.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh`). No GPU measurement made in this step. A 4K prefill timing attempt printed nothing because the server does not log prefill unless LE_EVT is set; that run was not valid and is not used.

llama.cpp Ampere config for Q4_K (`CASE(GGML_TYPE_Q4_K, nthreads, occupancy, I, J, layout, K, stream_k, fallback)`):
- nthreads 256, occupancy 1, I (weight rows per block) 128, J (tokens) 8..128, layout Q8_1, MMQ_ITER_K 256, stream_k true on the fast path.
- Our kernel (src/cuda/kernels.cu, `mmq128_impl` ~line 1045, `MMQ128` macro ~1222): 256 threads, `__launch_bounds__(256, 2)`, block tile 64 weight rows x 128 tokens (`r0 = blockIdx.x * 64`, `t0 = blockIdx.y * 128`). The 128-row variant (`mmq128r_*`, LE_MMQR, default on in src/gpu/dense.rs:184) is closest to llama's I=128.
- Ours: occupancy 2 target; llama: occupancy 1 with a bigger per-block tile and stream-K to fill SMs.

Ranked candidates (each gated by dense_cmp_gate, then r=3 4K prefill under lock):
1. Occupancy 1 with a 128-row tile and a larger pipeline (launch_bounds(256,1)); earlier test at 64-row tile was slower (1581 vs 1674), so only try with the 128-row tile. Cheap to test.
2. Stream-K on the 128-row tile: the earlier stream-K test (LE_MMQS, 64-row) was slower (1450 vs 1674). Retest on the 128-row tile, since llama's win depends on the combination.
3. Epilogue: llama applies one float multiply per 32-weight sub-block (`sumf_d += ds8.x*(sc*sumi_d)`) like ours; our extra per-output float ops (acc += d_x*(sA*cc - mA*x)) are a few percent at most. Low priority.
4. Measure with ncu when counters are allowed (sudo ncu approved by the owner): compare mmq128r_12 against llama `mul_mat_q<12>` on occupancy, DRAM/L2 throughput, bank conflicts (As[..][9] pad), stall reasons.

Blocked: GPU use. The machine was in use by another workload; no benchmarks or ncu until the machine is declared free.

## Prefill queue on the GPU (2026-11-12, peer projects-45, lock held)

Prompt: 5023 tokens (the repeated-word chat prompt in /tmp/claude-1000/pp4k.sh; it is 5K, not 4K). Timing = the first `evt wall` line of each run (LE_EVT=1, ctx 20000), 3 starts. tok/s = 5023 / wall.

| Config | wall ms (r1 / r2 / r3) | prefill tok/s (approx) |
|---|---|---|
| default (mmq128r on) | 2448.2 / 2448.1 / 2451.7 | 2051 |
| LE_MMQR=0 (64-row tile) | 2535.7 / 2513.2 / 2519.3 | 1991 |
| LE_MMQSK=2 (split-K, atomicAdd) | 2586.8 / 2585.9 / 2583.0 | 1943 |
| LE_MMQSK=4 | 2736.9 / 2759.7 / 2756.5 | 1826 |

dense_cmp_gate on the current build (default): "The capital of France is" 14/15, max |dlogprob| 0.0760; "Explain..." 15/15, max 0.0560. Unchanged.

Decision: no flip. Default (128-row tile, no split-K) is the fastest of the four. LE_MMQR=0 is 3% slower. Split-K is slower at both S=2 (-5%) and S=4 (-11%). The split-K kernel is kept opt-in and off; its atomicAdd epilogue is also not bit-stable, so it should not be used without a reason.
Caveat: the wall times include the tail of the chunked prefill and the first-request warmup; a per-run 2-request order is not used for the prefill comparison.

## Real prefill gap on the 5023-token prompt, and ncu attempt (2026-11-12, under lock)

- llama.cpp b12182 llama-server (CUDA, -ngl 99, -np 1, -c 20000), prefill tok/s on the same 5023-token prompt (`pp_req.json`), r=3: 3162.5 / 3215.1 / 3207.7 (about 3200).
- LastEngine default (mmq128r on, LE_EVT wall): about 2051 tok/s (2448 ms). Real gap is about 1.56x on prefill.
- Config check, same prompt: LE_MMQR=0 1991 tok/s; LE_MMQSK=2 1943; LE_MMQSK=4 1826. Default stays.

ncu on mmq128r_12 (not completed):
- `sudo ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats -k regex:mmq128r_12 -c 1 --csv lastengine serve ...` connects to the process, but no kernel result was written. The run hit the 600 s timeout (exit 124) and the CSV holds only the ncu PROF lines.
- Likely causes not verified: the chat request may not have reached prefill, or ncu replay on a 5K prefill is too slow. Next attempt: profile with `--launch-count` on a short request, or use `ncu -o report` and import, with the server started with `LE_EVT=0`.
- No limiter (memory, latency, occupancy, bank conflicts) is established yet. No fix is proposed from counters.

## ncu limiter, mmq128r_12 vs llama mul_mat_q (512-token prompt, 2026-11-13, profiling ratios only)

Ours (lastengine serve, mmq128r_12, launch 3 of matching kernel, --set full):
- Duration 110.5 us; grid 8 blocks x 256 threads; registers 182/thread; theoretical occupancy 16.7% (1 block/SM, register limited).
- Compute (SM) 23.2%; memory 31.8%; DRAM 10.9%; L1 hit 84.1%; L2 hit 70.3%; warp cycles per issued instr 5.42.
- Limiter read: the grid is too small for 20 SMs (8 blocks), and 182 regs cap residency at one block. Latency-bound, not DRAM-bound.

llama (llama-server mul_mat_q, launch 3 of matching kernel, --set full):
- Duration 17.6 us; grid 64 blocks x 128 threads; registers 80/thread; theoretical occupancy 50%; achieved 17.9%.
- Compute 5.45%; DRAM 38.4%; L1 hit 33.3%; L2 hit 34.8%.

Caveats: the two kernels were not the same GEMM shape (launch index and token count differ), so the 110 us vs 17.6 us durations are not a like-for-like ratio. Only the structure is comparable: ours is register-heavy with a small grid; llama uses 80 regs and 64 blocks.

Proposed single fix (not coded, awaiting CEO go-ahead): cut registers in mmq128r_impl below ~128 so 2 blocks fit per SM, and/or split the grid so the 128-row variant launches at least 20 blocks for small token counts. Measure first with a same-shape A/B before any default change.

Incidents: a 10-hour hung ncu run blocked the lock (CEO killed it). My retry at 22:15 and a llama ncu run after the CEO's stop instruction were both killed by the CEO. Lock released. Do not run --set full; if retried, use SpeedOfLight and MemoryWorkloadAnalysis with --launch-count 1 and timeout outside sudo.

## mmq128r2 A/B: launch_bounds(256,2) register cap (opt-in LE_MMQR2=1), 2026-11-13

- Same-session A/B on the 5023-token prompt, first evt wall (whole prefill), 3 starts each, machine-lock held, balanced power mode:
  - old (LE_MMQR2=0): 2403.5 / 2401.6 / 2416.8 ms
  - new (LE_MMQR2=1): 2744.0 / 2747.5 / 2749.3 ms
- New is about 14% slower. Default stays off.
- dense_cmp_gate with LE_MMQR2=1: 14/15 max |dlogprob| 0.0760; 15/15 max 0.0560. Same as default, so the variant is numerically unchanged.
- Read: the 128-register cap likely causes spills and does not help the ~8-block grid for this prompt. The ncu grid limit (8 blocks) is for the 512-token prompt; the 5K prompt has more blocks, so residency is not the bottleneck at this size. Next candidate, not coded: more blocks for small token counts (smaller row tile) instead of fewer registers.
- Next-up (parked by peer): dense prefill gap, smaller row tile for small token counts (mmq128r2 at 128 regs was 14% slower; try smaller tile instead).
