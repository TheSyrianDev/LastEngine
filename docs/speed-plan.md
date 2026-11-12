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
