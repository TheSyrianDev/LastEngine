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
