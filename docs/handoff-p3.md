# Handoff: concurrency-p3 (continue from here)

Branch `concurrency-p3`, local only, nothing pushed. Last commit `c740d84`.

## Done
- 3a `92e6136`: `batchgen::generate` library, `batch-gen` CLI; tokens equal `batch-loop` (6 prompts, N=20, under lock).
- S1 `f6fb39c`: `src/reqsched.rs` channel scheduler; 2 unit tests. Named reqsched because `src/sched.rs` already exists (FIFO slot scheduler).
- S2 `4a9a929` + `673ee06`: `LE_BATCH=1` starts a thread owning the dense CUDA GPU; greedy chat requests in `Local::generate` go through the channel. Temperature>0 returns an explicit error (S5 pending). `673ee06` makes the batch-path splitter start in think mode like the old path.
- Tests: 46/46 (`cargo test --release`).

## Not verified
- **S3 gate** (N=1 greedy curl, LE_BATCH=0 vs 1): not passed. Script `/tmp/claude-1000/s3.sh` (serve needs `--gpu --model M`; first runs used the wrong positional arg and failed). The pre-`673ee06` run showed the batch path emitted reasoning text as content (`out_1.txt`). `673ee06` fixes that, but the gate has not been rerun. The lock was held by other sessions on the last attempts (`codex-checklist`, `lh-slash-e2e`, `lastharness`).
- **S4** (N=2/4 concurrent curl): not started.
- **S5** (per-slot sampling RNG, temperature/top_p in batch path): not started. Plan: sampling with a per-slot RNG in the scheduler loop, not the shared `g.rng`.
- **`llama-server -np 8` bench**: not run. `docs/BENCHMARKS.md` has np 1/2/4/6 for llama; np 8 is missing.

## Next steps
1. Run S3 under `tools/gpu-run.sh` (fix `s3.sh` if needed), compare `out_0.txt` and `out_1.txt`. Equal means S3 passes.
2. S4: 2 and 4 concurrent curl requests, greedy, compare each with the solo output.
3. S5: sampling in the batch path with per-slot RNG; gate on determinism with a fixed seed.
4. Run `llama-server -np 8` and the batch server at np 1/2/4/8 under lock; add to `docs/BENCHMARKS.md`.

## Rules
- GPU only via `tools/gpu-run.sh <tag> <cmd>`. A refused take means wait, not run.
- Commit each step only after the full test suite and gate pass.

## Update (later session)
- S3 PASS: LE_BATCH=1 greedy equals LE_BATCH=0 (tools/gpu-run.sh s3-serve bash /tmp/claude-1000/s3.sh; out_0 == out_1).
- S4 PASS: 4 concurrent greedy requests equal solo (tools/s4_concurrent_gate.sh, commit c7eb9d3).
- S5 DONE: per-request sampler `batchgen::Picker` (temperature/top_p, xorshift per request), commit c740d84; unit test 47/47; smoke tools/s5_sampling_smoke.sh non-empty. Determinism across restarts not yet gated (seed is a per-request counter).
- -np 8 bench: docs/BENCHMARKS.md section added. Our batch path flat at ~96 tok/s from N=4; llama -np 8 reaches 336 at N=8. Known bug: batched rows >6 diverge (BATCH_ROW_CAP=6), so the N=8 LE number is invalid. N=1/2 LE rows lost; rerun tools/np8_bench.sh.
- Open next: fix rows 7-8 divergence, then scale-up of batch throughput; determinism gate for sampling; push needs Omar's approval.

## Update (step 1 and step 2)
- Step 1 DONE e2fe0f0: serve_interleaved (budget_step decodes + chunked prefill via DenseGpu::prefill_at), LE_CHUNK/LE_BUDGET. tools/chunk_gate.sh: 4/4 concurrent requests equal solo. Not benchmarked yet.
- Step 2 DONE (profile): tools/prof_decode_n4.sh (LE_PROF=1 batch-gen, N=4, 64 tokens). Per-kernel share of decode_batch: mmq128r_12 74%, mmq128r_14 16%, matvec_14 2%, quant_x 2%, dn_flash_128b 0.8%. dn_attn_rows was NOT the bottleneck; the earlier hypothesis (attention limits throughput) is wrong.
- Finding: decode_batch runs its weight GEMM through the prefill-style mmq128r kernel even at 4 rows, so each weight is streamed with little reuse across rows. The next lever is a small-N Q4_K GEMV that reads each weight once and produces all N row outputs (N<=8). Step 3 = write and gate that kernel (batch-invariance and solo-equality gates).
- Still open: rows above 6 diverge (BATCH_ROW_CAP=6); sampling determinism gate; beat llama.cpp (not met: N=8 LE 96 tok/s vs llama 337).
