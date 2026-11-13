# Handoff: concurrency-p3 (continue from here)

Branch `concurrency-p3`, local only, nothing pushed. Last commit `673ee06`.

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
