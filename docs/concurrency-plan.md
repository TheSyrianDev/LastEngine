# Concurrency plan (Beta as backend for Last AI)

Status: phase 1 implemented (LE_MAX_QUEUE, default 8: one active generation, bounded queue, 429 server_busy when full; tested by api_integration.sh "queue"). Other phases not started.

KV bytes per token (MiniCPM5-2B, from GGUF metadata): 42 layers x 2 KV heads x 128 head dim x (K+V) x f16 2 bytes = 43,008 B = 42 KiB per token. Engine KV is f16 (src/gpu/dense.rs:100).

## Current behaviour (measured or read from code)

- Single model, single decode loop. Concurrent HTTP requests are accepted and served one generation at a time through the shared backend. The integration test "3 concurrent requests all 200" passes, but it does not show throughput or latency under load.
- Client disconnect cancels generation (verified by `tests/api_integration.sh`, "disconnect cancels generation").
- KV cache is a single contiguous buffer sized by `--ctx`. Measured on the 4 GB RTX 3050 Laptop with MiniCPM5-2B Q4_K_M: `--ctx 20000` leaves 1911-2037 MiB free after weights (two runs, different loads). The free figure is the budget for KV plus activations.
- No prefix cache across requests beyond what the engine already does (`cache_n` in timings).

## Memory math for 4 GB

Budget: about 1.9 GB free for KV plus scratch at ctx 20000 with MiniCPM5. Per-token KV size is 42 KiB (see above). Sage is still TBD. Target sizing:

| Model | Context | Concurrent slots | KV per slot | Fits? |
|---|---|---|---|---|
| MiniCPM5-2B Q4_K_M | 4K | 1 (queue behind) | 42 KiB x 4096 = 172 MiB | yes (measured free 1.9 GB at ctx 20000) |
| MiniCPM5-2B Q4_K_M | 16K | 1 | 42 KiB x 16384 = 672 MiB | yes, tight with 2+ slots |
| Sage (if used) | 4K / 16K | TBD | TBD | TBD |

Rule: slots × ctx × KV-bytes-per-token + scratch must fit in measured free memory. Prefer one slot at 16K over several slots at 4K unless the workload says otherwise.

## Phases, smallest first

1. **Request queue with bounded concurrency (1 active slot).** DONE (LE_MAX_QUEUE). Keep one active generation; queue the rest with a cap (e.g. 8) and return 429 with a JSON error when full. Exit criterion: no crashes under 16 parallel clients; every request gets 200 or a clean 429.
2. **Fair scheduling.** FIFO across clients so one long prompt does not starve short ones. Exit: p99 TTFT of short requests bounded while a long prefill runs.
3. **Chunked prefill interleaved with decode.** Split prefill into chunks so decode steps for active sessions run between chunks. Needs the chunk size tuned for the 4 GB card. Exit: decode token gaps stay under a fixed bound during another client's prefill.
4. **Multiple KV slots (static).** Partition the KV buffer into N fixed slots sized from the memory math above. Exit: N concurrent generations produce output identical to running them one at a time (dense_cmp style check).
5. **Paged KV.** Fixed-size blocks allocated on demand, so short requests do not reserve a full context. Only after phase 4 shows memory is the limiter.
6. **Continuous batching.** Batch decode across active slots in one kernel launch. This needs the decode path to take a batch dimension. Measure against phase 4 before building.
7. **Prefix caching.** Reuse KV for shared system prompts across requests (Last AI sends repeated prefixes). Largest expected win for agent traffic.
8. **Preemption / swap.** Only if phases 5-7 still run out of memory under the target load.
9. **Batched MoE expert dispatch.** Only for MoE models; not needed for MiniCPM5 (dense).

## Benchmark plan

- Tool: `llama-server -np N` (b12182, same build as the baseline) and LastEngine on the same model, same prompt set, same card.
- Client counts: 1, 4, 8, 16.
- Metrics: aggregate throughput (tok/s), TTFT p50 and p99, per-token latency p50 and p99, error rate.
- Prompts: fixed set, 512-token and 4K-token prompts, 128 generated tokens.
- Run under `~/AI/Rig/bin/machine-lock`. Report power mode with every number; power-saver numbers are ratios only.
- Compare vs vLLM only if it runs on this card; otherwise llama-server is the reference.

## Open questions

- Per-token KV bytes for MiniCPM5 (read from GGUF metadata, not assumed).
- Whether the server should return 429 or queue indefinitely by default. Proposal: queue with a timeout, then 503 JSON error.
- Whether Last AI needs per-request slot priority.

## Phase 2 design: continuous batching for decode (MiniCPM5 dense), not started

Gate (projects-45): each sequence's greedy output equals its single-sequence output token for token, and aggregate tok/s at 1/2/4/8 clients beats `llama-server -np` at the same settings.

What the code does today (src/gpu/dense.rs):
- One KV buffer per layer, `kc[l]`/`vc[l]` sized `ctx_len * nkv * hd * 2` (f16), one sequence only.
- Decode position is a device scalar (`dpos`/`pdev`), and attention kernels (`dn_attn`, `dn_attn_gqa`, `dn_attn_split`) take a single `nk`.
- The linear layers already have a multi-token GEMM path (`bmm`, mmq128r) used by prefill. That path can serve N decode rows at once for the projections.

Steps, each a separate commit with its own gate:
1. Per-sequence KV regions: allocate `slots × ctx × KV` per layer, with a slot index passed to every KV write and attention kernel. Gate: N=1 output unchanged vs HEAD (dense_cmp gate).
2. Batched projections: decode N rows through the existing bmm path (rows = sequences). Gate: N=1 unchanged; N=2 rows equal to two single decodes (dense_cmp per row).
3. Per-sequence attention: a kernel that takes per-slot `nk` and `pos`, so sequences of different lengths share one launch. Gate: per-row logits equal single-sequence logits within the dense_cmp tolerance.
4. Scheduler: each iteration picks active sequences, runs one batched step, samples per sequence, admits new requests and retires finished ones. Gate: token-for-token match to single-sequence greedy at 1/2/4/8 clients.
5. Benchmark vs `llama-server -np 1/2/4/8` on the same prompts, under machine-lock, with power mode recorded.

Blocker to flag: step 1 touches every KV kernel and the graph capture. It is multi-session work, not a single change.

## Later phases (added by projects-45)

- **Grammar-constrained decoding for tool calls.** Small models need it: constrain sampling to the tool-call JSON schema from the tool definitions. Gate: tool-call output parses on every try at temperature 0 and at a sampled temperature.
- **Sage tool-call check.** Run the tool-call tests (forced, natural, stream) on Sage once RAM is free. Not possible now (RAM in use by other jobs).
