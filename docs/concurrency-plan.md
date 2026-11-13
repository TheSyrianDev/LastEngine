# Concurrency plan (Beta as backend for Last AI)

Status: plan only, written before coding (requested by projects-45). Nothing here is implemented yet.

## Current behaviour (measured or read from code)

- Single model, single decode loop. Concurrent HTTP requests are accepted and served one generation at a time through the shared backend. The integration test "3 concurrent requests all 200" passes, but it does not show throughput or latency under load.
- Client disconnect cancels generation (verified by `tests/api_integration.sh`, "disconnect cancels generation").
- KV cache is a single contiguous buffer sized by `--ctx`. Measured on the 4 GB RTX 3050 Laptop with MiniCPM5-2B Q4_K_M: `--ctx 20000` leaves 1911-2037 MiB free after weights (two runs, different loads). The free figure is the budget for KV plus activations.
- No prefix cache across requests beyond what the engine already does (`cache_n` in timings).

## Memory math for 4 GB

Budget: about 1.9 GB free for KV plus scratch at ctx 20000 with MiniCPM5. Per-token KV size is not yet measured in this doc; get it from the model config (layers, kv heads, head dim, f16 bytes) before any sizing decision. Target sizing (to fill in after measuring):

| Model | Context | Concurrent slots | KV per slot | Fits? |
|---|---|---|---|---|
| MiniCPM5-2B Q4_K_M | 4K | TBD | TBD | TBD |
| MiniCPM5-2B Q4_K_M | 16K | TBD | TBD | TBD |
| Sage (if used) | 4K / 16K | TBD | TBD | TBD |

Rule: slots × ctx × KV-bytes-per-token + scratch must fit in measured free memory. Prefer one slot at 16K over several slots at 4K unless the workload says otherwise.

## Phases, smallest first

1. **Request queue with bounded concurrency (1 active slot).** Keep one active generation; queue the rest with a cap (e.g. 8) and return 429 with a JSON error when full. Exit criterion: no crashes under 16 parallel clients; every request gets 200 or a clean 429.
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
