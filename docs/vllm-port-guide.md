# vLLM v1 porting guide for LastEngine

Attribution: the design below follows vLLM (github.com/vllm-project/vllm, Apache-2.0, Copyright vLLM contributors), read at `main` on 2026-11-13. Concepts and algorithms are re-implemented in Rust/CUDA, not copied. If any vLLM code is ever copied verbatim, keep its license header and add it to `NOTICE`. Note that `vllm/v1/worker/gpu_model_runner.py` is now `vllm/v1/worker/gpu/model_runner.py` plus `block_table.py`, `input_batch.py`.

Premise (from docs/concurrency-plan.md): one sequence = one contiguous slot today (`kc[l]/vc[l]` = `slots*ctx*nkv*hd` f16, 42 KiB/token for MiniCPM5-2B); `decode_batch(toks, slots, poss)` at src/gpu/dense.rs:279; CPU planning in src/batchplan.rs (`plan_step`, `advance`, `step_once`) and src/batchworker.rs (`run`). Open blocker: the prompt-specific batched-vs-solo divergence ("Say hello in French."). Nothing below ships as default until that gate is green.

## Mapping table

| vLLM piece | LastEngine change |
|---|---|
| `Scheduler.schedule()` (sched/scheduler.py): `token_budget = max_num_scheduled_tokens`; loop over `self.running` first, then `self.waiting` | New `plan_step2` in src/batchplan.rs returning per-seq `num_new_tokens` (1 for decode, up to chunk for prefill) under a step token budget; replaces `StepInput` rows-only model |
| Chunked prefill: `num_new_tokens = min(remaining, long_prefill_token_threshold, token_budget)` | Add `prompt_len`, `num_computed` to `Seq`; prefill rows use the existing multi-token `bmm` path with a chunk size tuned for 4 GB (concurrency-plan phase 3) |
| `KVCacheManager.allocate_slots()` (kv_cache_manager.py) returns `None` when blocks are insufficient | `BlockAllocator::allocate_slots(seq, n_new) -> Option<Vec<BlockId>>`; the scheduler's only admission and growth test |
| Recompute preemption: `Scheduler._preempt_request()` pops `self.running[-1]` (or lowest priority), `_free_request_blocks`, `num_computed_tokens = 0`, `waiting.prepend_request` | On `None` from `allocate_slots` for a running seq: free the newest running seq, reset its `num_computed`, push to front of the wait queue; restore `token_budget` for any row already planned for the victim (see the `scheduled_running_reqs.remove` path). No swap (matches phase 8: skip swap) |
| `BlockPool` (block_pool.py): `get_new_blocks(n)` -> `free_block_queue.popleft_n`, sets `ref_cnt=1`, evicts stale hash via `_maybe_evict_cached_block`; `free_blocks()` decrements `ref_cnt`, at 0 re-queues; `touch()` increments (removes from free queue) | `src/kvpool.rs`: `Vec<Block{ref_cnt, hash: Option<u64>, prev, next}>` index-linked list, no allocation on push/pop |
| `FreeKVCacheBlockQueue` (kv_cache_utils.py): doubly linked, O(1) remove from middle, LRU front; uncached blocks freed to front (LIFO), cached blocks to back (FIFO); a request's blocks freed in reverse so the chain tail evicts first | Same ordering rules; implement with `u32` prev/next arrays; unit test the eviction order |
| Block table: `BlockTables` (worker/gpu/block_table.py): `append_block_ids`, `gather_block_tables`, device `[reqs, max_blocks]` int32 | Device buffer `block_table[row][max_blocks]` (i32) uploaded per step, packed next to the existing `[pos][slot][len]` metadata (src/gpu/dense.rs:25) |
| Slot mapping: `_compute_slot_mappings_kernel`: `slot = block_table[req][pos // bs] * bs + pos % bs` | New tiny kernel (or CPU compute, upload) feeding KV-write kernels `dn_prep_rows`; replaces the current `slot*stride + pos` (stride bug at 0a3c15a shows units are easy to get wrong: keep everything in tokens, convert to f16 elements once) |
| Attention reads via block table (paged attention) | Variant of `dn_attn_rows` taking `block_table` and `nk` per row; gather K/V by `block_table[row][j/bs]*bs + j%bs`. Block size 16 tokens = 672 KiB per block across 42 layers |
| Prefix cache: `hash_block_tokens(hash_fn, parent_hash, block_token_ids, extra_keys)` (kv_cache_utils.py), `get_request_block_hasher`; `BlockPool.cache_full_blocks()` registers only full blocks; `get_cached_block()`/`KVCacheManager.get_computed_blocks()` walk the chain and stop at first miss, capped at `prompt_len - 1` so at least one token is computed | `hash = H(parent_hash, tokens_in_block, extra)` with a 64-bit (xxh3/fnv is enough, but see gate) hash stored in a `HashMap<(u64), BlockId>`; `extra` = model id + template/system flag. Matched blocks get `touch()` and become the start of the seq's table; `num_computed = hits*bs` |

## Order of implementation (each step its own commit and gate)

0. Root-cause the batched-vs-solo divergence. Gate: dense_cmp and batch-loop identical for the failing prompt at N=1 and N=6. Nothing below proceeds before this.
1. `BlockAllocator` + free queue, pure Rust, no GPU. Gate: unit tests for alloc/free/ref-count, eviction order (LIFO uncached, FIFO cached, reversed per-seq free), exhaustion returns `None`, double-free panics. A randomized test (10k ops) checks `free + used == total`.
2. Block table + slot mapping with fixed block size, one block table per slot, blocks assigned contiguously at first. KV kernels use the table but results are bit-identical. Gate: dense_cmp N=1 unchanged; batch-test N=2 and batch-loop N=4 30/30 identical to solo; memory unchanged.
3. Non-contiguous blocks: allocate on demand, shuffle block ids in a test mode (`LE_BLOCK_SHUFFLE=1`). Gate: same outputs with shuffled physical blocks (proves the table is really used). Then raise concurrent seqs beyond `LE_SLOTS`; slot count stops being a KV partition and becomes a row cap only.
4. Scheduler step with token budget + admission via `allocate_slots` in `batchplan.rs` (pure logic first). Gate: unit tests with a fake allocator: budget never exceeded, FIFO fairness, no admission when `None`, running seqs scheduled before waiting ones.
5. Chunked prefill mixed into the step. Gate: token-for-token vs solo greedy at 1/2/4/8 clients; decode inter-token gap bound while another client prefills (phase 3 exit criterion).
6. Recompute preemption. Gate: small pool forced to run out (e.g. 8 blocks), 4 clients; every output equals solo greedy; no leaked blocks (`free == total` at idle); a preempted seq's budget is returned.
7. Prefix cache. Gate: second identical-system-prompt request reports `cache_n` of whole blocks and output equals uncached output; eviction under pressure keeps outputs correct; `cache_n` never covers the final prompt token. Handle hash collisions by also comparing stored block tokens in debug builds.
8. Benchmarks vs `llama-server -np N` at 1/2/4/8/16 clients under machine-lock, power mode recorded.

## Skip

- Multi-group KV (`KVCacheGroup`, hybrid/Mamba/sliding-window `remove_skipped_blocks`): MiniCPM5 is dense; one group.
- Async scheduling, `num_output_placeholders`, stale-output accounting, speculative decode lookahead, encoder cache, KV connectors, KV events, LoRA, DCP/PCP context parallel, `BlockHashListWithBlockSize`, partial-block hashing (`cache_partial_block`).
- Priority scheduling, swap-to-CPU, `reset_prefix_cache` machinery (keep a plain `clear()`).
- Python-level object tricks (LRU-cached hash, pre-allocated linked-list nodes): the Rust index-linked arrays already give the same cost profile.
- Triton staged-write block table machinery (`apply_staged_writes`): one `cudaMemcpyAsync` of a small i32 table per step is enough at 6-16 rows.

## Notes and risks

- 4 GB card: block size trade-off. 16 tokens wastes at most 15 tokens (630 KiB) per seq; use 16 unless kernel gather cost shows up, then 32.
- Keep the CUDA graph path working: block table and slot data must be device buffers read by the captured kernels (like `dpos`), not baked-in scalars.
- Keep `batched_eligible` routing: sampled, tool and thinking requests stay on the single path until sampling in the batched loop exists.
- Retire `BATCH_ROW_CAP = 6` only after step 0 shows the real cause (the doc's correction says the cap rationale is wrong).
- vLLM line numbers drift; cite by function name when reviewing.
