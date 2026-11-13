# LastEngine plan (approved answers 2026-11-05)

Goal: fast local inference for MoE models on small machines, started by llama-swap, used by LastHarness. Picks its own flags.

## Decisions
- HTTP boundary stays: llama-swap launches `lastengine-server`; LastHarness is unchanged.
- Baselines: Sage-A1.5-35B-A3B (APEX-I-MiniPlus) on RTX 3050 4 GB / 24 GB RAM first; a Qwen3.x MoE as the universal second baseline. Compare against legacy binary and llama.cpp.
- Model targets: Sage/Qwen MoE, LFM3 series (LFM2.5 dropped).
- Platforms: all (Linux, Windows, macOS; NVIDIA, AMD/Intel, Apple, CPU-only). Design a backend trait from day one; CUDA + CPU first, then Vulkan, then Metal.
- Explicit flags always override auto-tune. Keep the legacy flag names accepted (--model --addr --gpu --ctx --kv --chunk --vram-margin --threads --mtp --slot-save-path --chat-template-file, YaRN flags).

## Server contract (from LastHarness, verified file:line in LastHarness/crates/lh-provider, lh-core)
/v1/chat/completions SSE with stream_options.include_usage and prompt_tokens_details.cached_tokens; reasoning_content; tool_calls by index; stop generating when the client socket closes; /v1/models (health, data[0].meta.n_params); /props (n_ctx, supports_tools, supports_reasoning_effort, modalities.vision); /slots/0?action=save|restore with --slot-save-path (n_saved/n_restored). Not needed first: tool_choice, logprobs, seed, stop, vision input, Anthropic route, embeddings.

## Milestones
1. Server skeleton with stub backend + bench harness (legacy, llama.cpp, new).
2. GGUF loader, tokenizer, chat templates (from specs).
3. Backend trait; CPU backend; dense CUDA decode + sampling.
4. Qwen MoE + DeltaNet, then LFM3; correctness vs legacy.
5. Expert tiering (pinned host experts, VRAM cache, CPU misses, prefetch, persisted heat).
6. KV slot save/restore.
7. Auto-tune (probe + measured search, cached per machine+model).
8. Prefill speed, KV quant, speculative decoding (MTP, n-gram).
9. Vulkan, then Metal; later SSD experts, vision, embeddings.

Ideas per source engine: see TECHNIQUES.md. Rule: ideas only, no source copying.

## Status log
- 2026-11-05 M1 done: `lastengine serve --stub` serves /v1/models, /props, /health, /v1/chat/completions (SSE + usage + cached_tokens, stops when client disconnects).
- 2026-11-05 M2 part 1: `src/gguf.rs` reads headers from spec; `lastengine inspect FILE` verified on Sage-A1.5 (15 GB).
- Sage-A1.5 facts (from header): arch qwen35moe, 41 layers, full attention every 4th layer (rest DeltaNet/SSM: conv 4, state 128, inner 4096), 256 experts / 8 used, expert ff 512 + shared expert 512, 16 heads / 2 KV heads, head dim 256, rope dim 64, 1 MTP layer, tokenizer gpt2/qwen35 pre-tokenizer, Jinja chat template (needs a Jinja engine; plan: minijinja crate).
- Local references: engines in ~/AI/engines (llama-mainline baseline, gs-mce-wiki docs for expert-cache ideas), llama-swap config ~/AI/rig/router/config.yaml, launch guard ~/AI/rig/tools/lastengine-serve.sh (first arg = min free MiB).
- 2026-11-05 Stats endpoints (user request): GET /stats (model info, uptime, active requests, totals incl. prefill/decode tok/s, avg TTFT, last request), GET /stats/requests?limit=N (last 200 requests). Each chat response also carries a llama.cpp-style `timings` object (prompt_n/ms/per_second, predicted_*, cache_n, time_to_first_token_ms). Backends report `Usage.prefill_ms`; the server measures TTFT and decode time. Per-token latency, VRAM/cache hit rate and expert-cache stats to be added when real backends exist.

## Status 2026-11-05 and next steps
Done: tcq3 KV (buun-llama MIT design), q8/q4 KV, YaRN, split-PV decode attention, flash prefill, DeltaNet register state, `probe --json`.
Next, in order:
1. **Measured auto-tune** (`lastengine tune --model M`): run short prefill+decode trials over chunk x kv x expert-slot budget, cache the best by (gpu, model hash, ctx) in ~/.cache/lastengine; `serve` loads the cached result when flags are unset. Today's defaults are only rules.
2. **MTP / speculative decode**: Sage has MTP block blk.40; draft 1-3 tokens, verify as a batch through the prefill path; legacy reached up to 110 tok/s on larger GPUs.
3. **Expert-miss reduction**: ~40% of expert lookups miss on 4 GB VRAM; options: CPU-compute missed experts (needs CPU kernels), persistent expert-heat file, predictive prefetch from the previous token's routing.
4. **Prefill GEMM pipelining** (cp.async double-buffer in mmq tiles).
5. **Per-layer sync removal / CUDA graphs** for decode (about 16 of 35 ms/token is host gaps).
6. Embedded library crate + Vulkan/Metal backends behind the Backend trait.

## MTP result (implemented, off by default: `LE_MTP=1`, `lastengine gpu`)
Loads the nextn head (eh_proj + layer 40 attn/MoE), drafts one token, verifies [cur, draft] as a 2-token batch, snapshots/restores DeltaNet state on reject. Greedy output is identical to plain decode.
Measured on the RTX 3050 4 GB: 73% of drafts accepted, but 9.9 tok/s vs 28 tok/s plain. A 2-token batch needs the *union* of both tokens' experts, so PCIe traffic rises ~3x (15.1K vs 5.2K misses on 64 tokens) and the batch path has a high fixed cost. MTP pays off only when experts are VRAM-resident (bigger GPUs). Not wired into `serve`.
Known gap: the MTP layer's KV is not filled for prompt positions (only for decoded ones).

## Proposals for large decode/prefill gains
Abstract view: decode here is a *data-movement* problem, not a compute one. Per token the model touches 8 experts x 40 layers (~1.3 MB each); VRAM holds ~25% of them, so ~80 expert fetches/token (~100 MB over PCIe, ~8-10 ms) plus ~16 ms of host/launch gaps plus ~19 ms of kernels. Prefill is the mirror image: each chunk must stream nearly all 12.9 GB through a 4 GB window (~1.1 s/chunk at PCIe speed), so chunk size x PCIe bandwidth caps it near ~1900 tok/s; we reach ~800.
(Figures are estimates from profiles above, not new measurements.)

1. **Move compute to the data (CPU runs missed experts).** A miss costs 1.3 MB of weights over PCIe; the activation is 8 KB. Run missed experts on CPU against the RAM copy (already pinned, already repacked in legacy) concurrently with the GPU hits and add the 8 KB result back. Cuts PCIe traffic ~150x for misses; the CPU only needs ~1-2 experts/layer on average. Expected: removes most of the ~8-10 ms miss wait and the cache-pressure problem; decode 28 -> ~40+. Biggest, most structural change. Needs an AVX2/VNNI IQ3_XXS/Q3_K/Q4_0 dot kernel.
2. **Kill the launch gaps (CUDA graphs / persistent per-layer kernel).** ~16 of 35 ms/token is host latency (routing readback, per-layer sync, three memcpy-style steps). Make routing device-side: top-k writes expert pointers into device memory, the expert matvecs read pointers indirectly, so a full layer is capturable as a graph with no host round trip. Misses become a *device-signalled* slow path (graph exits to host only when it needs a fetch). Ceiling ~19 ms/token -> ~50 tok/s before misses.
3. **Deeper, learned prefetch.** Today we prefetch layer l+1 from layer l. The residual stream changes slowly, so a tiny predictor (or just reusing router weights on the layer-l hidden state) can predict l+2..l+4, giving PCIe 3-4 layers of lead time instead of one and hiding the remaining misses behind compute.
4. **Profile-guided resident set.** Offline (or during `tune`) record per-expert routing frequency on representative prompts; pin the hottest ~N experts permanently and use LRU only for the rest. Persist per model. Raises hit rate without any per-token cost; combines with 1 (CPU handles the cold tail).
5. **Cache-aware routing (opt-in, lossy).** When expert k+1 is already resident and its router score is within epsilon of expert k's non-resident score, take the resident one. Tunable quality/speed knob exposed to `tune` with a perplexity guard.
6. **Prefill: stop dequantizing in the inner loop.** At load, repack hot tile layouts so the int8 mma path loads weights without unpack/grid lookup (IQ3 grid lookups are ~half of GEMM time). Combine with a software-pipelined (cp.async double-buffered) GEMM and larger effective chunks via KV tcq3 (frees VRAM -> bigger chunk). Target 800 -> ~1200+ tok/s.
7. **Prefill: overlap CPU as a second worker.** While the GPU streams wave w+1, the CPU can compute a share of experts for the chunk from RAM (proposal 1's kernel reused for batched rows), so effective bandwidth is PCIe + RAM compute rather than PCIe alone.
8. **Speculation that does not multiply expert traffic.** Plain MTP fails here (union of experts). Only worth revisiting after 1-4 make misses cheap; then tree/batch-2 verification becomes compute-bound and MTP's 73% acceptance would yield ~1.7x.

Recommended order: 2 (cheapest large win, no new kernels) -> 1 (structural) -> 4 -> 3 -> 6 -> 7 -> 5 -> 8.

## Update (after MTP)
Decode 28 -> 34.7 tok/s via: one routing readback per layer, warp-parallel top-k, fused gate+up expert matvec. Fusing add+rmsnorm gave no measurable change: small-kernel launch count is NOT the bottleneck (kernels ~21 ms of ~29 ms/token; the rest is the per-layer host sync + miss waits). Remaining levers are structural: device-side routing (graph capture), CPU execution of missed experts, learned multi-layer prefetch. 28K tcq3: prefill 54.6 s (513 tok/s), decode 27.2 tok/s.

## Correction to proposal 1 (CPU runs missed experts)
Re-costed before building: an expert is ~1.3 MB; at ~12 GB/s PCIe a miss is ~0.1 ms, so ~68 misses/token is ~7 ms, partly overlapped. The CPU would need ~3M multiply-adds per expert; even with a vectorised int8 kernel at ~20 GMAC/s/core that is ~0.15 ms/core/expert, i.e. no better than the PCIe copy, and the existing `dequant_row` path is far slower. So CPU-side experts are NOT a clear win on this machine; dropped to low priority. Remaining credible decode levers: device-side routing + graph capture (removes host syncs), deeper prefetch, a profile-guided resident set.

## Design for device-side routing + graphs (investigated, not built)
Blocker found in `src/gpu.rs` (`slot_for`/`pick_slot`/`prefetch`): the expert cache (slot map, LRU ticks, copy issue) is entirely host state, so every layer needs the routed expert ids on the host before the expert matvecs can be launched. Graph capture is therefore impossible until the cache lookup moves to the device. Required pieces:
1. Device-resident table `slot_of[layer*n_expert+e] -> slot|-1` and device LRU stamps; `router_topk` also emits slot pointers (gather from the table) into the pointer buffer the matvecs already read.
2. Hit-only fast path inside a captured per-layer graph; a miss flag written by the kernel. Host checks one flag per layer (still a sync, but only for fetch decisions) or per token, falling back to the current path on a miss.
3. Because ~25-30% of lookups miss at 4 GB VRAM, most layers would still take the slow path; expected gain is bounded by hit rate (~70%) times the ~8 ms/token of host gap, i.e. maybe 3-5 ms/token (34 -> ~38-40 tok/s). Real win needs a higher hit rate first (bigger VRAM budget, or resident-set pinning), so this is deferred until that lever exists.

## Hot-expert pinning (tried, rejected)
`LE_PIN=f` makes the top fraction f of prefill-hot experts non-evictable. 1200-token prompt, 128 decoded: f=0 -> 35.3 tok/s (10.5K misses), f=0.3 -> 34.9 (10.9K), f=0.6 -> 32.7 (12.1K). Pinning raises misses: routing during decode is not well predicted by prefill heat, and pinned slots shrink the LRU pool. Default stays 0 (off); kept as an experiment knob. Hit rate therefore cannot be bought cheaply, which further weakens the case for the device-side cache rewrite.

## Device-side expert pointer build (built, neutral)
`build_ptrs` kernel + a mapped host/device slot table (`tab_h`/`tab_d`, mirrored in `pick_slot`/`warm_cache`) now build the expert pointer arrays on the GPU, replacing a `memcpy_htod_sync` per layer. Output unchanged (same hits/misses, sensible text). Throughput with the user's background build running: 28.2/30.5/31.4 tok/s vs baseline 31.0/31.0/31.3 -- no measurable gain. The remaining host sync is the idx readback needed for miss fetches, so graph capture still needs a hit-only path (flag already written at idx+128 for that).
