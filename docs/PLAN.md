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
