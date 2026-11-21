# Steal plan (Omar, 2026-11-21): vLLM concurrency, llama.cpp MoE cache, edge engines

Clean-room rule stays: ideas from READMEs/docs/papers/specs only, re-derived in
Rust/CUDA, every technique benchmark-gated on this machine (RTX 3050 4 GB,
24 GB RAM) before it ships. Sources read 2026-11-21, READMEs and public docs
only, no source copied. Numbers below are authors' claims, unverified here.

## What we already have (verified in src, this branch)

- MoE expert VRAM cache: LRU slots + heat table + async prefetch on a copy
  stream through pinned staging (`src/gpu.rs` pick_slot/slot_for/prefetch,
  warm_cache). Hits/misses counted. This is the Strata/gs-mce shape already.
- vLLM-style paged KV primitives, pure Rust, tested: `blockalloc.rs`
  (ref counts, LIFO uncached / FIFO cached free queue, reverse free_seq,
  touch) and `blocktable.rs` (BS=16, slot mapping, shuffle bijection test).
  NOT wired to the GPU path yet.
- Token-budget scheduler `budget_step` (decodes first, FIFO admits, allocation
  hook) + chunked prefill `prefill_chunks`, both unit-tested.
- Interleaved serving `batchgen::serve_interleaved`: many requests share the
  GPU, long prompts no longer stall decodes; per-request Picker sampling.
- Known gap: rows >6 diverge from solo (BATCH_ROW_CAP=6); profile says weight
  GEMM is 74% of decode (mmq128r at small N), attention <1%.

## Priority queue (each item: its own commit + gate)

### P1. Concurrency throughput (vLLM ideas, our existing plan)
1. Root-cause rows 7-8 divergence (gate: dense_cmp + batch-loop identical at
   N=1..8 for the failing prompt). Nothing batch-related ships before this.
2. Small-N Q4_K GEMV kernel (design in handoff-p3.md step 3): read each weight
   once, produce all N row outputs; oracle gemv_batch_ref exists. This is the
   74% lever. Gates: vs oracle per slot, N=1 bitwise-equal to matvec_12,
   batch invariance N=1/2/4/6/7/8.
3. Wire blockalloc/blocktable into the dense KV path (vllm-port-guide steps
   2-3): bit-identical outputs with the table, then LE_BLOCK_SHUFFLE proves
   non-contiguity, then slot count becomes a row cap, not a KV partition.
4. Prefix caching (guide step 7 shape): hash chain of full blocks, extra keys
   = model + template flag; agent traffic repeats system prompts, largest
   expected win for LastAI. Gate: cached vs uncached outputs identical,
   cached_tokens reported, eviction order tested.
5. Recompute preemption when the pool is exhausted (guide step 6): free the
   newest running seq, reset num_computed, requeue at front. No swap.

### P2. MoE expert caching (llama.cpp PRs #29887/#30112 + moefast + gs-mce)
6. Auto-size the expert cache like moefast: reserve = max(700 MiB, 15% free
   VRAM); cache = free - reserve - KV - dense; MIN = top_k x layers x
   bytes_per_expert, below which the cache is silently bypassed upstream.
   We must fail loudly or disable honestly, never pretend. Our n_slots math
   (gpu.rs:169-176) already budgets slots; add the minimum check + a
   `lastengine probe`-style report of the chosen numbers and why.
7. Persist expert heat across runs + prewarm (cafe moe-direct, buun heatmap):
   we have pheat + warm_cache within a run; add save/load keyed by model sha
   (tune.rs already keys by model+gpu). Gate: cold vs warm first-200-token
   decode on Sage/MiniCPM-V, measured, under lock.
8. Async cache re-learn on a helper thread (PolyStrata --adapt-async, claimed
   +4-5% decode): our prefetch is already async; the idea is updating heat /
   eviction decisions off the decode thread. Measure before believing 4-5%.
9. CPU-compute cache misses concurrently with GPU (Strata): today a miss
   copies host->device then computes on GPU. Strata computes cold experts on
   CPU in place while GPU does the hot ones. On 4 GB this trades PCIe for
   CPU; gate on Sage with misses forced (small cache), compare tok/s both ways.

### P3. Edge streaming (BigMoeOnEdge, cafe -ssd, PolyStrata)
10. SSD expert streaming: mmap only selected experts (cafe -ssd), model used
    in place, no second copy. BigMoeOnEdge runs 35B MoE from flash on phones;
    its --auto planner measures machine+model and explains every choice
    (Plan screen). For us: a `--stream` mode for models bigger than RAM,
    lossless gate vs full-RAM run. Lower priority: our RAM (24 GB) holds the
    models we target; this unlocks 70B+ class later.
11. RAM trimming: drop file pages already resident on GPU (PolyStrata claim
    17.9 -> 13.1 GiB). madvise(MADV_DONTNEED) on mapped expert ranges we hold
    in VRAM slots; gate: RSS drop + no decode regression on re-miss.

### P4. KV quality ladder (buun VBR) — after P1-P2
12. VBR: start f16, degrade one (layer, side) tensor at a time under real
    memory pressure, per-model KLD-ordered ladder. We have q8/tcq3 static;
    the idea is dynamic per-tensor tiers + advertising the largest context
    that fits above a quality floor. Needs KLD panels per model first
    (measure, do not assume the ladder order).
13. Asymmetric K/V widths (buun: K tcq3 / V tcq2 beats reverse 15-17% KLD).

### Compatibility (llama.cpp semantics, ongoing gate)
- CLI semantics reference: --n-cpu-moe, -ot regex placement, --moe-cache-mib,
  -fit. Our probe/auto-fit should speak the same concepts so Rig configs and
  moefast-style launchers map 1:1.
- lhd boundary stays green: /health, streamed tool_calls, /props, slot
  save/restore, prefix reuse, disconnect cancel (tests/api_integration.sh).
- llama.cpp upstream PR #29887 notes: LRU, batches <=32, claimed 1.6-2.2x on
  4090/5090. gs-mce wiki (MoE-Cache, Prefetch, Grouped-MoE-Decode) is the
  eviction/slot design reference — read before changing pick_slot.

## Sources (README/docs only, 2026-11-21)
- vLLM v1 scheduler/KV manager: already mapped in docs/vllm-port-guide.md.
- llama.cpp MoE cache: PRs #29887, #30112 (upstream), gs-mce fork wiki.
- conlododo/moefast (MIT): launcher only, no inference code. Value = the
  sizing math + validation rules (min cache, model/machine mismatch refusal).
  Claimed on 16 GB: decode 60.7 -> 114.8 tok/s with cache on (Qwen3.6-35B).
- Helldez/BigMoeOnEdge (Apache-2.0): llama.cpp public-API wrapper; adaptive
  streaming planner (--auto), any-quant/any-backend, lossless tests. Value =
  planner transparency + SSD streaming design (docs/architecture.md).
- Niko1221/Strata (MIT): docs/HOW_IT_WORKS.md — GPU hot experts + CPU cold
  experts in parallel, pinned RAM, SSD n-gram table, MTP + prompt lookup.
- VecSzn/PolyStrata (MIT): 12 models, two engines; --adapt-async, warm_start
  (read Claude Code system prompt at server start), RAM trimming, per-model
  draft choice (Clover diffusion drafting measured and disabled: net loss).
- spiritbuun/buun-llama-cpp (MIT): VBR KV ladder, tcq codecs, mmproj GPU swap.
- quimmedes/cafe-llama.cpp (MIT): -hmoe/-cmoe/-ssd flag family, TurboQuant
  WHT KV inside fused FA, MTP draft keeps dense Q8_0, n-max 4 recommended.

## Rules
- One technique per commit, gate first, benchmark on quiet machine under
  ~/AI/Rig/bin/machine-lock, numbers into docs/BENCHMARKS.md with versions.
- Drop what does not win on THIS card (PolyStrata's Clover lesson).
- No source copying; if any third-party code is ever copied, NOTICE + license.
- Push/public: Omar only.
