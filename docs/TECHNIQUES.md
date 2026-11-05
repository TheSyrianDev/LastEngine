# Techniques survey and provenance

Purpose: record ideas from other engines so LastEngine can re-derive them independently (clean-room).
Rule: read for ideas only; do not copy or transliterate source. Re-implement from the idea, benchmark on our hardware, drop what does not help.
Compiled 2026-11-05 from READMEs only (no source read). Numbers are the authors' own claims, unverified.

## Licenses

| Project | License | Notes |
|---|---|---|
| llama.cpp (ggml-org) | MIT | keep notice if code is copied |
| ik_llama.cpp (ikawrakow) | MIT | LastEngine commit 087e7de says its AVX2 kernels are "after ik_llama.cpp ggml-quants.c": ported-code risk |
| buun-llama-cpp (spiritbuun) | MIT (ggml authors text) | research fork |
| cafe-llama.cpp (quimmedes) | MIT (ggml authors text) | research fork |
| GSllama (therattto) | MIT per README, LICENSE file not checked | Separate project; not gs-mce |
| gs-mce = GenerelSchwerz/llama.cpp, branch `moe-cache` | MIT (c) ggml authors; includes MIT components adapted from Strata (notice kept at ggml/src/ggml-cpu/moe-reference-LICENSE) | CUDA-only host-backed experts with VRAM expert cache (`--moe-expert-cache-mib/-size/-layers/-host-pinned-mb`), grouped decode, cached prefill, expert prefetch, decode overlap, CUDA graph reuse, MTP/draft with own cache, bounded host pinning. Claimed decode 2-3x vs stock on RTX 5070 Ti 16 GB (historical). Upstream cousin: ggml-org PR #29887 (LRU, batches <=32, 1.6-2.2x on 4090/5090). Eviction/slot details unread: study its wiki before designing ours. |
| Strata (Niko1221) | MIT | vendors ggml in third_party/ |
| PolyStrata (VecSzn / simplexai-labs) | MIT | reuses Strata engine + second engine for 11 other models |
| HyperQwen (syv-ai) | Apache-2.0 | patch series on vLLM 0.32.0, Python |
| RustingStrata (RustingStudio) | RustingStrata License 1.0 (custom, not OSI) | LastEngine's base. s3 is share-alike: derivatives must stay under the same license, name Vasyl Trefilov as original author, describe modifications. s2: no reselling the engine itself. s5: terminates on violation (30-day cure) |

Model weights carry separate licenses (e.g. LFM2.5: commercial use only under $10M revenue).

## Techniques

Status: `have` = LastEngine already does something similar (check src), `idea` = candidate to re-derive.

### Memory tiering and MoE expert placement
- Experts in pinned host RAM, hot experts cached in spare VRAM, misses computed on CPU concurrently with GPU (Strata, PolyStrata, RustingStrata, cafe `-hmoe`, buun expert cache). Claim (Strata): each extra GB VRAM holds ~700 more experts. Status: have (`expert_cache.rs`, `cpu_moe.rs`).
- Persist expert heat across runs, prewarm at start (cafe `~/.cache/llama.cpp/moe-direct`, buun heatmap). Status: check.
- Re-learn cache contents during use; async cache update on a helper thread: +4-5% decode (PolyStrata `--adapt-async`). Status: idea.
- Stream next layer's experts over PCIe while current layer's attention runs (Strata prefill, chunks up to 8192). Status: check.
- SSD expert streaming via mmap of only selected experts (cafe `-ssd`); model used in place from SSD, no second copy (PolyStrata). Status: idea.
- Multi-GB n-gram/PLE table on SSD, a few rows per token, prefetched for draft tokens (Strata, cafe `--ngram-ssd`: verify round 45.9-50.9 -> 44.2-48.6 ms). Status: model-specific.
- RAM trimming: stop holding file pages already on GPU (PolyStrata: Qwen3.6 17.9 -> 13.1 GiB). Status: idea.
- llama.cpp upstream: `--n-cpu-moe`, `-ot` regex tensor placement, `--moe-cache-mib`, `--op-offload`, `-fit` auto-fit to VRAM. Status: reference for CLI semantics.

### Speculative decoding
- Model's own MTP head drafts N tokens, one verify pass (RustingStrata 118 -> up to 160 tok/s, Strata 1.6-1.8x, cafe 53-58 t/s). Status: have (`--mtp`).
- Prompt-lookup / n-gram drafts from context (Strata +6-11% on code edits; HyperQwen DFlash2 381 vs 127 tok/s when answer quotes prompt). Status: idea.
- cafe MTP detail: draft keeps dense weights Q8_0, quantizes only routed experts, shares target embeddings/LM head; >=7 tokens per verify makes CUDA switch kernels and halves decode; n-max 4 recommended. Status: idea.
- EAGLE-style draft, tree drafting (cafe: tree slower than chain on a 3090). Status: low value.
- Per-model draft choice (PolyStrata: Clover diffusion drafting measured and disabled as net loss). Lesson: measure per model.
- Note: LOOP_LOG #88 already concluded lookahead is not viable on DeltaNet.

### KV cache
- q8 KV (RustingStrata: 345 vs 650 MB at 32K, ~6% slower decode). Status: have (`q8`, `tcq3`).
- TCQ trellis-coded KV: FWHT rotation + Viterbi on 512-state trellis, 3.25 bpv, KLD ~40% below scalar turbo3 (buun). Cost: prefill 1015 -> 818 tok/s. Status: have `tcq3` (verify provenance).
- VBR: start f16, degrade one (layer, side) tensor at a time under memory pressure along a per-model KLD-ordered ladder (buun). Status: idea.
- Asymmetric K/V bit widths (buun: K tcq3 / V tcq2 beats reverse by 15-17% KLD). Status: idea.
- Hadamard transform before KV quant (ik_llama PRs 1033/1034/1527; cafe TurboQuant WHT inside fused FA). Status: idea.
- Long context: only most-read KV stays in VRAM from 64K, rest streams from RAM (Strata). Status: idea.
- Small-KV architectures (DeepSeek V4 6,880 B/token; ring windows) are model features, not engine tricks.
- Compute buffers sized for the chunk window, not full context (GSllama: worst-case graph reserve was ~5x KV). Status: check.

### Prefill
- Larger ubatch matters hugely (GSllama: `-ub 2048` vs 512 gave 2.28x prefill). Status: have (`--chunk`, `calibrate`).
- Per-machine calibration of chunk size / drafts, cached (RustingStrata `calibrate`). Status: have.
- INT8 tensor-core activations for prefill GEMM (HyperQwen 1.44k -> 1.85-1.94k tok/s at 1k; sm86). Status: idea; we already have an MMQ path.
- Fused delta-net kernels for Qwen3-Next/3.5 (ik_llama PRs 1315/1333/1362/1373); keep recurrent state in 16-bit (HyperQwen). Status: check.
- PCIe topology outweighs software (GSllama: x4 chipset -> x16 CPU slot, 306 -> 900 tok/s). Operational advice, not code.
- First device in `-dev` computes CPU-resident experts, not tensor-split (GSllama).

### Kernels and quantization
- NVRTC run-time compile with cubin cache, no C++ build (RustingStrata). Status: have.
- Vulkan backend from the same kernels as GLSL->SPIR-V (RustingStrata). Status: idea (AMD/Intel reach).
- ik_llama quant families: IQ*_K, IQ*_KS, trellis IQ*_KT, row-interleaved _R4/_R8/_R16, Bitnet IQ1_BN/IQ2_BN, MXFP4. Re-derive from the format description, not from ggml-quants.c. Status: ours are possibly derived: AUDIT.
- Row-interleaved repack for CPU speed, but it pins matmuls to CPU when no CUDA kernel exists (ik_llama `-rtr` warning).
- Fusions: FFN, norm+scale folding (GSllama removed 104 ops, +2.8% decode on Vulkan), but hc_combine fusion gave +0.3% decode, -2.5% prefill. Lesson: profile first. LOOP_LOG #88 found norm/router fusion < 1% on the 3050.
- Int8 Marlin GEMM layer selection with tuned tables (HyperQwen). Status: low relevance for GGUF.
- Persistent CUDA graph reuse across steps (GSllama: 72,817 reuses, 0 reallocs). Status: check.

### Server / integration
- OpenAI + Anthropic compatible API, tools, prefix reuse via cached_tokens (RustingStrata, Strata, cafe). Status: have (`serve.rs`).
- Vision encoder GPU swap with drafter unloaded during image encode (buun `--mmproj-gpu-swap`: 30-60 s CPU -> 1-2 s GPU). Status: idea (we have `vision.rs`).
- Safetensors / EXL3 / FP8 loading without GGUF conversion (cafe, buun). Status: idea.
- GGUF format: documented in ggml docs/gguf.md (magic GGUF, v3, u64 counts, alignment 32 default, tensor offsets relative to data section). Implement `gguf.rs` from the spec.
- Engine stall sentinel, SSE keep-alive (HyperQwen). Status: small ops wins.

## Open items
1. gs-mce identified (GenerelSchwerz moe-cache). Next: read its wiki pages (MoE-Cache, Prefetch, Grouped-MoE-Decode) for eviction and slot design.
2. Clean-room rewrite plan vs the RustingStrata-derived code (see below).
3. Audit files for ported third-party code: `src/gguf.rs`, `src/quant/mod.rs`, `src/cpu_moe.rs`, `src/cuda/kernels.cu`.
4. Technique READMEs only; mechanisms live in docs/PRs/papers (Strata `docs/HOW_IT_WORKS.md`, `docs/paper/Strata-Paper.pdf`, ik_llama wiki, HyperQwen `docs/optimizations.md`).

## Provenance of the current tree
- `LICENSE.md` is "LastEngine License 1.0" (Vasyl Trefilov), a renaming of RustingStrata License 1.0. The tree is a RustingStrata derivative and the upstream s3 share-alike term applies until each file is rewritten.
- Git history starts 2026-10-16 (55 commits), a snapshot, not upstream history.
