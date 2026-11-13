# Last Engine on Apple Silicon: options

Status: research only, no Apple code yet. Numbers below come from third-party write-ups, not from our runs; none are matched to our Q4_K_M MiniCPM5 setup.

## Runners on Apple Silicon

- **llama.cpp (Metal backend).** Our baseline engine family. Prefill is competitive; in one 2026 comparison on Llama 3 8B Q4 it trailed MLX on generation (65.8 vs 71.3 tok/s, different builds). Already the fallback in the app.
- **mlx-lm.** Apple's MLX Python runner. Recent reports (M3 Ultra, M4 Pro) show faster decode than llama.cpp at short context (e.g. Qwen3-Coder-30B-A3B about 130 tok/s on MLX vs 43 on Ollama's llama.cpp backend; one raw llama.cpp M4 Max run reached 89.4 tok/s, so the gap partly reflects Ollama). At 30K+ context one community report found MLX about 50% slower than llama.cpp with flash attention (mlx-lm issue #763). Prefill is roughly tied at very long prompts.
- **mlx-engine / "ultrafast" MLX runners.** Wrappers around the same MLX kernels; no independent numbers we could verify. Treat their claims as unverified.

Sources:
- https://yage.ai/share/mlx-apple-silicon-en-20260331.html
- https://vettedconsumer.com/mlx-vs-llama-cpp-on-a-mac-which-is-faster-and-when-the-answer-flips/
- https://pub.towardsai.net/apples-mlx-runs-local-llms-3x-faster-than-llama-cpp-until-your-context-hits-40k-715ec441afbb
- https://contracollective.com/blog/llama-cpp-metal-vs-mlx-backend-apple-silicon-2026
- https://arxiv.org/pdf/2511.05502

## Neural Engine (ANE)

- ANE is reachable only through CoreML. ANEMLL converts models to CoreML for ANE.
- Real reported numbers are low for 8B-class models: one M4 Max user measured about 9.3 tok/s on an 8B R1 distill via ANEMLL vs about 31 tok/s on MLX 8-bit (2025, older ANEMLL). A 2026 guide lists 47–62 tok/s for Llama 3.2 1B on ANE, vs about 204 tok/s for MLX on GPU for the same size.
- CoreML-LLM reports about 52 tok/s and another 2026 benchmark about 32 tok/s on ANE decoding. Both are unverified single reports.
- What runs on ANE vs GPU: ops ANE does not support fall back to CPU, copying through IOSurface, which costs 2–4x on small ops. Power is much lower (about 2 W vs 20 W cited for GPU).
- Conversion is per-model and per-context-length; there is no ANE path for our Q4_K block formats, so models would need re-export.

Sources:
- https://github.com/vakharwalad23/Anemone
- https://insiderllm.com/guides/apple-neural-engine-llm-inference/
- https://github.com/john-rocky/apple-silicon-llm-bench
- https://pradeep.md/2026/03/30/apple-neural-engine-local-llm-agents.html

## Recommendation

1. **Native Metal backend** (preferred for us). Keeps our Q4_K_M format and our engine, matches the CUDA design (flash attention, quantized GEMM). Cost: a full kernel set. Reason: we already have a working model and correctness gates; the MLX path would need a second model format.
2. **Wrap MLX.** Fastest path to decode numbers on Apple, but it abandons our engine on that platform and needs its own model conversion. Use only as an interim fallback if we need Apple users before a Metal backend.
3. **ANE.** Not recommended now. Reported throughput is below GPU for 8B-class models, conversion is per model, and it is not useful for our Q4_K format.

Next step: measure llama.cpp Metal and mlx-lm on one matched Apple machine before choosing between 1 and 2.
