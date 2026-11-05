# LastEngine

Clean-room LLM inference engine in Rust (CUDA kernels via NVRTC), aimed at MoE models on small GPUs (RTX 3050 4 GB first).

License: PolyForm Noncommercial 1.0.0 (see LICENSE, NOTICE). Source-available, not open source.

## Rules
- No code from the previous LastEngine / RustingStrata tree (kept at ~/Projects/LastEngine-legacy, tag `legacy-final`, used only as a benchmark baseline binary).
- Ideas from other engines come from READMEs, papers and format specs only. See docs/TECHNIQUES.md.
- Every module records the idea's source in its header comment. Benchmark every technique; drop what doesn't win.
- If third-party code is ever copied, keep its license notice and list it in NOTICE.
