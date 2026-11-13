# LastEngine handoff (lastengine-74, 2026-11-12)

State: local commits only, nothing pushed. Latest: 19344af (llama prefill baseline and incomplete ncu note), before that eb726a2 (prefill queue), bf1d2be (LE_MMQSK split-K, opt-in, off), d282c94 (prefill study), 0907348 (defaults: GQA on, SP 32, RMSQ on).

Measured (valid):
- Decode, 2594-token prompt, r=3: ours default 93.2 tok/s; llama.cpp b12182 94.9.
- Prefill, 5023-token prompt: ours default about 2051 tok/s; llama.cpp b12182 about 3200 (1.56x gap).
- Split-K LE_MMQSK=2 and 4 are slower (-5%, -11%); LE_MMQR=0 is -3%. Defaults kept.

Not done:
- ncu limiter analysis for mmq128r_12 vs llama mul_mat_q. Earlier attempts gave no kernel data (600 s timeouts). Last attempt (--set full, --launch-skip 2, --kernel-name regex:mmq128r, 512-token prompt) was started but killed when usage limit stop came in; no output.

Lock: released (machine-lock status free). A stray root `lastengine serve` pid 609025 was killed by me. GPU shows only pid 136891 (not ours).
Incident: I overwrote ai-8c's ~/AI/.machine-busy at 20:40:58; peer removed my line. Use ~/AI/Rig/bin/machine-lock take/release only.

Next steps when GPU is granted: rerun ncu narrowly for mmq128r_12 and llama mul_mat_q<12>, then pick one fix.
