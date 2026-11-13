#!/usr/bin/env bash
# Step 2: per-kernel decode_batch profile at N=4 (LE_PROF=1 sync-per-launch; distorts overlap, use for ranking only). Run under tools/gpu-run.sh.
cd "$(dirname "$0")/.." || exit 1
M=/home/omar/AI/Models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf
LE_PROF=1 LE_SLOTS=4 LE_NOTUNE=1 target/release/lastengine batch-gen "$M" "What is 17 times 3?|What is the capital of France?|Name a primary color.|Say hello in French." 64 2>&1 | grep -E "^prof" | head -30
