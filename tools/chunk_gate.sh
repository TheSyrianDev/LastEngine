#!/usr/bin/env bash
# Step-1 gate: LE_BATCH=1 with chunked prefill (LE_CHUNK=8, budget 16) and 4 concurrent greedy requests must equal solo LE_BATCH=0 outputs.
# Needs solo reference files /tmp/claude-1000/solo_{0..3}.txt (from tools/s4_concurrent_gate.sh). Run under tools/gpu-run.sh.
cd "$(dirname "$0")/.." || exit 1
M=/home/omar/AI/Models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf
Q=("What is 17 times 3? Answer briefly." "What is the capital of France? Answer briefly." "Name a primary color. Answer briefly." "Say hello in French. Answer briefly.")
LE_BATCH=1 LE_CHUNK=8 LE_BUDGET=16 LE_SLOTS=4 LE_NOTUNE=1 target/release/lastengine serve --gpu --model "$M" --addr 127.0.0.1:8471 --ctx 4096 >/tmp/claude-1000/chunk_srv.log 2>&1 &
P=$!
for i in $(seq 1 240); do curl -s 127.0.0.1:8471/health | grep -q ok && break; sleep 1; done
ask() { curl -s 127.0.0.1:8471/v1/chat/completions -H 'content-type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":200,\"temperature\":0}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'; }
PIDS=(); for k in 0 1 2 3; do ask "${Q[$k]}" > /tmp/claude-1000/chunk_$k.txt & PIDS+=($!); done; wait "${PIDS[@]}"
kill $P; wait $P 2>/dev/null
rc=0
for k in 0 1 2 3; do if diff -q /tmp/claude-1000/solo_$k.txt /tmp/claude-1000/chunk_$k.txt >/dev/null; then echo "chunk req$k SAME"; else echo "chunk req$k DIFF"; rc=1; fi; done
exit $rc
