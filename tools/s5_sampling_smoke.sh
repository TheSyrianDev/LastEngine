#!/usr/bin/env bash
# S5 smoke: LE_BATCH=1 sampled chat (temperature 0.8, top_p 0.9) returns non-empty output; run under tools/gpu-run.sh.
cd "$(dirname "$0")/.." || exit 1
M=/home/omar/AI/Models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf
LE_BATCH=1 LE_SLOTS=4 LE_NOTUNE=1 target/release/lastengine serve --gpu --model "$M" --addr 127.0.0.1:8451 --ctx 4096 >/tmp/claude-1000/s5srv.log 2>&1 &
P=$!
for i in $(seq 1 240); do curl -s 127.0.0.1:8451/health | grep -q ok && break; sleep 1; done
for k in 1 2; do curl -s 127.0.0.1:8451/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Name a fruit."}],"max_tokens":200,"temperature":0.8,"top_p":0.9}' | python3 -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["message"]["content"][:120]))'; done
kill $P; wait $P 2>/dev/null
