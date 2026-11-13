#!/usr/bin/env bash
# Concurrency bench for LE_BATCH=1 serve at N=1/2/4/8 and llama-server -np 8 at the same N. Run under tools/gpu-run.sh.
cd "$(dirname "$0")/.." || exit 1
M=/home/omar/AI/Models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf
LL=/home/omar/AI/engines/llama-prebuilt-b12182/full/llama-b12182/llama-server
LE_BATCH=1 LE_SLOTS=8 LE_NOTUNE=1 target/release/lastengine serve --gpu --model "$M" --addr 127.0.0.1:8461 --ctx 4096 >/tmp/claude-1000/np8le.log 2>&1 &
P=$!
for i in $(seq 1 240); do curl -s 127.0.0.1:8461/health | grep -q ok && break; sleep 1; done
for n in 1 2 4 8; do echo "LE_BATCH N=$n"; tools/conc_bench.sh 8461 $n 64; done
kill $P; wait $P 2>/dev/null
"$LL" -m "$M" -ngl 99 -c 4096 -np 8 --port 8462 >/tmp/claude-1000/np8ll.log 2>&1 &
P=$!
for i in $(seq 1 240); do curl -s 127.0.0.1:8462/health | grep -q ok && break; sleep 1; done
for n in 1 2 4 8; do echo "llama -np 8 N=$n"; tools/conc_bench.sh 8462 $n 64; done
kill $P; wait $P 2>/dev/null
