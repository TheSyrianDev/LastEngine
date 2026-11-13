set -u
cd /home/omar/Projects/LastEngine
M=/home/omar/AI/Models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf
Q=("What is 17 times 3? Answer briefly." "What is the capital of France? Answer briefly." "Name a primary color. Answer briefly." "Say hello in French. Answer briefly.")
ask() { curl -s 127.0.0.1:$1/v1/chat/completions -H 'content-type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":200,\"temperature\":0}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'; }
start() { LE_BATCH=$1 LE_SLOTS=4 LE_NOTUNE=1 target/release/lastengine serve --gpu --model $M --addr 127.0.0.1:$2 --ctx 4096 >/tmp/claude-1000/s4srv_$2.log 2>&1 & echo $!; }
for i in $(seq 1 240); do :; done
P0=$(start 0 8421); for i in $(seq 1 240); do curl -s 127.0.0.1:8421/health | grep -q ok && break; sleep 1; done
for k in 0 1 2 3; do ask 8421 "${Q[$k]}" > /tmp/claude-1000/solo_$k.txt; done
kill $P0; wait $P0 2>/dev/null
P1=$(start 1 8422); for i in $(seq 1 240); do curl -s 127.0.0.1:8422/health | grep -q ok && break; sleep 1; done
for k in 0 1 2 3; do ask 8422 "${Q[$k]}" > /tmp/claude-1000/conc_$k.txt & done; wait
kill $P1; wait $P1 2>/dev/null
for k in 0 1 2 3; do [ -s /tmp/claude-1000/solo_$k.txt ] || echo "req$k EMPTY"; diff -q /tmp/claude-1000/solo_$k.txt /tmp/claude-1000/conc_$k.txt >/dev/null && echo "S4 req$k SAME" || echo "S4 req$k DIFF"; done
