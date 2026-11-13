#!/usr/bin/env bash
# Concurrency baseline: N parallel chat requests against one running server; prints wall time and total completion tokens/s.
# Usage: tools/conc_bench.sh <port> <N> [max_tokens]
set -u
P=$1; N=$2; MT=${3:-64}; B=http://127.0.0.1:$P
BODY=$(python3 -c "import json;print(json.dumps({'messages':[{'role':'user','content':'Write a short paragraph about rivers.'}],'max_tokens':$MT,'temperature':0}))")
T0=$(date +%s.%N)
for i in $(seq 1 $N); do curl -s -m 900 -H 'content-type: application/json' -d "$BODY" $B/v1/chat/completions -o /tmp/conc_${P}_$i.json & done
wait
T1=$(date +%s.%N)
python3 - "$N" "$T0" "$T1" "$P" <<'PY'
import json,sys,glob
n,t0,t1=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])
tok=0
for i in range(1,n+1):
    try: tok+=json.load(open(f"/tmp/conc_{sys.argv[4]}_{i}.json"))["usage"]["completion_tokens"]
    except Exception: pass
w=t1-t0
print(f"N={n} wall={w:.2f}s completion_tokens={tok} aggregate={tok/w:.1f} tok/s")
PY
