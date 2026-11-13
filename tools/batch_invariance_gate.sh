#!/usr/bin/env bash
# Batch-invariance gate: each prompt's greedy token stream must be identical at N=1 and at N=2/4/6 slots.
# Usage: tools/batch_invariance_gate.sh MODEL.gguf   (take machine-lock first; needs CUDA)
set -u
M="$1"; BIN="${BIN:-target/release/lastengine}"; NTOK="${NTOK:-20}"
PS=("The capital of France is" "Explain in one sentence why the sky is blue." "Name three rivers in Europe." "Write a haiku about rain." "What is 17 times 3?" "Say hello in French.")
T=$(mktemp -d); fail=0
for i in 0 1 2 3 4 5; do
  LE_PRINT_TOK=1 LE_SLOTS=1 LE_ROWS=1 "$BIN" batch-loop "$M" "${PS[$i]}" "$NTOK" 2>/dev/null | grep "toks 0:" | cut -d: -f2- > "$T/n1_$i"
done
for N in 2 4 6; do
  P=$(IFS='|'; echo "${PS[*]:0:$N}")
  LE_PRINT_TOK=1 LE_SLOTS=$N LE_ROWS=1 "$BIN" batch-loop "$M" "$P" "$NTOK" 2>/dev/null | grep "toks " > "$T/xn_$N"
  for ((i=0; i<N; i++)); do
    b=$(grep "toks $i:" "$T/xn_$N" | cut -d: -f2-)
    if [ "$b" = "$(cat "$T/n1_$i")" ]; then echo "N=$N slot$i identical"; else echo "N=$N slot$i DIFF"; fail=1; fi
  done
done
rm -rf "$T"; [ $fail = 0 ] && echo "BATCH INVARIANCE OK" || echo "BATCH INVARIANCE FAIL"; exit $fail
