#!/usr/bin/env bash
# Correctness gate for prefill changes: dense-cmp on two prompts, so one near-tie cannot decide it.
# usage: tools/dense_cmp_gate.sh <model.gguf> [binary]
set -u
M="$1"; B="${2:-./target/release/lastengine}"
for P in "The capital of France is" "Explain in one sentence why the sky is blue."; do
  echo "== prompt: $P"
  "$B" dense-cmp "$M" "$P" 15 2>&1 | tail -1
done
