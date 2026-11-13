#!/usr/bin/env bash
# Run a GPU command only while holding the machine lock.
# Usage: tools/gpu-run.sh <tag> <command...>
# take refused -> exit 75 without running the command. take granted -> run, release on any exit.
set -u
TAG="${1:?usage: gpu-run.sh <tag> <command...>}"; shift
LOCK=~/AI/Rig/bin/machine-lock
OUT=$("$LOCK" take lastengine-74 "$TAG" 2>&1); echo "$OUT" | head -1
case "$OUT" in took:*) ;; *) echo "gpu-run: lock not granted, command NOT run" >&2; exit 75 ;; esac
trap '"$LOCK" release lastengine-74 >/dev/null 2>&1' EXIT INT TERM
"$@"
