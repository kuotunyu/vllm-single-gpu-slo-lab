#!/usr/bin/env bash
# Run w2-night.sh with its output captured to a timestamped log on ext4.
# The Windows-side `| tr | grep | tee` pipeline used on 2026-09-10 block-buffered the chain's
# output and lost every line, so the run had to be reconstructed from the manifests. Capture the
# log inside WSL instead and tail it from outside.
# usage: w2-night-logged.sh            (DRY=1 passes through)
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/vllm-slo-lab/runs-w2"
LOG="$HOME/vllm-slo-lab/runs-w2/w2-night-$(date +%Y%m%d-%H%M%S).log"
ln -sfn "$LOG" "$HOME/vllm-slo-lab/runs-w2/w2-night-latest.log"
echo "log: $LOG"
bash "$DIR/w2-night.sh" "$@" > "$LOG" 2>&1
echo "chain exited rc=$? ; log: $LOG"
