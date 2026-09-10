#!/usr/bin/env bash
# Run any W2 driver with its output captured to a timestamped ext4 log, and point
# runs-w2/w2-night-latest.log at it so an already-running watch-night.sh (tail -F) follows along.
# usage: run-logged.sh <script under scripts/wsl> [args...]
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$1"; shift
mkdir -p "$HOME/vllm-slo-lab/runs-w2"
LOG="$HOME/vllm-slo-lab/runs-w2/${SCRIPT%.sh}-$(date +%Y%m%d-%H%M%S).log"
: > "$LOG"
ln -sfn "$LOG" "$HOME/vllm-slo-lab/runs-w2/w2-night-latest.log"
echo "log: $LOG"
bash "$DIR/$SCRIPT" "$@" > "$LOG" 2>&1
echo "exited rc=$? ; log: $LOG"
