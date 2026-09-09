#!/usr/bin/env bash
# usage: wsl-analyze-batch.sh <out-name> <batch-dir>...
set -uo pipefail
OUT="$1"; shift
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
cd "$REPO" || exit 1
"$HOME/vllm-slo-lab/.venv-slolab/bin/python" scripts/analyze_batch.py "$@" --out "$HOME/vllm-slo-lab/analysis/$OUT" | head -60
echo "rc=$?"; echo "--- tables.md ---"; cat "$HOME/vllm-slo-lab/analysis/$OUT/tables.md" | head -40
