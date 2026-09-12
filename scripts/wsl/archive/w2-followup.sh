#!/usr/bin/env bash
# After the first chain: open-loop refinement/resume for seeds 1-3 (ADR 0008), then TMMLU+ slices.
set -uo pipefail
WSL="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl"
bash "$WSL/w2-refine.sh" 1 2 3
bash "$WSL/w2-chain.sh" tmmlu
echo "[$(date +%H:%M:%S)] FOLLOWUP DONE"
