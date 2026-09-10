#!/usr/bin/env bash
# GPU follow-up after w2-night.sh (plan 2026-09-10, run log "Queued"):
#   1. FP8 TMMLU+ full set, so every cell's quality is a 19,680-item estimate
#   2. open-loop knee refinement at 0.80 / 0.85 / 0.90 x r_sat for AWQ, GPTQ and BF16: all three
#      knees fell in the frozen 0.75 -> 1.0 gap, while FP8's was already resolved to ~8 % by the
#      0.55 - 0.70 points of ADR 0008. Seeds 1-3 each, so r_SLO keeps its all-seeds rule.
# Everything is skip-if-present, so re-running resumes. Output goes to the same ext4 log style.
set -uo pipefail
WSL="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[$(date +%H:%M:%S)] [followup] $*"; }
log "START fp8 full set"
bash "$WSL/tmmlu-only.sh" fp8 Qwen/Qwen3-8B-FP8 256
log "START awq refinement"
bash "$WSL/refine-cell.sh" awq Qwen/Qwen3-8B-AWQ 256 30.15 "0.80 0.85 0.90" 1 2 3
log "START gptq refinement"
bash "$WSL/refine-cell.sh" gptq JunHowie/Qwen3-8B-GPTQ-Int4 256 30.26 "0.80 0.85 0.90" 1 2 3
log "START bf16 refinement"
bash "$WSL/refine-cell.sh" bf16 Qwen/Qwen3-8B 40 11.40 "0.80 0.85 0.90" 1 2 3
log "FOLLOWUP DONE"
