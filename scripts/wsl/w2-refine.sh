#!/usr/bin/env bash
# Open-loop grid refinement (ADR 0008): the frozen grid {0.25..2.0} x r_sat jumps from attainment 1.000
# at 21.8 rps to 0.103 at 32.75 rps, so r_SLO needs points in between. Adds 0.55/0.60/0.65/0.70 x 43.67
# = {24.02, 26.20, 28.39, 30.57} rps. For each seed the batch runs every rate listed in RATES that has no
# manifest yet (so a killed chain is resumed, not repeated), ascending, in one server session.
# usage: bash scripts/wsl/w2-refine.sh <seed>... ; env RATES overrides the rate list.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
ROOT="$HOME/vllm-slo-lab/runs-w2/open-loop"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
MODEL="Qwen/Qwen3-8B-FP8"
RATES="${RATES:-10.92 21.84 24.02 26.20 28.39 30.57 32.75 43.67 54.59 65.50 87.34}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
for seed in "$@"; do
  batch="$ROOT/fp8/seed-$seed"
  mkdir -p "$batch"
  specs=""
  for r in $RATES; do
    [ -f "$batch/ol-rate-$r/manifest.json" ] || specs="$specs ol:$r:300"
  done
  if [ -z "$specs" ]; then log "seed $seed: nothing missing"; continue; fi
  bash "$WSL/io-sampler.sh" "$batch" >/dev/null 2>&1 &
  sampler=$!
  log "START open-loop seed=$seed stages=$specs"
  # shellcheck disable=SC2086
  WARMUP=100 MAX_NUM_SEQS=256 RUN_ROOT="$ROOT" bash "$WSL/batch.sh" fp8 "$MODEL" "$seed" "" $specs
  rc=$?
  kill "$sampler" 2>/dev/null
  log "END open-loop seed=$seed rc=$rc"
  "$LABPY" "$REPO/scripts/analyze_batch.py" "$batch" --out "$ROOT/analysis-seed-$seed" 2>&1 \
    | grep -E '"r_slo"|"suspect_rates_excluded"' | tr -d '\n'; echo
done
log "REFINE DONE"
