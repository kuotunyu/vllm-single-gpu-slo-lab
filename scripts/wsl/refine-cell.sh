#!/usr/bin/env bash
# Add open-loop rates to a cell that already has its main grid, so the SLO knee is resolved to a
# comparable relative precision across cells (ADR 0008 did this for FP8; the frozen multipliers
# leave a 0.75 -> 1.0 gap that straddles the knee for faster-saturating cells).
#
# Only rates without a manifest are run, so this is also the resume path for an interrupted cell.
#
# usage: refine-cell.sh <cell> <model> <max_num_seqs> <r_sat> "<extra multipliers>" [seeds...]
#   e.g. refine-cell.sh awq Qwen/Qwen3-8B-AWQ 256 30.15 "0.85 0.95" 1 2 3
#   env: GPU_MEM_UTIL (0.82), MAX_BATCHED_TOKENS (2048)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0
CELL="$1"; MODEL="$2"; MAX_NUM_SEQS="$3"; R_SAT="$4"; MULTIPLIERS="$5"; shift 5
SEEDS="${*:-1 2 3}"
# The refinement is a second server session in the same batch dir; tag its session-level logs so
# promotion keeps the main run's vllm.log / quiet_gpu.json (see promote-w2.sh).
TAG="refine-$(echo "$MULTIPLIERS" | awk '{print $1}')"
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
ROOT="$HOME/vllm-slo-lab/runs-w2/open-loop-cells"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
log() { echo "[$(date +%H:%M:%S)] [$CELL refine] $*"; }

for seed in $SEEDS; do
  dir="$ROOT/$CELL/seed-$seed"
  specs=""
  for m in $MULTIPLIERS; do
    rate=$(awk -v r="$R_SAT" -v m="$m" 'BEGIN{printf "%.2f", r*m}')
    [ -f "$dir/ol-rate-$rate/manifest.json" ] || specs="$specs ol:$rate:300"
  done
  if [ -z "$specs" ]; then log "seed $seed: nothing missing"; continue; fi
  mkdir -p "$dir"
  bash "$WSL/io-sampler.sh" "$dir" >/dev/null 2>&1 &
  sampler=$!
  log "START seed=$seed stages=$specs"
  # shellcheck disable=SC2086
  WARMUP=100 MAX_NUM_SEQS="$MAX_NUM_SEQS" RUN_ROOT="$ROOT" bash "$WSL/batch.sh" "$CELL" "$MODEL" "$seed" "" $specs
  rc=$?
  kill "$sampler" 2>/dev/null
  log "END seed=$seed rc=$rc"
done
"$LABPY" "$REPO/scripts/analyze_batch.py" "$ROOT/$CELL" --out "$ROOT/$CELL/analysis" >/dev/null 2>&1
"$LABPY" "$WSL/read_r_sat.py" "$ROOT/$CELL/analysis/open_loop.json" "$CELL" r_slo
bash "$WSL/promote-w2.sh" "$ROOT/$CELL/seed-1" "$CELL/open-loop/seed-1" "$TAG" 2>&1 | grep -v "redact: 0"
for seed in $SEEDS; do
  [ "$seed" = 1 ] && continue
  bash "$WSL/promote-w2.sh" "$ROOT/$CELL/seed-$seed" "$CELL/open-loop/seed-$seed" "$TAG" 2>&1 | grep -v "redact: 0"
done
log "REFINE DONE"
