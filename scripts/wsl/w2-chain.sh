#!/usr/bin/env bash
# W2 measurement chain for one user-declared GPU-quiet window (ADR 0006 §決策 5):
#   1. FP8 closed-loop re-run of c = 1..96 with a 300-request warm-up (sufficiency check)
#   2. FP8 open-loop Poisson sweep, 7 rates x seeds 1..3 (one server session per seed)
#   3. TMMLU+ three frozen slices against FP8
# Each batch gets its own io-pressure sampler and a quick analysis print (suspect flags included).
# usage: bash scripts/wsl/w2-chain.sh [closed|open|tmmlu|all]   (default all)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
ROOT="$HOME/vllm-slo-lab/runs-w2"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
MODEL="Qwen/Qwen3-8B-FP8"
R_SAT=43.67
STEP="${1:-all}"
mkdir -p "$ROOT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

run_batch() {  # <label> <run_root> <warmup> <seed> <stage specs...>
  local label="$1" root="$2" warmup="$3" seed="$4"; shift 4
  local batch="$root/fp8/seed-$seed"
  mkdir -p "$batch"
  bash "$WSL/io-sampler.sh" "$batch" >/dev/null 2>&1 &
  local sampler=$!
  log "START $label seed=$seed warmup=$warmup stages=$*"
  WARMUP="$warmup" MAX_NUM_SEQS=256 RUN_ROOT="$root" bash "$WSL/batch.sh" fp8 "$MODEL" "$seed" "" "$@"
  local rc=$?
  kill "$sampler" 2>/dev/null
  log "END $label seed=$seed rc=$rc"
  "$LABPY" "$REPO/scripts/analyze_batch.py" "$batch" --out "$root/analysis-seed-$seed" 2>&1 \
    | grep -E '"r_sat_rps"|"r_sat_is_lower_bound"|"capacity_C"|"suspect_concurrencies_excluded"|"r_slo"|"suspect_rates_excluded"' | tr -d '\n'; echo
  return "$rc"
}

wait_ready() {  # <serve.log> <pid>
  for _ in $(seq 1 120); do
    grep -q "Application startup complete" "$1" 2>/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

run_tmmlu() {
  local out="$ROOT/tmmluplus/fp8"
  mkdir -p "$out"
  cd "$HOME/vllm-slo-lab" || return 1
  "$LABCLI" quiet-gpu --out "$out/quiet_gpu.json" || { log "TMMLU quiet-gpu refused"; return 2; }
  pkill -f ".venv/bin/vllm serve" 2>/dev/null; sleep 3
  .venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8013 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --max-num-seqs 256 --max-num-batched-tokens 2048 > "$out/serve.log" 2>&1 &
  local pid=$!
  wait_ready "$out/serve.log" "$pid" || { log "TMMLU server not ready"; kill "$pid" 2>/dev/null; return 1; }
  log "TMMLU server ready"
  cd "$REPO" || return 1
  for s in 1 2 3; do
    "$LABPY" scripts/tmmluplus_eval.py --slice "eval/tmmluplus/slice-$s.jsonl" --base-url http://127.0.0.1:8013 \
      --model "$MODEL" --out "$out/slice-$s.json" --concurrency 16 2>&1 | tail -2
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  log "TMMLU done -> $out"
}

if [ "$STEP" = all ] || [ "$STEP" = closed ]; then
  run_batch closed-loop-rerun "$ROOT/closed-loop-rerun" 300 1 \
    cl:1:90 cl:2:200 cl:4:400 cl:8:780 cl:16:1550 cl:32:2800 cl:64:4700 cl:96:5800
fi
if [ "$STEP" = all ] || [ "$STEP" = open ]; then
  rates=""
  for m in 0.25 0.5 0.75 1.0 1.25 1.5 2.0; do
    rates="$rates ol:$(awk -v r=$R_SAT -v m=$m 'BEGIN{printf "%.2f", r*m}'):300"
  done
  for seed in 1 2 3; do
    # shellcheck disable=SC2086
    run_batch open-loop "$ROOT/open-loop" 100 "$seed" $rates
  done
fi
if [ "$STEP" = all ] || [ "$STEP" = tmmlu ]; then
  run_tmmlu
fi
log "CHAIN DONE ($STEP)"
