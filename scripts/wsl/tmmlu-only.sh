#!/usr/bin/env bash
# TMMLU+ for one cell without the serving sweep: the three slices and the full set, each only if
# its output is missing, then promote into evidence/raw/w2/<cell>/tmmluplus/.
# Used for FP8, whose slices predate eval/tmmluplus/full.jsonl (2026-09-09), so the four-precision
# quality column compares full sets rather than a +-0.065 slice estimate against +-0.007 ones.
# usage: tmmlu-only.sh <cell> <model> <max_num_seqs>
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0
CELL="$1"; MODEL="$2"; MAX_NUM_SEQS="$3"
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
out="$HOME/vllm-slo-lab/runs-w2/tmmluplus/$CELL"
mkdir -p "$out"
log() { echo "[$(date +%H:%M:%S)] [$CELL tmmlu-only] $*"; }

need=0
for s in 1 2 3; do [ -f "$out/slice-$s.json" ] || need=1; done
[ -f "$out/full.json" ] || need=1
if [ "$need" = 0 ]; then log "slices and full set already present"; else
  cd "$HOME/vllm-slo-lab" || exit 1
  "$LABCLI" quiet-gpu --out "$out/quiet_gpu-full.json" || { log "QUIET_GPU_REFUSED"; exit 2; }
  pkill -f ".venv/bin/vllm serve" 2>/dev/null; pkill -f "EngineCore" 2>/dev/null; sleep 3
  .venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8013 --max-model-len 4096 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 2048 > "$out/serve-full.log" 2>&1 &
  pid=$!
  # readiness by command substitution, never `curl | grep -q` under pipefail (see batch.sh)
  READY=0
  for _ in $(seq 1 180); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8013/v1/models 2>/dev/null || true)
    if [ "$CODE" = "200" ]; then READY=1; break; fi
    kill -0 "$pid" 2>/dev/null || { log "SERVER_EXITED_EARLY"; exit 1; }
    sleep 5
  done
  [ "$READY" = 1 ] || { log "SERVER_NOT_READY"; kill "$pid"; exit 1; }
  log "server ready: $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$out/serve-full.log" | tail -1)"
  cd "$REPO" || exit 1
  for s in 1 2 3; do
    [ -f "$out/slice-$s.json" ] || "$LABPY" scripts/tmmluplus_eval.py --slice "eval/tmmluplus/slice-$s.jsonl" \
      --base-url http://127.0.0.1:8013 --model "$MODEL" --out "$out/slice-$s.json" --concurrency 16 2>&1 | tail -1
  done
  [ -f "$out/full.json" ] || "$LABPY" scripts/tmmluplus_eval.py --slice eval/tmmluplus/full.jsonl \
    --base-url http://127.0.0.1:8013 --model "$MODEL" --out "$out/full.json" --concurrency 32 2>&1 | tail -1
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
fi

T="$REPO/evidence/raw/w2/$CELL/tmmluplus"
mkdir -p "$T"
for f in slice-1.json slice-2.json slice-3.json full.json quiet_gpu-full.json; do
  [ -f "$out/$f" ] && sed -e "s#$HOME#~#g" "$out/$f" > "$T/$f"
done
[ -f "$out/serve-full.log" ] && sed -e "s#$HOME#~#g" "$out/serve-full.log" \
  | "$LABPY" "$REPO/scripts/redact.py" redact - -o "$T/vllm-full.log" 2>/dev/null
log "TMMLU DONE"
