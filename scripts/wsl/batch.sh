#!/usr/bin/env bash
# Batch driver for one engine cell on the 4090 (WSL2): quiet-GPU gate -> server -> stages -> stop.
# usage: wsl-batch.sh <cell> <model> <seed> <extra vllm flags or ''> <stage spec>...
#   stage spec: ol:<rate_rps>:<duration_s>   (open-loop Poisson)
#               cl:<concurrency>:<num_requests> (closed-loop)
#   env: WARMUP (default 100), RUN_ROOT (default ~/vllm-slo-lab/runs), MAX_NUM_SEQS (default 64)
set -uo pipefail
# open-loop at 2x r_sat keeps ~13k requests in flight; the default 1024 fds would turn them into client errors
ulimit -n 65536 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
CELL="$1"; MODEL="$2"; SEED="$3"; EXTRA="$4"; shift 4
WARMUP="${WARMUP:-100}"; RUN_ROOT="${RUN_ROOT:-$HOME/vllm-slo-lab/runs}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
IPF="$HOME/vllm-slo-lab/.venv-loadgen/bin/inference-perf"
BATCH="$RUN_ROOT/$CELL/seed-$SEED"
mkdir -p "$BATCH"
cd "$HOME/vllm-slo-lab"
pkill -f ".venv/bin/vllm serve" 2>/dev/null; pkill -f "EngineCore" 2>/dev/null; sleep 3
# quiet-GPU gate before the card is taken (WSL2: memory + utilization criteria)
"$LABCLI" quiet-gpu --out "$BATCH/quiet_gpu.json" || { echo "QUIET_GPU_REFUSED"; cat "$BATCH/quiet_gpu.json" | head -40; exit 2; }
# shellcheck disable=SC2086
.venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8013 --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 2048 $EXTRA > "$BATCH/serve.log" 2>&1 &
PID=$!
for i in $(seq 1 120); do
  curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8013/v1/models | grep -q 200 && break
  kill -0 $PID 2>/dev/null || { echo "SERVER_EXITED_EARLY"; grep -i "error" "$BATCH/serve.log" | tail -3 | cut -c1-200; exit 1; }
  sleep 5
done
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8013/v1/models | grep -q 200 || { echo "SERVER_NOT_READY"; kill $PID; exit 1; }
echo "server ready ($CELL seed $SEED): $(grep -o 'Loading weights took [0-9.]* seconds' "$BATCH/serve.log" | tail -1) | $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$BATCH/serve.log" | tail -1)"
FLAGS_JSON=$(printf '{"model":"%s","max_model_len":4096,"gpu_memory_utilization":0.90,"max_num_seqs":%s,"max_num_batched_tokens":2048,"extra":"%s"}' "$MODEL" "$MAX_NUM_SEQS" "$EXTRA")
first=1
for spec in "$@"; do
  kind="${spec%%:*}"; rest="${spec#*:}"; a="${rest%%:*}"; b="${rest#*:}"
  wu=$WARMUP; [ $first -eq 0 ] && wu=20   # full warm-up once per server, short re-warm between stages
  first=0
  if [ "$kind" = "ol" ]; then
    dir="$BATCH/ol-rate-$a"
    "$LABCLI" run-stage --run-dir "$dir" --cell "$CELL" --model "$MODEL" --kind open_loop --seed "$SEED" --rate-rps "$a" --duration-s "$b" --warmup-requests "$wu" --inference-perf-bin "$IPF" --engine-flags "$FLAGS_JSON"
  else
    dir="$BATCH/cl-conc-$a"
    "$LABCLI" run-stage --run-dir "$dir" --cell "$CELL" --model "$MODEL" --kind closed_loop --seed "$SEED" --concurrency "$a" --num-requests "$b" --warmup-requests "$wu" --inference-perf-bin "$IPF" --engine-flags "$FLAGS_JSON"
  fi
  echo "stage $spec exit=$?"
done
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "batch done -> $BATCH"
