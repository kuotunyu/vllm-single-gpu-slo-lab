#!/usr/bin/env bash
# Batch driver for one engine cell on the 4090 (WSL2): quiet-GPU gate -> server -> [shim] -> stages -> stop.
# usage: batch.sh <cell> <model> <seed> <extra vllm flags or ''> <stage spec>...
#   stage spec: ol:<rate_rps>:<duration_s>[:<seed>]     (open-loop Poisson)
#               cl:<concurrency>:<num_requests>[:<seed>] (closed-loop)
#   A stage's own seed (W4: the three open-loop seeds share one server session, ADR 0015 item 3)
#   writes under $RUN_ROOT/$CELL/seed-<stage seed>/; the session's serve.log, shim.log and
#   quiet_gpu.json stay in the session seed's directory ($RUN_ROOT/$CELL/seed-<seed>).
#   env: WARMUP (default 100), REWARM (20), RUN_ROOT (default ~/vllm-slo-lab/runs), MAX_NUM_SEQS (default 64),
#        GPU_MEM_UTIL (0.82), MAX_BATCHED_TOKENS (2048),
#        SHIM=1        every stage goes through the passthrough admission shim on 8021 (W4, ADR 0015 item 7)
#        SPECDEC / FAMILY   labels copied into every manifest (W4)
#        FAKE_ENGINE=1 CPU dry run: fake_vllm.py instead of vLLM, no quiet-GPU gate (FAKE_SLOTS, FAKE_TOKEN_MS)
set -uo pipefail
# open-loop at 2x r_sat keeps ~13k requests in flight; the default 1024 fds would turn them into client errors
ulimit -n 65536 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
CELL="$1"; MODEL="$2"; SEED="$3"; EXTRA="$4"; shift 4
WARMUP="${WARMUP:-100}"; REWARM="${REWARM:-20}"
RUN_ROOT="${RUN_ROOT:-$HOME/vllm-slo-lab/runs}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
# 0.82 not 0.90: the 4090 is a desktop GPU; at 0.90 the WDDM clients (compositor, browsers) push total committed
# VRAM past the physical 24 GiB and VidMm paging halves vLLM throughput (ADR 0007). Override with GPU_MEM_UTIL.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"   # W2 contrast cell uses 8192
SHIM="${SHIM:-0}"; SPECDEC="${SPECDEC:-}"; FAMILY="${FAMILY:-}"; FAKE_ENGINE="${FAKE_ENGINE:-0}"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
IPF="$HOME/vllm-slo-lab/.venv-loadgen/bin/inference-perf"
WSL="$(cd "$(dirname "$0")" && pwd)"
VLLM_PORT=8013; SHIM_PORT=8021
BATCH="$RUN_ROOT/$CELL/seed-$SEED"
mkdir -p "$BATCH"
cd "$HOME/vllm-slo-lab"
PID=""; SHIM_PID=""
cleanup() {
  if [ -n "$SHIM_PID" ]; then kill "$SHIM_PID" 2>/dev/null; wait "$SHIM_PID" 2>/dev/null; fi
  if [ -n "$PID" ]; then kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; fi
}
trap cleanup EXIT
pkill -f ".venv/bin/vllm serve" 2>/dev/null; pkill -f "EngineCore" 2>/dev/null
pkill -f "slo-lab shim" 2>/dev/null; pkill -f "fake_vllm.py" 2>/dev/null; sleep 3
if [ "$FAKE_ENGINE" = 1 ]; then
  # CPU-only rehearsal (W3 plan Task A7, W4 plan Task A4): no GPU gate, a fake engine on the same port
  fake_args=(--port "$VLLM_PORT" --model "$MODEL" --slots "${FAKE_SLOTS:-256}" --token-ms "${FAKE_TOKEN_MS:-10}")
  if [ -n "$SPECDEC" ] && [ "$SPECDEC" != none ]; then fake_args+=(--specdec); fi
  "$LABPY" "$WSL/fake_vllm.py" "${fake_args[@]}" > "$BATCH/serve.log" 2>&1 &
  PID=$!
else
  # quiet-GPU gate before the card is taken (WSL2: memory + utilization criteria), retried.
  # The previous cell's server keeps the GPU busy for ~30 s after it exits: on 2026-09-10 a
  # single reading 20 s after BF16's teardown (utilization 11 % against the 10 % threshold,
  # samples falling 15 -> 4) skipped the whole 2.5-hour FP8 contrast cell. The threshold is
  # unchanged; only a transient right after a teardown gets time to clear.
  QG_OK=0
  for attempt in 1 2 3 4 5; do
    if "$LABCLI" quiet-gpu --out "$BATCH/quiet_gpu.json" >/dev/null 2>&1; then QG_OK=1; break; fi
    echo "quiet-gpu attempt $attempt refused, retrying in 30 s"
    sleep 30
  done
  [ "$QG_OK" = 1 ] || { echo "QUIET_GPU_REFUSED"; head -40 "$BATCH/quiet_gpu.json"; exit 2; }
  # shellcheck disable=SC2086
  .venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port "$VLLM_PORT" --max-model-len 4096 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" $EXTRA > "$BATCH/serve.log" 2>&1 &
  PID=$!
fi
# Readiness must never use `curl … | grep -q`: grep -q exits on first match, curl takes SIGPIPE
# (141) and `set -o pipefail` turns the whole pipeline into a failure. That race silently killed
# the AWQ cell of 2026-09-10 after its server had answered 200. Command substitution has no pipe,
# and the loop's own result is reused instead of probing a second time.
# 180 polls x 5 s = 15 min: a cold post-reboot AWQ load took 6.3 min, BF16 weights are twice as big.
READY=0
for _ in $(seq 1 180); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$VLLM_PORT/v1/models" 2>/dev/null || true)
  if [ "$CODE" = "200" ]; then READY=1; break; fi
  kill -0 $PID 2>/dev/null || { echo "SERVER_EXITED_EARLY"; grep -i "error" "$BATCH/serve.log" | tail -3 | cut -c1-200; exit 1; }
  sleep 5
done
[ "$READY" = 1 ] || { echo "SERVER_NOT_READY after 900s (last code ${CODE:-none})"; exit 1; }
echo "server ready ($CELL seed $SEED) at $(date +%H:%M:%S): $(grep -o 'Loading weights took [0-9.]* seconds' "$BATCH/serve.log" | tail -1) | $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$BATCH/serve.log" | tail -1)"
BASE_URL="http://127.0.0.1:$VLLM_PORT"
STAGE_OPTS=()
if [ "$SHIM" = 1 ]; then
  # passthrough shim: one upstream connection per request, so the client never races uvicorn's
  # 5 s keep-alive close (ADR 0012 appendix); the stage records the shim's counters and CPU time
  "$LABCLI" shim --upstream "http://127.0.0.1:$VLLM_PORT" --host 127.0.0.1 --port "$SHIM_PORT" --policy passthrough > "$BATCH/shim.log" 2>&1 &
  SHIM_PID=$!
  SHIM_READY=0
  for _ in $(seq 1 60); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$SHIM_PORT/_shim/stats" 2>/dev/null || true)
    if [ "$CODE" = "200" ]; then SHIM_READY=1; break; fi
    kill -0 "$SHIM_PID" 2>/dev/null || break
    sleep 1
  done
  [ "$SHIM_READY" = 1 ] || { echo "SHIM_NOT_READY"; tail -5 "$BATCH/shim.log" | cut -c1-200; exit 1; }
  BASE_URL="http://127.0.0.1:$SHIM_PORT"
  STAGE_OPTS+=(--shim-stats-url "http://127.0.0.1:$SHIM_PORT/_shim/stats" --shim-pid "$SHIM_PID")
  echo "shim ready (passthrough, port $SHIM_PORT, pid $SHIM_PID)"
fi
if [ -n "$SPECDEC" ]; then STAGE_OPTS+=(--specdec "$SPECDEC"); fi
if [ -n "$FAMILY" ]; then STAGE_OPTS+=(--family "$FAMILY"); fi
# the extra flags may hold a JSON --speculative-config; its quotes must be escaped inside the manifest JSON
EXTRA_JSON="${EXTRA//\"/\\\"}"
FLAGS_JSON=$(printf '{"model":"%s","max_model_len":4096,"gpu_memory_utilization":%s,"max_num_seqs":%s,"max_num_batched_tokens":%s,"extra":"%s"}' "$MODEL" "$GPU_MEM_UTIL" "$MAX_NUM_SEQS" "$MAX_BATCHED_TOKENS" "$EXTRA_JSON")
first=1
for spec in "$@"; do
  IFS=: read -r kind a b c <<< "$spec"
  sseed="${c:-$SEED}"
  sdir="$RUN_ROOT/$CELL/seed-$sseed"; mkdir -p "$sdir"
  wu=$WARMUP; [ $first -eq 0 ] && wu=$REWARM   # full warm-up once per server, short re-warm between stages
  first=0
  if [ "$kind" = "ol" ]; then
    dir="$sdir/ol-rate-$a"
    "$LABCLI" run-stage --run-dir "$dir" --cell "$CELL" --model "$MODEL" --kind open_loop --seed "$sseed" --rate-rps "$a" --duration-s "$b" --warmup-requests "$wu" --inference-perf-bin "$IPF" --engine-flags "$FLAGS_JSON" --base-url "$BASE_URL" --metrics-url "http://127.0.0.1:$VLLM_PORT/metrics" ${STAGE_OPTS[@]+"${STAGE_OPTS[@]}"}
  else
    dir="$sdir/cl-conc-$a"
    "$LABCLI" run-stage --run-dir "$dir" --cell "$CELL" --model "$MODEL" --kind closed_loop --seed "$sseed" --concurrency "$a" --num-requests "$b" --warmup-requests "$wu" --inference-perf-bin "$IPF" --engine-flags "$FLAGS_JSON" --base-url "$BASE_URL" --metrics-url "http://127.0.0.1:$VLLM_PORT/metrics" ${STAGE_OPTS[@]+"${STAGE_OPTS[@]}"}
  fi
  echo "stage $spec exit=$?"
done
cleanup
PID=""; SHIM_PID=""
echo "batch done -> $BATCH"
