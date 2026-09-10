#!/usr/bin/env bash
# W3 admission trace for one engine cell (ADR 0012, plan 2026-09-11-w3-admission-trace.md).
# Per seed: one vLLM session; the seed's trace is replayed through each admission policy in a
# rotated order (a Latin square, so every policy runs once in every position); then the seed dir
# is promoted to evidence/raw/w3/<cell>/trace/seed-N. Stages with a manifest are skipped, so
# running the same command again resumes. Suspect stages (probe TPOT drift, VRAM
# oversubscription) are quarantined and re-run in a fresh session, at most RERUN_MAX rounds.
#
# usage: w3-trace-chain.sh <cell> <model> <max_num_seqs> <capacity> <rate_ref>
#   env: SEEDS ("1 2 3"), POLICIES (overrides the rotation, e.g. for the smoke), PROFILE
#        (config/traffic/burst25.yaml), RUN_ROOT (~/vllm-slo-lab/runs-w3), PROMOTE (1),
#        RERUN_MAX (2), DRY (0), GPU_MEM_UTIL (0.82), MAX_BATCHED_TOKENS (2048), TIMEOUT_S (1.0)
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0
CELL="$1"; MODEL="$2"; MAX_NUM_SEQS="$3"; CAPACITY="$4"; RATE_REF="$5"
SEEDS="${SEEDS:-1 2 3}"
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
PROFILE="${PROFILE:-$REPO/config/traffic/burst25.yaml}"
RUN_ROOT="${RUN_ROOT:-$HOME/vllm-slo-lab/runs-w3}"
PROMOTE="${PROMOTE:-1}"; DRY="${DRY:-0}"; RERUN_MAX="${RERUN_MAX:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"; MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
TIMEOUT_S="${TIMEOUT_S:-1.0}"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
IPF="$HOME/vllm-slo-lab/.venv-loadgen/bin/inference-perf"
VLLM_PORT=8013; SHIM_PORT=8021
SERVER_PID=""; SHIM_PID=""; SAMPLER_PID=""
log() { echo "[$(date +%H:%M:%S)] [$CELL w3] $*"; }

rotation() {
  if [ -n "${POLICIES:-}" ]; then echo "$POLICIES"; return; fi
  case "$1" in
    1) echo "passthrough hard_cap bounded_queue" ;;
    2) echo "hard_cap bounded_queue passthrough" ;;
    *) echo "bounded_queue passthrough hard_cap" ;;
  esac
}

stop_shim() {
  if [ -n "$SHIM_PID" ]; then kill "$SHIM_PID" 2>/dev/null; wait "$SHIM_PID" 2>/dev/null; fi
  SHIM_PID=""
}

stop_server() {
  stop_shim
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; fi
  SERVER_PID=""
  if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null; fi
  SAMPLER_PID=""
}
trap stop_server EXIT

start_server() {  # $1 = seed dir
  local dir="$1" qg=0 ready=0 code=""
  pkill -f ".venv/bin/vllm serve" 2>/dev/null; pkill -f "EngineCore" 2>/dev/null
  pkill -f "slo-lab shim" 2>/dev/null; pkill -f "fake_vllm.py" 2>/dev/null; sleep 3
  if [ "${FAKE_ENGINE:-0}" = 1 ]; then
    # CPU-only dry run (plan Task A7): no GPU gate, a fake engine on the same port
    "$LABPY" "$WSL/fake_vllm.py" --port "$VLLM_PORT" --model "$MODEL" --slots "${FAKE_SLOTS:-8}" --token-ms "${FAKE_TOKEN_MS:-2}" > "$dir/serve.log" 2>&1 &
    SERVER_PID=$!
  else
    for attempt in 1 2 3 4 5; do
      if "$LABCLI" quiet-gpu --out "$dir/quiet_gpu.json" >/dev/null 2>&1; then qg=1; break; fi
      log "quiet-gpu attempt $attempt refused, retrying in 30 s"; sleep 30
    done
    [ "$qg" = 1 ] || { log "QUIET_GPU_REFUSED"; return 2; }
    cd "$HOME/vllm-slo-lab" || return 1
    # shellcheck disable=SC2086
    .venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port "$VLLM_PORT" --max-model-len 4096 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" > "$dir/serve.log" 2>&1 &
    SERVER_PID=$!
  fi
  # readiness: command substitution, never `curl | grep -q` (SIGPIPE under pipefail, 2026-09-10)
  for _ in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$VLLM_PORT/v1/models" 2>/dev/null || true)
    if [ "$code" = "200" ]; then ready=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { log "SERVER_EXITED_EARLY"; grep -i error "$dir/serve.log" | tail -3 | cut -c1-200; return 1; }
    sleep 5
  done
  [ "$ready" = 1 ] || { log "SERVER_NOT_READY after 900 s (last code ${code:-none})"; return 1; }
  log "server ready: $(grep -o 'Loading weights took [0-9.]* seconds' "$dir/serve.log" | tail -1) | $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$dir/serve.log" | tail -1)"
  IO_DRIVER_PATTERN="w3-trace-chain.sh" IO_MAX_SAMPLES=1500 bash "$WSL/io-sampler.sh" "$dir" >/dev/null 2>&1 &
  SAMPLER_PID=$!
  return 0
}

vllm_idle() {  # engine running + waiting back to 0, polled up to 600 s
  local m r w
  for _ in $(seq 1 120); do
    m=$(curl -s --max-time 5 "http://127.0.0.1:$VLLM_PORT/metrics" 2>/dev/null || true)
    r=$(printf '%s\n' "$m" | awk '/^vllm:num_requests_running[{ ]/ {s += $NF} END {printf "%d", s}')
    w=$(printf '%s\n' "$m" | awk '/^vllm:num_requests_waiting[{ ]/ {s += $NF} END {printf "%d", s}')
    if [ -n "$m" ] && [ "$r" = 0 ] && [ "$w" = 0 ]; then return 0; fi
    sleep 5
  done
  return 1
}

start_shim() {  # $1 = policy, $2 = seed dir
  local policy="$1" dir="$2" code=""
  local args=(--upstream "http://127.0.0.1:$VLLM_PORT" --host 127.0.0.1 --port "$SHIM_PORT" --policy "$policy")
  if [ "$policy" != passthrough ]; then args+=(--capacity "$CAPACITY"); fi
  if [ "$policy" = bounded_queue ]; then args+=(--queue-limit "$CAPACITY" --timeout-s "$TIMEOUT_S"); fi
  "$LABCLI" shim "${args[@]}" > "$dir/shim-$policy.log" 2>&1 &
  SHIM_PID=$!
  for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$SHIM_PORT/_shim/stats" 2>/dev/null || true)
    if [ "$code" = "200" ]; then return 0; fi
    kill -0 "$SHIM_PID" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

run_seed() {  # $1 = seed; runs every policy of the seed that has no manifest yet
  local seed="$1" dir="$RUN_ROOT/$CELL/seed-$1" order missing="" first=1 wu rc
  mkdir -p "$dir"
  order=$(rotation "$seed")
  for p in $order; do [ -f "$dir/trace-$p/manifest.json" ] || missing="$missing $p"; done
  if [ -z "$missing" ]; then log "seed $seed complete"; return 0; fi
  local trace="$dir/trace-seed-$seed.csv"
  if [ ! -f "$trace" ]; then
    "$LABCLI" make-trace --rate-ref "$RATE_REF" --seed "$seed" --out "$trace" --profile "$PROFILE" > "$dir/trace-seed-$seed.json" || { log "MAKE_TRACE_FAILED"; return 1; }
  fi
  log "seed $seed: order [$order], to run [$missing], arrivals $(grep -o '"arrivals": [0-9]*' "$dir/trace-seed-$seed.json" | head -1 | grep -o '[0-9]*$')"
  if [ "$DRY" = 1 ]; then return 0; fi
  start_server "$dir" || { rc=$?; stop_server; return "$rc"; }
  local flags
  flags=$(printf '{"model":"%s","max_model_len":4096,"gpu_memory_utilization":%s,"max_num_seqs":%s,"max_num_batched_tokens":%s,"extra":""}' "$MODEL" "$GPU_MEM_UTIL" "$MAX_NUM_SEQS" "$MAX_BATCHED_TOKENS")
  for p in $order; do
    [ -f "$dir/trace-$p/manifest.json" ] && continue
    vllm_idle || log "WARNING engine not idle after 600 s before $p"
    local target=(--base-url "http://127.0.0.1:$SHIM_PORT" --shim-stats-url "http://127.0.0.1:$SHIM_PORT/_shim/stats")
    if [ "$p" = direct ]; then
      # control arm for the smoke only: no shim, the client talks to vLLM (ADR 0012 item 4)
      target=(--base-url "http://127.0.0.1:$VLLM_PORT")
    else
      start_shim "$p" "$dir" || { log "SHIM_FAILED $p"; stop_shim; continue; }
      target+=(--shim-pid "$SHIM_PID")
    fi
    wu=20; [ "$first" = 1 ] && wu=100; first=0
    log "START seed=$seed policy=$p warmup=$wu"
    "$LABCLI" run-stage --run-dir "$dir/trace-$p" --cell "$CELL" --model "$MODEL" --kind trace --seed "$seed" "${target[@]}" --metrics-url "http://127.0.0.1:$VLLM_PORT/metrics" --policy "$p" --trace-file "$trace" --profile "$PROFILE" --warmup-requests "$wu" --inference-perf-bin "$IPF" --engine-flags "$flags"
    rc=$?
    stop_shim
    log "END seed=$seed policy=$p rc=$rc"
  done
  stop_server
  return 0
}

promote_seed() {
  local seed="$1"
  [ "$PROMOTE" = 1 ] || return 0
  EVIDENCE_WEEK=w3 bash "$WSL/promote-w2.sh" "$RUN_ROOT/$CELL/seed-$seed" "$CELL/trace/seed-$seed" 2>&1 | grep -v "redact: 0"
}

quarantine_suspects() {  # moves suspect stages out of the tree; prints how many (log -> stderr)
  "$LABPY" "$REPO/scripts/analyze_batch.py" "$RUN_ROOT/$CELL" --out "$RUN_ROOT/$CELL/analysis" >/dev/null 2>&1
  local moved=0 stamp
  stamp=$(date +%Y%m%d-%H%M%S)
  while read -r seed policy; do
    [ -n "$seed" ] || continue
    mkdir -p "$RUN_ROOT/quarantine/$CELL/seed-$seed"
    mv "$RUN_ROOT/$CELL/seed-$seed/trace-$policy" "$RUN_ROOT/quarantine/$CELL/seed-$seed/trace-$policy-$stamp"
    log "QUARANTINED seed=$seed policy=$policy" >&2
    moved=$((moved + 1))
  done < <("$LABPY" "$WSL/list_trace_suspects.py" "$RUN_ROOT/$CELL/analysis/admission.json")
  echo "$moved"
}

log "START cell=$CELL model=$MODEL max_num_seqs=$MAX_NUM_SEQS C=$CAPACITY r=$RATE_REF seeds=[$SEEDS] profile=$(basename "$PROFILE")"
for seed in $SEEDS; do
  run_seed "$seed"
done
if [ "$DRY" != 1 ]; then
  # suspects are re-measured before anything is promoted, so evidence never holds a quarantined stage
  for round in $(seq 1 "$RERUN_MAX"); do
    n=$(quarantine_suspects)
    [ "${n:-0}" = 0 ] && break
    log "re-run round $round for $n suspect stage(s)"
    for seed in $SEEDS; do run_seed "$seed"; done
  done
  "$LABPY" "$REPO/scripts/analyze_batch.py" "$RUN_ROOT/$CELL" --out "$RUN_ROOT/$CELL/analysis" >/dev/null 2>&1
  for seed in $SEEDS; do promote_seed "$seed"; done
fi
log "CELL DONE"
