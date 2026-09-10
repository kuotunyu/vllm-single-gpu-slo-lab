#!/usr/bin/env bash
# One precision cell end to end (ADR 0006-0008 protocol), unattended and resumable:
#   closed-loop grid -> r_sat from the analysis -> open-loop 11 rates x seeds 1..3 -> suspect re-run pass
#   -> TMMLU+ slices (and full set when eval/tmmluplus/full.jsonl exists) -> promote into evidence/raw/w2/<cell>/
# Stages that already have a manifest are skipped, so an interrupted cell resumes where it stopped.
# usage: w2-cell-chain.sh <cell> <model> <max_num_seqs> "<closed concurrencies>"
#   env: MAX_BATCHED_TOKENS (default 2048), GPU_MEM_UTIL (default 0.82), DRY=1 prints the plan only,
#        OPEN_SEEDS (default "1 2 3"), SKIP_TMMLU=1, RERUN_MAX (default 2)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0
CELL="$1"; MODEL="$2"; MAX_NUM_SEQS="$3"; CLOSED_GRID="$4"
OPEN_SEEDS="${OPEN_SEEDS:-1 2 3}"
RERUN_MAX="${RERUN_MAX:-2}"
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
ROOT="$HOME/vllm-slo-lab/runs-w2"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
MULTIPLIERS="0.25 0.5 0.55 0.6 0.65 0.7 0.75 1.0 1.25 1.5 2.0"
# closed-loop num_requests per concurrency: >= 180 s at the FP8 rps of 2026-09-09 (slower cells run longer, fine)
declare -A NREQ=([1]=90 [2]=200 [4]=400 [8]=780 [16]=1550 [24]=2200 [32]=2800 [40]=3300 [48]=3800 [64]=4700 [96]=5800 [128]=6700 [192]=8000 [256]=9000)
log() { echo "[$(date +%H:%M:%S)] [$CELL] $*"; }

batch() {  # <label> <run_root> <seed> <stage specs...> ; skips stages whose manifest exists
  local label="$1" root="$2" seed="$3"; shift 3
  local dir="$root/$CELL/seed-$seed" specs="" s kind a
  for s in "$@"; do
    kind="${s%%:*}"; a="${s#*:}"; a="${a%%:*}"
    if [ "$kind" = cl ]; then [ -f "$dir/cl-conc-$a/manifest.json" ] || specs="$specs $s"
    else [ -f "$dir/ol-rate-$a/manifest.json" ] || specs="$specs $s"; fi
  done
  if [ -z "$specs" ]; then log "$label seed $seed: complete, skipping"; return 0; fi
  mkdir -p "$dir"
  log "START $label seed=$seed stages=$specs"
  if [ "${DRY:-0}" = 1 ]; then echo "DRY: WARMUP=100 MAX_NUM_SEQS=$MAX_NUM_SEQS RUN_ROOT=$root batch.sh $CELL $MODEL $seed '' $specs"; return 0; fi
  bash "$WSL/io-sampler.sh" "$dir" >/dev/null 2>&1 &
  local sampler=$!
  # shellcheck disable=SC2086
  WARMUP=100 MAX_NUM_SEQS="$MAX_NUM_SEQS" RUN_ROOT="$root" bash "$WSL/batch.sh" "$CELL" "$MODEL" "$seed" "" $specs
  local rc=$?
  kill "$sampler" 2>/dev/null
  log "END $label seed=$seed rc=$rc"
  return "$rc"
}

analyze() {  # <run_root> <out_dir>
  if [ "${DRY:-0}" = 1 ]; then echo "DRY: analyze $1 -> $2"; return 0; fi
  "$LABPY" "$REPO/scripts/analyze_batch.py" "$1/$CELL" --out "$2" >/dev/null 2>&1
}

list_suspects() {  # <analysis_dir> <cl|ol> -> lines "seed:value"
  "$LABPY" "$WSL/list_suspects.py" "$1" "$2"
}

quarantine_suspects() {  # <run_root> <analysis_dir> <cl|ol> -> moves suspect stage dirs aside; prints count
  local root="$1" out="$2" kind="$3" n=0 entry seed val d
  if [ "${DRY:-0}" = 1 ]; then echo 0; return 0; fi
  # quarantined stages leave the analysed tree entirely (their manifests must not be re-counted)
  local q="$root/quarantine/$CELL"
  for entry in $(list_suspects "$out" "$kind"); do
    seed="${entry%%:*}"; val="${entry#*:}"
    if [ "$kind" = cl ]; then d="$root/$CELL/seed-$seed/cl-conc-$val"; else d="$root/$CELL/seed-$seed/ol-rate-$val"; fi
    if [ -d "$d" ]; then mkdir -p "$q/seed-$seed" && mv "$d" "$q/seed-$seed/$(basename "$d").$(date +%H%M%S)" && n=$((n + 1)); fi
  done
  echo "$n"
}

# ---- 1. closed-loop ----
CL_ROOT="$ROOT/closed-loop-cells"
specs=""; for c in $CLOSED_GRID; do specs="$specs cl:$c:${NREQ[$c]}"; done
# shellcheck disable=SC2086
batch closed-loop "$CL_ROOT" 1 $specs
analyze "$CL_ROOT" "$CL_ROOT/$CELL/analysis-seed-1"
for attempt in $(seq 1 "$RERUN_MAX"); do
  n=$(quarantine_suspects "$CL_ROOT" "$CL_ROOT/$CELL/analysis-seed-1" cl)
  [ "$n" = 0 ] && break
  log "closed-loop: $n suspect stage(s) quarantined, re-running (attempt $attempt)"
  # shellcheck disable=SC2086
  batch closed-loop "$CL_ROOT" 1 $specs
  analyze "$CL_ROOT" "$CL_ROOT/$CELL/analysis-seed-1"
done
if [ "${DRY:-0}" = 1 ]; then R_SAT=40; else
  R_SAT=$("$LABPY" "$WSL/read_r_sat.py" "$CL_ROOT/$CELL/analysis-seed-1/closed_loop.json" "$CELL" 2>/dev/null)
fi
if [ -z "${R_SAT:-}" ]; then log "no r_sat from closed-loop analysis; stopping cell"; exit 1; fi
log "r_sat=$R_SAT"

# ---- 2. open-loop 11 rates x seeds ----
OL_ROOT="$ROOT/open-loop-cells"
rates=""; for m in $MULTIPLIERS; do rates="$rates ol:$(awk -v r="$R_SAT" -v m="$m" 'BEGIN{printf "%.2f", r*m}'):300"; done
for seed in $OPEN_SEEDS; do
  # shellcheck disable=SC2086
  batch open-loop "$OL_ROOT" "$seed" $rates
done
analyze "$OL_ROOT" "$OL_ROOT/$CELL/analysis"
for attempt in $(seq 1 "$RERUN_MAX"); do
  n=$(quarantine_suspects "$OL_ROOT" "$OL_ROOT/$CELL/analysis" ol)
  [ "$n" = 0 ] && break
  log "open-loop: $n suspect stage(s) quarantined, re-running (attempt $attempt)"
  for seed in $OPEN_SEEDS; do
    # shellcheck disable=SC2086
    batch open-loop "$OL_ROOT" "$seed" $rates
  done
  analyze "$OL_ROOT" "$OL_ROOT/$CELL/analysis"
done
if [ "${DRY:-0}" != 1 ]; then "$LABPY" "$WSL/read_r_sat.py" "$OL_ROOT/$CELL/analysis/open_loop.json" "$CELL" r_slo; fi

# ---- 3. TMMLU+ ----
if [ "${SKIP_TMMLU:-0}" != 1 ]; then
  out="$ROOT/tmmluplus/$CELL"; mkdir -p "$out"
  if [ "${DRY:-0}" = 1 ]; then echo "DRY: tmmlu slices (+full if present) for $CELL -> $out"; else
    cd "$HOME/vllm-slo-lab" || exit 1
    "$LABCLI" quiet-gpu --out "$out/quiet_gpu.json" || log "TMMLU quiet-gpu refused"
    pkill -f ".venv/bin/vllm serve" 2>/dev/null; sleep 3
    .venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8013 --max-model-len 4096 \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.82}" --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}" > "$out/serve.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 120); do grep -q "Application startup complete" "$out/serve.log" 2>/dev/null && break; kill -0 "$pid" 2>/dev/null || break; sleep 5; done
    cd "$REPO" || exit 1
    for s in 1 2 3; do
      [ -f "$out/slice-$s.json" ] || "$LABPY" scripts/tmmluplus_eval.py --slice "eval/tmmluplus/slice-$s.jsonl" --base-url http://127.0.0.1:8013 --model "$MODEL" --out "$out/slice-$s.json" --concurrency 16 2>&1 | tail -1
    done
    if [ -f eval/tmmluplus/full.jsonl ] && [ ! -f "$out/full.json" ]; then
      "$LABPY" scripts/tmmluplus_eval.py --slice eval/tmmluplus/full.jsonl --base-url http://127.0.0.1:8013 --model "$MODEL" --out "$out/full.json" --concurrency 32 2>&1 | tail -1
    fi
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    log "TMMLU done -> $out"
  fi
fi

# ---- 4. promote ----
if [ "${DRY:-0}" = 1 ]; then echo "DRY: promote $CELL"; else
  bash "$WSL/promote-w2.sh" "$CL_ROOT/$CELL/seed-1" "$CELL/closed-loop/seed-1" 2>&1 | grep -v "redact: 0"
  for seed in $OPEN_SEEDS; do bash "$WSL/promote-w2.sh" "$OL_ROOT/$CELL/seed-$seed" "$CELL/open-loop/seed-$seed" 2>&1 | grep -v "redact: 0"; done
  T="$REPO/evidence/raw/w2/$CELL/tmmluplus"; mkdir -p "$T"
  for f in slice-1.json slice-2.json slice-3.json full.json quiet_gpu.json; do
    [ -f "$ROOT/tmmluplus/$CELL/$f" ] && sed -e "s#$HOME#~#g" "$ROOT/tmmluplus/$CELL/$f" > "$T/$f"
  done
  [ -f "$ROOT/tmmluplus/$CELL/serve.log" ] && sed -e "s#$HOME#~#g" "$ROOT/tmmluplus/$CELL/serve.log" | "$LABPY" "$REPO/scripts/redact.py" redact - -o "$T/vllm.log" 2>/dev/null
  "$LABPY" "$REPO/scripts/compress_evidence.py" "$T"   # vllm.log -> vllm.log.gz (ADR 0011)
fi
log "CELL DONE"
