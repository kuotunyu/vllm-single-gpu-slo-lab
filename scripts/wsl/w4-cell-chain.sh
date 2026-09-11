#!/usr/bin/env bash
# One W4 speculative-decoding cell end to end (ADR 0015, plan 2026-09-11-w4-specdec.md), unattended
# and resumable:
#   closed-loop grid (seed 1) -> r_sat from the analysis -> open-loop 5 rates x 3 seeds in ONE server
#   session (rates scaled from the family baseline's r_sat when given) -> suspect quarantine and re-run
#   -> promote into evidence/raw/w4/<cell>/{closed-loop/seed-1,open-loop/seed-N}
# Every stage runs through the passthrough shim (batch.sh SHIM=1); stages with a manifest are skipped,
# so an interrupted cell resumes where it stopped.
#
# usage: w4-cell-chain.sh <cell> <model> <family> <specdec config> <max_num_seqs> [rate_ref]
#   specdec config: none | ngram | eagle3_4b   (config/specdec/<name>.yaml; the manifest label is the
#                   part before the first underscore: none / ngram / eagle3)
#   rate_ref: rps that the open-loop multipliers scale (the family's none cell r_sat); default: own r_sat
#   env: CLOSED_GRID ("1 8 32 128 256"), NREQ_SCALE (1.0; the W2 num_requests table times this, so a
#        faster cell still gets a window after the 60 s discard), MULTIPLIERS ("0.1 0.25 0.5 0.75 0.9"),
#        OPEN_SEEDS ("1 2 3"), STAGE_S (300), RUN_ROOT (~/vllm-slo-lab/runs-w4), RERUN_MAX (2), PROMOTE (1),
#        DRY (0), WARMUP (100), REWARM (20), FAKE_ENGINE (0), EVIDENCE_WEEK (w4),
#        GPU_MEM_UTIL (0.82), MAX_BATCHED_TOKENS (2048)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0
CELL="$1"; MODEL="$2"; FAMILY="$3"; SPECDEC_CFG="$4"; MAX_NUM_SEQS="$5"; RATE_REF_ARG="${6:-}"
SPECDEC="${SPECDEC_CFG%%_*}"
CLOSED_GRID="${CLOSED_GRID:-1 8 32 128 256}"
NREQ_SCALE="${NREQ_SCALE:-1.0}"
MULTIPLIERS="${MULTIPLIERS:-0.1 0.25 0.5 0.75 0.9}"
OPEN_SEEDS="${OPEN_SEEDS:-1 2 3}"
STAGE_S="${STAGE_S:-300}"
RERUN_MAX="${RERUN_MAX:-2}"; PROMOTE="${PROMOTE:-1}"; DRY="${DRY:-0}"
WARMUP="${WARMUP:-100}"; REWARM="${REWARM:-20}"; FAKE_ENGINE="${FAKE_ENGINE:-0}"
EVIDENCE_WEEK="${EVIDENCE_WEEK:-w4}"
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
ROOT="${RUN_ROOT:-$HOME/vllm-slo-lab/runs-w4}"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
# closed-loop num_requests per concurrency: the W2 table (>= 180 s at FP8's rps; faster cells run shorter,
# the same work per concurrency keeps the cells comparable, ADR 0009); NREQ_SCALE shrinks it for the dry run
declare -A NREQ=([1]=90 [2]=200 [4]=400 [8]=780 [16]=1550 [24]=2200 [32]=2800 [40]=3300 [48]=3800 [64]=4700 [96]=5800 [128]=6700 [192]=8000 [256]=9000)
log() { echo "[$(date +%H:%M:%S)] [$CELL w4] $*"; }

EXTRA=$("$LABCLI" specdec-flags "$REPO/config/specdec/$SPECDEC_CFG.yaml") || { log "SPECDEC_CONFIG_FAILED $SPECDEC_CFG"; exit 1; }

batch() {  # <label> <run_root> <session seed> <stage specs...> ; skips stages whose manifest exists
  local label="$1" root="$2" seed="$3"; shift 3
  local specs="" s kind a rest sseed dir
  for s in "$@"; do
    IFS=: read -r kind a rest sseed <<< "$s"
    sseed="${sseed:-$seed}"
    dir="$root/$CELL/seed-$sseed"
    if [ "$kind" = cl ]; then [ -f "$dir/cl-conc-$a/manifest.json" ] || specs="$specs $s"
    else [ -f "$dir/ol-rate-$a/manifest.json" ] || specs="$specs $s"; fi
  done
  if [ -z "$specs" ]; then log "$label: complete, skipping"; return 0; fi
  mkdir -p "$root/$CELL/seed-$seed"
  log "START $label seed=$seed stages=$specs"
  if [ "$DRY" = 1 ]; then echo "DRY: SHIM=1 SPECDEC=$SPECDEC FAMILY=$FAMILY WARMUP=$WARMUP MAX_NUM_SEQS=$MAX_NUM_SEQS RUN_ROOT=$root batch.sh $CELL $MODEL $seed '$EXTRA' $specs"; return 0; fi
  bash "$WSL/io-sampler.sh" "$root/$CELL/seed-$seed" >/dev/null 2>&1 &
  local sampler=$!
  # shellcheck disable=SC2086
  SHIM=1 SPECDEC="$SPECDEC" FAMILY="$FAMILY" WARMUP="$WARMUP" REWARM="$REWARM" FAKE_ENGINE="$FAKE_ENGINE" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" RUN_ROOT="$root" bash "$WSL/batch.sh" "$CELL" "$MODEL" "$seed" "$EXTRA" $specs
  local rc=$?
  kill "$sampler" 2>/dev/null
  log "END $label seed=$seed rc=$rc"
  return "$rc"
}

analyze() {  # <run_root> <out_dir>
  if [ "$DRY" = 1 ]; then echo "DRY: analyze $1 -> $2"; return 0; fi
  "$LABPY" "$REPO/scripts/analyze_batch.py" "$1/$CELL" --out "$2" >/dev/null 2>&1
}

quarantine_suspects() {  # <run_root> <analysis_dir> <cl|ol> [suspect|short] -> moves flagged stage dirs aside; prints count
  local root="$1" out="$2" kind="$3" mode="${4:-suspect}" moved=0 entry seed val d
  if [ "$DRY" = 1 ]; then echo 0; return 0; fi
  local q="$root/quarantine/$CELL"
  for entry in $("$LABPY" "$WSL/list_suspects.py" "$out" "$kind" "$mode"); do
    seed="${entry%%:*}"; val="${entry#*:}"
    if [ "$kind" = cl ]; then d="$root/$CELL/seed-$seed/cl-conc-$val"; else d="$root/$CELL/seed-$seed/ol-rate-$val"; fi
    if [ -d "$d" ]; then mkdir -p "$q/seed-$seed" && mv "$d" "$q/seed-$seed/$(basename "$d").$mode.$(date +%H%M%S)" && moved=$((moved + 1)); fi
  done
  echo "$moved"
}

closed_specs() {  # <scale> -> "cl:<c>:<n>" for the grid, num_requests = W2 table x scale
  local scale="$1" c nreq out=""
  for c in $CLOSED_GRID; do
    nreq=$(awk -v n="${NREQ[$c]}" -v s="$scale" 'BEGIN{printf "%d", n*s}')
    out="$out cl:$c:$nreq"
  done
  echo "$out"
}

log "START cell=$CELL model=$MODEL family=$FAMILY specdec=$SPECDEC max_num_seqs=$MAX_NUM_SEQS extra='$EXTRA'"

# ---- 1. closed-loop (seed 1, one session) ----
CL_ROOT="$ROOT/closed-loop-cells"
specs=$(closed_specs "$NREQ_SCALE")
# shellcheck disable=SC2086
batch closed-loop "$CL_ROOT" 1 $specs
analyze "$CL_ROOT" "$CL_ROOT/$CELL/analysis-seed-1"
# An accelerated cell can finish a W2-sized stage inside the 60 s discard period (CPU dry run,
# 2026-09-11: the 2.2x faster fake emptied both closed-loop windows and the cell had no r_sat).
# Such stages are set aside and re-run once with three times the requests (ADR 0015 item 3).
moved=$(quarantine_suspects "$CL_ROOT" "$CL_ROOT/$CELL/analysis-seed-1" cl short)
if [ "$moved" != 0 ]; then
  log "CLOSED_LOOP_SHORT: $moved stage(s) ended inside the discard period, re-running with 3x requests"
  specs3=$(closed_specs "$(awk -v s="$NREQ_SCALE" 'BEGIN{printf "%.3f", s*3}')")
  # shellcheck disable=SC2086
  batch closed-loop "$CL_ROOT" 1 $specs3
  analyze "$CL_ROOT" "$CL_ROOT/$CELL/analysis-seed-1"
fi
for attempt in $(seq 1 "$RERUN_MAX"); do
  moved=$(quarantine_suspects "$CL_ROOT" "$CL_ROOT/$CELL/analysis-seed-1" cl)
  [ "$moved" = 0 ] && break
  log "closed-loop: $moved suspect stage(s) quarantined, re-running (attempt $attempt)"
  # shellcheck disable=SC2086
  batch closed-loop "$CL_ROOT" 1 $specs
  analyze "$CL_ROOT" "$CL_ROOT/$CELL/analysis-seed-1"
done
if [ "$DRY" = 1 ]; then R_SAT=40; else
  R_SAT=$("$LABPY" "$WSL/read_r_sat.py" "$CL_ROOT/$CELL/analysis-seed-1/closed_loop.json" "$CELL" 2>/dev/null)
fi
if [ -z "${R_SAT:-}" ]; then log "no r_sat from closed-loop analysis; stopping cell"; exit 1; fi
[ "$DRY" = 1 ] || echo "$R_SAT" > "$CL_ROOT/$CELL/r_sat.txt"
log "r_sat=$R_SAT"

# ---- 2. open-loop: 5 rates x seeds, one server session, rates scaled from the family baseline ----
if [ -n "$RATE_REF_ARG" ]; then RATE_REF="$RATE_REF_ARG"; else
  RATE_REF="$R_SAT"
  [ "$SPECDEC" = none ] || log "WARNING no rate_ref given for an accelerated cell; scaling from its own r_sat"
fi
log "rate_ref=$RATE_REF (multipliers: $MULTIPLIERS; seeds: $OPEN_SEEDS; $STAGE_S s per stage)"
OL_ROOT="$ROOT/open-loop-cells"
specs=""
for seed in $OPEN_SEEDS; do
  for m in $MULTIPLIERS; do
    specs="$specs ol:$(awk -v r="$RATE_REF" -v m="$m" 'BEGIN{printf "%.2f", r*m}'):$STAGE_S:$seed"
  done
done
first_seed=$(echo "$OPEN_SEEDS" | awk '{print $1}')
# shellcheck disable=SC2086
batch open-loop "$OL_ROOT" "$first_seed" $specs
analyze "$OL_ROOT" "$OL_ROOT/$CELL/analysis"
for attempt in $(seq 1 "$RERUN_MAX"); do
  moved=$(quarantine_suspects "$OL_ROOT" "$OL_ROOT/$CELL/analysis" ol)
  [ "$moved" = 0 ] && break
  log "open-loop: $moved suspect stage(s) quarantined, re-running (attempt $attempt)"
  # shellcheck disable=SC2086
  batch open-loop "$OL_ROOT" "$first_seed" $specs
  analyze "$OL_ROOT" "$OL_ROOT/$CELL/analysis"
done
if [ "$DRY" != 1 ]; then
  mkdir -p "$OL_ROOT/$CELL"; echo "$RATE_REF" > "$OL_ROOT/$CELL/rate_ref.txt"
  "$LABPY" "$WSL/read_r_sat.py" "$OL_ROOT/$CELL/analysis/open_loop.json" "$CELL" r_slo
fi

# ---- 3. promote ----
if [ "$DRY" = 1 ]; then echo "DRY: promote $CELL"; elif [ "$PROMOTE" = 1 ]; then
  EVIDENCE_WEEK="$EVIDENCE_WEEK" bash "$WSL/promote-w2.sh" "$CL_ROOT/$CELL/seed-1" "$CELL/closed-loop/seed-1" 2>&1 | grep -v "redact: 0"
  for seed in $OPEN_SEEDS; do
    [ -d "$OL_ROOT/$CELL/seed-$seed" ] || continue
    EVIDENCE_WEEK="$EVIDENCE_WEEK" bash "$WSL/promote-w2.sh" "$OL_ROOT/$CELL/seed-$seed" "$CELL/open-loop/seed-$seed" 2>&1 | grep -v "redact: 0"
  done
fi
log "CELL DONE"
