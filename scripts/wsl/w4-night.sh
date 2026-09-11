#!/usr/bin/env bash
# W4 night (plan docs/superpowers/plans/2026-09-11-w4-specdec.md, ADR 0015): a 20-minute GPU smoke on
# the two accelerated servers, then the five speculative-decoding cells in value order, every stage
# through the passthrough shim. Launch through run-logged.sh so the log lands on ext4. Stages with a
# manifest are skipped, so relaunching after an interruption resumes.
#   env: DRY (plan only), SKIP_SMOKE, CELLS (subset, default all five; ADR 0015 item 11 cut options),
#        MAX_NUM_SEQS_Q4B (256; 192 if the smoke shows preemption at c = 256), RUN_ROOT, SMOKE_ROOT,
#        NREQ_SCALE_FP8 / NREQ_SCALE_Q4B / NREQ_SCALE_EAGLE (closed-loop request multipliers 1 / 2 / 3),
#        MULTIPLIERS (open-loop rate multipliers, e.g. "0.25 0.5 0.75 0.9" to drop the 0.1 x point)
set -uo pipefail
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
RUN_ROOT="${RUN_ROOT:-$HOME/vllm-slo-lab/runs-w4}"
SMOKE_ROOT="${SMOKE_ROOT:-$HOME/vllm-slo-lab/runs-w4-smoke}"
CELLS="${CELLS:-fp8-none fp8-ngram q4b-none q4b-eagle3 q4b-ngram}"
Q4B_SEQS="${MAX_NUM_SEQS_Q4B:-256}"
# closed-loop num_requests = W2 table x speed factor, so a faster cell still has >= 60 s of window
# after the discard (4B about twice FP8 8B's throughput, EAGLE-3 up to three times single-stream;
# ADR 0015 item 3). The chain re-runs any stage that still ends inside the discard with 3x requests.
NREQ_SCALE_FP8="${NREQ_SCALE_FP8:-1.0}"; NREQ_SCALE_Q4B="${NREQ_SCALE_Q4B:-2.0}"; NREQ_SCALE_EAGLE="${NREQ_SCALE_EAGLE:-3.0}"
DRY="${DRY:-0}"
export DRY RUN_ROOT
log() { echo "[$(date +%H:%M:%S)] [w4-night] $*"; }

smoke_session() {  # <cell> <model> <family> <specdec config> <stage specs...>
  local cell="$1" model="$2" family="$3" cfg="$4"; shift 4
  local label="${cfg%%_*}" extra
  extra=$("$LABCLI" specdec-flags "$REPO/config/specdec/$cfg.yaml")
  if [ "$DRY" = 1 ]; then echo "DRY: smoke SHIM=1 SPECDEC=$label FAMILY=$family WARMUP=30 batch.sh $cell $model 1 '$extra' $*"; return 0; fi
  if [ -f "$SMOKE_ROOT/$cell/seed-1/${*##* }/manifest.json" ]; then log "smoke $cell already ran"; return 0; fi
  log "START smoke $cell"
  # the smoke is never a reported number: 30 warm-up requests, 10 re-warm
  SHIM=1 SPECDEC="$label" FAMILY="$family" WARMUP=30 REWARM=10 MAX_NUM_SEQS=256 RUN_ROOT="$SMOKE_ROOT" \
    bash "$WSL/batch.sh" "$cell" "$model" 1 "$extra" "$@"
  log "END smoke $cell rc=$?"
}

rate_ref() {  # <baseline cell> -> its closed-loop r_sat, or '' when that cell has not run
  local f="$RUN_ROOT/closed-loop-cells/$1/r_sat.txt"
  if [ -f "$f" ]; then cat "$f"; else echo ""; fi
}

log "START (DRY=$DRY) cells=[$CELLS]"
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  # 4B + EAGLE-3: c = 1 and the full c = 256 (preemption check), one open-loop stage; 8B + n-gram: c = 1
  # and one open-loop stage. Proves: spec config loads, /metrics has the spec-decode counters, drafts
  # happen on natural text, zero preemption, shim clean, labels in the manifests (ADR 0015 item 10).
  smoke_session smoke-q4b-eagle3 Qwen/Qwen3-4B q4b eagle3_4b cl:1:60 cl:256:2000 ol:20:90
  smoke_session smoke-fp8-ngram Qwen/Qwen3-8B-FP8 fp8 ngram cl:1:60 ol:20:90
  if [ "$DRY" != 1 ]; then
    if ! "$LABPY" "$WSL/check_w4_smoke.py" "$SMOKE_ROOT" \
        smoke-q4b-eagle3:q4b:eagle3:cl-conc-1,cl-conc-256,ol-rate-20 \
        smoke-fp8-ngram:fp8:ngram:cl-conc-1,ol-rate-20; then
      log "SMOKE_FAILED: chain stopped before the main cells"
      exit 3
    fi
    for cell in smoke-q4b-eagle3 smoke-fp8-ngram; do
      EVIDENCE_WEEK=w4 bash "$WSL/promote-w2.sh" "$SMOKE_ROOT/$cell/seed-1" "smoke/${cell#smoke-}/seed-1" 2>&1 | grep -v "redact: 0"
    done
  fi
fi

for cell in $CELLS; do
  case "$cell" in
    fp8-none)   NREQ_SCALE="$NREQ_SCALE_FP8"   bash "$WSL/w4-cell-chain.sh" fp8-none   Qwen/Qwen3-8B-FP8 fp8 none      256 ;;
    fp8-ngram)  NREQ_SCALE="$NREQ_SCALE_FP8"   bash "$WSL/w4-cell-chain.sh" fp8-ngram  Qwen/Qwen3-8B-FP8 fp8 ngram     256 "$(rate_ref fp8-none)" ;;
    q4b-none)   NREQ_SCALE="$NREQ_SCALE_Q4B"   bash "$WSL/w4-cell-chain.sh" q4b-none   Qwen/Qwen3-4B     q4b none      "$Q4B_SEQS" ;;
    q4b-eagle3) NREQ_SCALE="$NREQ_SCALE_EAGLE" bash "$WSL/w4-cell-chain.sh" q4b-eagle3 Qwen/Qwen3-4B     q4b eagle3_4b "$Q4B_SEQS" "$(rate_ref q4b-none)" ;;
    q4b-ngram)  NREQ_SCALE="$NREQ_SCALE_Q4B"   bash "$WSL/w4-cell-chain.sh" q4b-ngram  Qwen/Qwen3-4B     q4b ngram     "$Q4B_SEQS" "$(rate_ref q4b-none)" ;;
    *) log "unknown cell $cell, skipped" ;;
  esac
done

if [ "$DRY" != 1 ]; then
  # tables for the monitor; the committed ones are rebuilt from analysis/tables/index.json at wrap-up
  for fam in fp8 q4b; do
    dirs=""
    for cell in $CELLS; do
      case "$cell" in "$fam"-*) [ -d "$REPO/evidence/raw/w4/$cell" ] && dirs="$dirs $REPO/evidence/raw/w4/$cell" ;; esac
    done
    # shellcheck disable=SC2086
    [ -n "$dirs" ] && "$LABPY" "$REPO/scripts/analyze_batch.py" $dirs --out "$RUN_ROOT/tables/w4-$fam-specdec" >/dev/null 2>&1 && log "tables -> $RUN_ROOT/tables/w4-$fam-specdec"
  done
fi
log "W4 NIGHT DONE"
