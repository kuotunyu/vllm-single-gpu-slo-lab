#!/usr/bin/env bash
# W3 night (plan docs/superpowers/plans/2026-09-11-w3-admission-trace.md, ADR 0012):
# a four-minute GPU smoke through the shim, then FP8 and BF16 admission traces, three seeds each.
# Launch through run-logged.sh so the log lands on ext4. Stages with a manifest are skipped, so
# relaunching after an interruption resumes. env: DRY (plan only), SKIP_SMOKE.
set -uo pipefail
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
DRY="${DRY:-0}"
log() { echo "[$(date +%H:%M:%S)] [w3-night] $*"; }
export DRY

log "START (DRY=$DRY)"
if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  SEEDS=1 POLICIES="direct passthrough bounded_queue" RERUN_MAX=0 \
    PROFILE="$REPO/config/traffic/burst-smoke.yaml" RUN_ROOT="$HOME/vllm-slo-lab/runs-w3-smoke" \
    bash "$WSL/w3-trace-chain.sh" fp8-smoke Qwen/Qwen3-8B-FP8 256 256 43.67
  if [ "$DRY" != 1 ]; then
    if ! "$LABPY" "$WSL/check_w3_smoke.py" "$HOME/vllm-slo-lab/runs-w3-smoke/fp8-smoke/seed-1"; then
      log "SMOKE_FAILED: chain stopped before the main cells"
      exit 3
    fi
  fi
fi
bash "$WSL/w3-trace-chain.sh" fp8 Qwen/Qwen3-8B-FP8 256 256 43.67
bash "$WSL/w3-trace-chain.sh" bf16 Qwen/Qwen3-8B 40 40 11.40
log "W3 NIGHT DONE"
