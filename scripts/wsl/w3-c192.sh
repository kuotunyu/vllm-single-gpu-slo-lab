#!/usr/bin/env bash
# FP8 burst-safe cap follow-up (HANDOFF "FP8 突發安全上限補點"; exploratory, NOT preregistered):
# the W3 hard-cap arm re-run with C = 192 instead of the closed-loop C = 256, replaying the same
# seeded trace per seed as W3 (burst25 profile, rate_ref 43.67), three seeds, one vLLM session
# per seed. Promotes to evidence/raw/w3/fp8-c192/trace/seed-N. Launch through run-logged.sh.
# env: SEEDS ("1 2 3"), POLICIES ("hard_cap"), RERUN_MAX (1), RUN_ROOT (~/vllm-slo-lab/runs-w3-c192)
set -uo pipefail
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
log() { echo "[$(date +%H:%M:%S)] [w3-c192] $*"; }
log "START"
SEEDS="${SEEDS:-1 2 3}" POLICIES="${POLICIES:-hard_cap}" RERUN_MAX="${RERUN_MAX:-1}" \
  RUN_ROOT="${RUN_ROOT:-$HOME/vllm-slo-lab/runs-w3-c192}" \
  bash "$REPO/scripts/wsl/w3-trace-chain.sh" fp8-c192 Qwen/Qwen3-8B-FP8 256 192 43.67
log "DONE rc=$?"
