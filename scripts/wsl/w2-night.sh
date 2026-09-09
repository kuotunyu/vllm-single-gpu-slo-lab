#!/usr/bin/env bash
# Overnight W2 master chain (user-declared GPU-quiet window, 2026-09-09 night):
#   AWQ -> GPTQ-Int4 -> BF16 (each: closed-loop, open-loop x3 seeds, TMMLU+, promote)
#   -> FP8 max-num-batched-tokens 8192 contrast (closed-loop + open-loop seed 1)
#   -> FP8 `vllm bench serve` cross-check
# usage: bash scripts/wsl/w2-night.sh            (DRY=1 prints the plan)
set -uo pipefail
WSL="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab/scripts/wsl"
log() { echo "[$(date +%H:%M:%S)] [night] $*"; }
FULL="1 2 4 8 16 32 64 96 128 192 256"
# BF16: KV at 0.82 is about 1.7 GiB (W1: 25,056 tokens at 0.90 = 3.7 GiB), i.e. ~11.7k tokens = 48 requests
# of 240 tokens; the grid stops at 40 with max-num-seqs 40 to stay clear of preemption.
BF16="1 2 4 8 16 24 32 40"
log "START awq";  bash "$WSL/w2-cell-chain.sh" awq  Qwen/Qwen3-8B-AWQ 256 "$FULL"
log "START gptq"; bash "$WSL/w2-cell-chain.sh" gptq JunHowie/Qwen3-8B-GPTQ-Int4 256 "$FULL"
log "START bf16"; bash "$WSL/w2-cell-chain.sh" bf16 Qwen/Qwen3-8B 40 "$BF16"
log "START fp8-mbt8192 (contrast, open-loop seed 1 only)"
MAX_BATCHED_TOKENS=8192 OPEN_SEEDS="1" SKIP_TMMLU=1 bash "$WSL/w2-cell-chain.sh" fp8-mbt8192 Qwen/Qwen3-8B-FP8 256 "$FULL"
log "START fp8 crosscheck"; [ "${DRY:-0}" = 1 ] && echo "DRY: crosscheck.sh" || bash "$WSL/crosscheck.sh" 21.83
log "NIGHT DONE"
