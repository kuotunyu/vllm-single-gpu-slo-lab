#!/usr/bin/env bash
# vllm bench serve closed-loop cross-check at one concurrency (completion API, same token shape).
# usage: wsl-bench-serve-cl.sh <concurrency> <num_prompts> <out-dir> [seed]
# Expects a vLLM server already on 127.0.0.1:8013 (start it with batch.sh style flags).
set -uo pipefail
C="$1"; N="$2"; OUT="$3"; SEED="${4:-1}"
mkdir -p "$OUT"; cd "$HOME/vllm-slo-lab" || exit 1
export HF_HUB_OFFLINE=1
.venv/bin/vllm bench serve --backend vllm --base-url http://127.0.0.1:8013 --endpoint /v1/completions \
  --model Qwen/Qwen3-8B-FP8 --tokenizer Qwen/Qwen3-8B-FP8 --dataset-name random \
  --random-input-len 108 --random-output-len 132 --random-range-ratio 0 --ignore-eos \
  --num-prompts "$N" --max-concurrency "$C" --request-rate inf --seed "$SEED" --num-warmups 20 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --save-result --save-detailed --result-dir "$OUT" --result-filename "bench-serve-cl-c${C}-seed${SEED}.json" 2>&1 | tail -40
