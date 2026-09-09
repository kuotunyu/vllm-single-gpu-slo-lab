#!/usr/bin/env bash
# `vllm bench serve` cross-check (spec §3.3, deferred §5.2): same token shape, a different load
# generator (single asyncio loop). Recorded next to the inference-perf numbers, never used for
# conclusions. Runs closed-loop c=64 and c=256 plus one open-loop rate (0.5 x r_sat, burstiness 1).
# usage: bash scripts/wsl/crosscheck.sh [rate_rps] [out_dir]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
MODEL="Qwen/Qwen3-8B-FP8"
RATE="${1:-21.83}"
OUT="${2:-$HOME/vllm-slo-lab/runs-w2/crosscheck/fp8}"
mkdir -p "$OUT"
cd "$HOME/vllm-slo-lab" || exit 1
"$LABCLI" quiet-gpu --out "$OUT/quiet_gpu.json" || { echo "QUIET_GPU_REFUSED"; exit 2; }
pkill -f ".venv/bin/vllm serve" 2>/dev/null; sleep 3
.venv/bin/vllm serve "$MODEL" --host 127.0.0.1 --port 8013 --max-model-len 4096 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.82}" --max-num-seqs 256 --max-num-batched-tokens 2048 > "$OUT/serve.log" 2>&1 &
PID=$!
for _ in $(seq 1 120); do
  grep -q "Application startup complete" "$OUT/serve.log" 2>/dev/null && break
  kill -0 "$PID" 2>/dev/null || { echo "SERVER_EXITED_EARLY"; exit 1; }
  sleep 5
done
echo "server ready $(date +%H:%M:%S)"
bench() {  # <label> <extra args...>
  local label="$1"; shift
  .venv/bin/vllm bench serve --backend vllm --base-url http://127.0.0.1:8013 --endpoint /v1/completions \
    --model "$MODEL" --tokenizer "$MODEL" --dataset-name random --random-input-len 108 --random-output-len 132 \
    --random-range-ratio 0 --ignore-eos --seed 1 --num-warmups 100 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "$label.json" "$@" 2>&1 | grep -E "Successful|Request throughput|Output token throughput|TTFT|TPOT|Mean|Median|P95|P99" | head -20
  echo "--- $label done $(date +%H:%M:%S)"
}
bench closed-c64 --num-prompts 3000 --max-concurrency 64 --request-rate inf
bench closed-c256 --num-prompts 6000 --max-concurrency 256 --request-rate inf
bench "open-r${RATE}" --num-prompts "$(awk -v r="$RATE" 'BEGIN{printf "%d", r*300}')" --request-rate "$RATE" --burstiness 1
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
echo "crosscheck done -> $OUT"
