#!/usr/bin/env bash
# Poll the batch serve.log until the server is ready or the driver gave up; print the outcome once.
set -uo pipefail
log="${1:-$HOME/vllm-slo-lab/runs/fp8/seed-1/serve.log}"
for i in $(seq 1 45); do
  if grep -q "Application startup complete" "$log" 2>/dev/null; then
    echo "READY after poll $i ($(date +%H:%M:%S))"
    grep -o -E "Loading weights took [0-9.]+ seconds|Model loading took [0-9.]+ GiB and [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$log"
    exit 0
  fi
  if ! pgrep -f "wsl/batch.sh|wsl-batch.sh" >/dev/null; then echo "DRIVER_GONE at poll $i"; tail -5 "$log" | cut -c1-200; exit 2; fi
  if ! pgrep -f ".venv/bin/vllm serve" >/dev/null; then echo "SERVER_GONE at poll $i"; grep -iE "error|Traceback" "$log" | tail -5 | cut -c1-200; exit 3; fi
  sleep 20
done
echo "STILL_LOADING after 15 min"; tail -3 "$log" | cut -c1-200; exit 4
