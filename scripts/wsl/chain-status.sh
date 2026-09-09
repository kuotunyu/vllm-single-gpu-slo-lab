#!/usr/bin/env bash
# One-screen status of an unattended chain: which cell/stage is running, server progress, GPU,
# host pressure, and how many stages each batch directory has finished.
# usage: chain-status.sh
set -uo pipefail
ROOT="$HOME/vllm-slo-lab/runs-w2"
date +%H:%M:%S
echo "== drivers =="
pgrep -af "w2-night.sh|w2-cell-chain.sh|wsl/batch.sh|slo-lab run-stage|inference-perf|vllm serve" \
  | grep -v pgrep | sed -e 's#/home/tun2404/vllm-slo-lab/.venv[^ ]*/bin/##' -e 's#/mnt/d/[^ ]*/scripts/wsl/##' | cut -c1-120 \
  || echo "  (nothing running)"
echo "== gpu =="
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
echo "== host =="
head -1 /proc/pressure/io
head -1 /proc/pressure/cpu
cut -d' ' -f1-3 /proc/loadavg
echo "== stages finished per batch =="
for d in "$ROOT"/closed-loop-cells/*/seed-* "$ROOT"/open-loop-cells/*/seed-*; do
  [ -d "$d" ] || continue
  n=$(find "$d" -maxdepth 2 -name manifest.json 2>/dev/null | wc -l)
  q=$(find "$d/.." -maxdepth 3 -path "*quarantine*" -name manifest.json 2>/dev/null | wc -l)
  cur=$(ls -dt "$d"/*/ 2>/dev/null | head -1)
  printf "  %-58s %2d done  (latest: %s)\n" "${d#"$ROOT"/}" "$n" "$(basename "${cur:-none}")"
done
echo "== active server log tail =="
newest=$(ls -t "$ROOT"/*/*/seed-*/serve.log "$ROOT"/tmmluplus/*/serve.log 2>/dev/null | head -1)
if [ -n "$newest" ]; then
  echo "  $newest"
  grep -o -E "Loading weights took [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens|Application startup complete" "$newest" | tail -3 | sed 's/^/  /'
  grep -o "Avg generation throughput: [0-9.]* tokens/s, Running: [0-9]* reqs, Waiting: [0-9]* reqs" "$newest" | tail -2 | sed 's/^/  /'
  echo "  requests so far: $(grep -c "POST /v1/completions" "$newest")"
fi
