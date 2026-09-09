#!/usr/bin/env bash
# Sample /proc/pressure/{io,cpu} + loadavg every 10 s into the batch dir while the sweep runs.
set -uo pipefail
out="${1:-$HOME/vllm-slo-lab/runs/fp8/seed-1}/io-pressure.log"; mkdir -p "$(dirname "$out")"
echo "t_iso io_some_avg10 io_full_avg10 cpu_some_avg10 loadavg1 vllm_running" >> "$out"
for i in $(seq 1 480); do
  ios=$(awk '/^some/{print $2}' /proc/pressure/io | cut -d= -f2); iof=$(awk '/^full/{print $2}' /proc/pressure/io | cut -d= -f2)
  cpus=$(awk '/^some/{print $2}' /proc/pressure/cpu | cut -d= -f2); la=$(cut -d' ' -f1 /proc/loadavg)
  run=$(curl -s -m 2 http://127.0.0.1:8013/metrics 2>/dev/null | awk '/^vllm:num_requests_running/{print $2}' | head -1)
  echo "$(date -Is) $ios $iof $cpus $la ${run:-NA}" >> "$out"
  [ $i -gt 6 ] && { pgrep -f "wsl/batch.sh|wsl-batch.sh" >/dev/null || { echo "driver gone at sample $i"; exit 0; }; }
  sleep 10
done
echo "sampler timeout (70 min)"
