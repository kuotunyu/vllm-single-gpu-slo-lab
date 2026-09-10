#!/usr/bin/env bash
# Stream one line per interesting event from the running night chain (for an external monitor).
# Covers progress AND every failure signature the chain can print, so silence really means
# "still working" rather than "died quietly".
set -uo pipefail
LOG="${1:-$HOME/vllm-slo-lab/runs-w2/w2-night-latest.log}"
for _ in $(seq 1 120); do
  [ -f "$LOG" ] && break
  sleep 5
done
[ -f "$LOG" ] || { echo "WATCH: no log at $LOG"; exit 1; }
stdbuf -o0 tail -F -n +1 "$LOG" 2>/dev/null | stdbuf -o0 grep -E \
  "server ready|CELL DONE|NIGHT DONE|START |END seed=|r_sat=|r_slo |no r_sat|TMMLU done|promoted -> |quarantined|QUARANTINED|SMOKE_|SHIM_FAILED|FAIL:|WARNING|SERVER_NOT_READY|SERVER_EXITED_EARLY|QUIET_GPU_REFUSED|Traceback|Error|error:|Killed|OOM|exit=[1-9]"
