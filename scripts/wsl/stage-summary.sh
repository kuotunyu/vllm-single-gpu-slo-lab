#!/usr/bin/env bash
# Per-stage summary of a batch directory (closed- or open-loop): the numbers a monitor needs to
# decide "clean or contaminated" without opening the manifests by hand.
#
# usage: stage-summary.sh <batch dir>            # e.g. ~/vllm-slo-lab/runs-w2/closed-loop-cells/awq/seed-1
#        stage-summary.sh <cell dir> --seeds     # iterates seed-* under the given cell directory
set -uo pipefail
TARGET="${1:?usage: stage-summary.sh <batch dir|cell dir --seeds>}"
MODE="${2:-single}"
PY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"

summarise() {
  "$PY" - "$1" <<'PYEOF'
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])


def key(p: Path) -> float:
    name = p.parent.name
    try:
        return float(name.rsplit("-", 1)[-1])
    except ValueError:
        return 0.0


stages = sorted(d.glob("*/manifest.json"), key=key)
if not stages:
    print(f"  (no completed stage in {d.name})")
    raise SystemExit(0)
print(f"== {d.parent.name}/{d.name}: {len(stages)} stage(s)")
for path in stages:
    m = json.loads(path.read_text(encoding="utf-8"))
    s = m.get("summary") or {}
    pw = m.get("power_window") or {}
    hb = ((m.get("host_before") or {}).get("windows_gpu_memory") or {}).get("committed_mb")
    w = m.get("warmup") or {}
    point = f"c={m['concurrency']}" if m.get("concurrency") else f"r={m.get('rate_rps')}"
    print(
        f"  {point:>10} win={m.get('window_s') and round(m['window_s']):>4}s n={m.get('window_records'):>6} "
        f"rps={m.get('achieved_rps') or 0:>6.2f} tok/s={m.get('output_tok_per_s') or 0:>6.0f} "
        f"ttft_p95={m.get('ttft_p95_s') or 0:>7.3f} tpot_p95={(m.get('tpot_p95_s') or 0) * 1000:>5.1f}ms "
        f"att={s.get('attainment_offered') if s.get('attainment_offered') is None else round(s['attainment_offered'], 3):>5} "
        f"probe={(m.get('probe_tpot_median_s') or 0) * 1000:>5.2f}ms W/util={pw.get('w_per_util_point')} "
        f"W={pw.get('mean_w')} tok/Wh={pw.get('output_tok_per_wh')} committed={hb} warm_n={w.get('n')}"
    )
PYEOF
}

if [ "$MODE" = "--seeds" ]; then
  for seed in "$TARGET"/seed-*; do
    [ -d "$seed" ] && summarise "$seed"
  done
else
  summarise "$TARGET"
fi
date +%H:%M:%S
