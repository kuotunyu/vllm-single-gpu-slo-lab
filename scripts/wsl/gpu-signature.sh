#!/usr/bin/env bash
# GPU signature per stage for any batch root: util, SM clock, power, and W per util-point.
set -uo pipefail
root="$1"
"$HOME/vllm-slo-lab/.venv-slolab/bin/python" - "$root" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
stages = sorted((d for d in root.iterdir() if (d / "manifest.json").exists()), key=lambda d: int(d.name.split("-")[-1]))
print(f"{'stage':>12} {'n':>4} {'util%':>6} {'sm_mhz':>7} {'W':>6} {'W/util':>7} {'tok/s':>7} {'tpot_p50':>9} {'ttft_p95':>9}")
for d in stages:
    m = json.loads((d / "manifest.json").read_text())
    rows = [r for r in csv.DictReader((d / "power.csv").open()) if float(r["t_s"]) >= 20]
    util = sum(float(r["util_gpu_pct"]) for r in rows) / len(rows)
    clk = sum(float(r["clocks_sm_mhz"]) for r in rows) / len(rows)
    w = sum(float(r["power_w"]) for r in rows) / len(rows)
    print(f"{d.name:>12} {len(rows):>4} {util:>6.1f} {clk:>7.0f} {w:>6.1f} {w/util if util else 0:>7.2f} {m.get('output_tok_per_s') or 0:>7.0f} {m.get('tpot_p50_s') or 0:>9.4f} {m.get('ttft_p95_s') or 0:>9.3f}")
PY
