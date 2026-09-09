#!/usr/bin/env bash
# Warm-up sufficiency, server-side vs client-side TTFT per stage, host state, and I/O pressure summary.
set -uo pipefail
root="${1:-$HOME/vllm-slo-lab/runs-w2/closed-loop/fp8/seed-1}"
"$HOME/vllm-slo-lab/.venv-slolab/bin/python" - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
stages = sorted((d for d in root.iterdir() if (d / "manifest.json").exists()), key=lambda d: int(d.name.split("-")[-1]))
first = json.loads((stages[0] / "manifest.json").read_text())
w = first["warmup"]
print("WARM-UP (stage", stages[0].name, "): n", w["n"], "wall", round(w["wall_s"]), "s | medians 1-100 / 101-200 / 201-300:",
      [round(w[k], 4) if w[k] else None for k in ("median_1_100", "median_101_200", "median_201_300")])
if w["median_101_200"] and w["median_201_300"]:
    diff = abs(w["median_201_300"] - w["median_101_200"]) / w["median_101_200"]
    print("  |201-300 vs 101-200| =", f"{diff:.1%}", "-> 100 sufficient" if diff <= 0.05 else "-> raise to 200")
tt = json.loads((stages[0] / "warmup-ttft.json").read_text())["ttft_s"]
blocks = [(i, sorted(tt[i:i+50])) for i in range(0, len(tt), 50)]
print("  50-block medians:", [(i, round(b[len(b)//2], 4)) for i, b in blocks])
print("  first 5 ttft:", [round(x, 3) for x in tt[:5]], "| max after 10:", round(max(tt[10:]), 3))
print()
print(f"{'stage':>12} {'win_rec':>7} {'win_s':>6} {'rps':>6} {'c_ttft_p95':>10} {'srv_ttft_p95_bounds':>22} {'srv_queue_p95':>18} {'srv_tpot_p95':>16} {'meanW':>6} {'tok/Wh':>7} {'temp':>5} {'load1 b/a':>12} {'io_before':>28}")
for d in stages:
    m = json.loads((d / "manifest.json").read_text())
    sh = m.get("server_histograms") or {}
    def b(name, q="p95_bounds_s"):
        x = (sh.get(name) or {}).get(q); return f"{x[0]}-{x[1]}" if x else "-"
    pw = m.get("power_window") or {}
    hb = (m.get("host_before") or {}).get("loadavg") or [None]; ha = (m.get("host_after") or {}).get("loadavg") or [None]
    print(f"{d.name:>12} {m.get('window_records'):>7} {m.get('window_s') and round(m['window_s']):>6} {m.get('achieved_rps') and round(m['achieved_rps'],2):>6} "
          f"{m.get('ttft_p95_s') and round(m['ttft_p95_s'],3):>10} {b('time_to_first_token_seconds'):>22} {b('request_queue_time_seconds'):>18} {b('request_time_per_output_token_seconds'):>16} "
          f"{pw.get('mean_w'):>6} {pw.get('output_tok_per_wh'):>7} {pw.get('max_temp_c'):>5} {str(hb[0])+'/'+str(ha[0]):>12} {str(m.get('io_pressure_before'))[:28]:>28}")
    # preemption / waiting evidence
print()
print("server-side TTFT count per stage vs window records (histogram covers the whole stage incl. discard):")
for d in stages:
    m = json.loads((d / "manifest.json").read_text())
    sh = (m.get("server_histograms") or {}).get("time_to_first_token_seconds") or {}
    print(" ", d.name, "count", sh.get("count"), "records", m.get("records"), "| queue p50", ((m.get("server_histograms") or {}).get("request_queue_time_seconds") or {}).get("p50_bounds_s"))
PY
echo "== preemption / waiting in serve.log =="
grep -c -i "preempt" "$root/serve.log"; grep -o "Waiting: [0-9]* reqs" "$root/serve.log" | sort | uniq -c | sort -rn | head -4
grep -o "GPU KV cache usage: [0-9.]*%" "$root/serve.log" | sort -t' ' -k5 -n | tail -1
echo "== io pressure: samples>5% during batch =="; awk 'NR>1 && ($2+0>5 || $3+0>5)' "$root/io-pressure.log" | wc -l; wc -l < "$root/io-pressure.log"
