"""Aggregate stage manifests from one or more batch directories into tables (spec §3.2, §3.3).

usage: python scripts/analyze_batch.py <batch_dir> [<batch_dir> ...] --out analysis/tables/<name>

Reads every ``*/manifest.json`` under the given directories (one per stage). Produces:

- ``closed_loop.json`` / ``.md``: per (cell, seed, concurrency) achieved rps, TTFT/TPOT p95,
  attainment, window length, mean power and tokens per Wh; r_sat per cell (throughput plateau:
  highest concurrency whose achieved rps is within 5% of the maximum, reporting that maximum,
  flagged as a lower bound unless the top grid point gains < 5% over the previous one), and
  capacity C (highest concurrency whose window attainment >= 95%).
- ``open_loop.json`` / ``.md``: per (cell, seed, offered rate) attainment with Wilson CI, TTFT/TPOT
  p95, achieved rps; r_SLO per cell across seeds via ``slo_lab.slo.r_slo``.

Nothing here touches raw inference-perf output; it only reads manifests and records.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from slo_lab.slo import SweepCell, r_slo


def _load_manifests(dirs: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in dirs:
        for path in sorted(d.rglob("manifest.json")):
            m = json.loads(path.read_text(encoding="utf-8"))
            m["_path"] = str(path.relative_to(d.parent) if path.is_relative_to(d.parent) else path)
            out.append(m)
    return out


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return "" if v is None else str(v)

    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n" for r in rows)


def analyze(manifests: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    closed: dict[str, Any] = {"rows": [], "per_cell": {}}
    open_: dict[str, Any] = {"rows": [], "per_cell": {}}
    by_cell_cl: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell_ol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in manifests:
        s = m.get("summary") or {}
        row = {
            "cell": m["cell"],
            "seed": m["seed"],
            "records": m.get("records"),
            "achieved_rps": m.get("achieved_rps"),
            "output_tok_per_s": m.get("output_tok_per_s"),
            "ttft_p50_s": m.get("ttft_p50_s"),
            "ttft_p95_s": m.get("ttft_p95_s"),
            "tpot_p50_s": m.get("tpot_p50_s"),
            "tpot_p95_s": m.get("tpot_p95_s"),
            "attainment": s.get("attainment_offered"),
            "attainment_ci95": s.get("attainment_offered_ci95"),
            "rejection_rate": s.get("rejection_rate"),
            "goodput_rps": s.get("goodput_rps"),
            "warmup_ttft_median_s": m.get("warmup_ttft_median_s"),
            "window_records": m.get("window_records"),
            "window_s": m.get("window_s"),
            "power_mean_w": (m.get("power_window") or {}).get("mean_w"),
            "tok_per_wh": (m.get("power_window") or {}).get("output_tok_per_wh"),
            "server_ttft_p95_bounds_s": (
                (m.get("server_histograms") or {}).get("time_to_first_token_seconds") or {}
            ).get("p95_bounds_s"),
            "path": m.get("_path"),
        }
        if m.get("kind") == "closed_loop":
            row["concurrency"] = m.get("concurrency")
            closed["rows"].append(row)
            by_cell_cl[m["cell"]].append(row)
        elif m.get("kind") == "open_loop":
            row["offered_rps"] = m.get("rate_rps")
            open_["rows"].append(row)
            by_cell_ol[m["cell"]].append(row)
    closed["rows"].sort(key=lambda r: (r["cell"], r["seed"], r["concurrency"]))
    open_["rows"].sort(key=lambda r: (r["cell"], r["seed"], r["offered_rps"]))
    for cell, rows in by_cell_cl.items():
        rows.sort(key=lambda r: (r["seed"], r["concurrency"]))
        by_conc: dict[int, list[float]] = defaultdict(list)
        att_by_conc: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            if r["achieved_rps"] is not None:
                by_conc[r["concurrency"]].append(r["achieved_rps"])
            if r["attainment"] is not None:
                att_by_conc[r["concurrency"]].append(r["attainment"])
        mean_rps = {c: sum(v) / len(v) for c, v in by_conc.items()}
        r_max = max(mean_rps.values()) if mean_rps else None
        plateau = [c for c, v in sorted(mean_rps.items()) if r_max and v >= 0.95 * r_max]
        cap = [c for c, v in sorted(att_by_conc.items()) if v and min(v) >= 0.95]
        # A plateau is only observed when the top grid point adds < 5% over the one below it;
        # otherwise r_sat is a lower bound and the grid must be extended (W2 exploratory sweep:
        # c=128 still +43% over c=64).
        concs = sorted(mean_rps)
        gain = {
            cur: (mean_rps[cur] - mean_rps[prev]) / mean_rps[prev] if mean_rps[prev] else None
            for prev, cur in pairwise(concs)
        }
        top_gain = gain.get(concs[-1]) if concs else None
        plateau_reached = top_gain is not None and top_gain < 0.05
        closed["per_cell"][cell] = {
            "r_sat_rps": r_max,
            "r_sat_is_lower_bound": not plateau_reached,
            "plateau_reached": plateau_reached,
            "plateau_first_concurrency": plateau[0] if plateau else None,
            "capacity_C": max(cap) if cap else None,
            "mean_rps_by_concurrency": mean_rps,
            "gain_vs_prev_by_concurrency": gain,
            "min_attainment_by_concurrency": {c: min(v) for c, v in att_by_conc.items()},
        }
    for cell, rows in by_cell_ol.items():
        rows.sort(key=lambda r: (r["seed"], r["offered_rps"]))
        cells = [
            SweepCell(offered_rate=r["offered_rps"], seed=r["seed"], attainment=r["attainment"])
            for r in rows
            if r["attainment"] is not None
        ]
        result = r_slo(cells) if cells else None
        open_["per_cell"][cell] = result.model_dump() if result else None
    return closed, open_


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dirs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifests = _load_manifests([Path(d) for d in args.batch_dirs])
    closed, open_ = analyze(manifests)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop.json").write_text(
        json.dumps(closed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "open_loop.json").write_text(
        json.dumps(open_, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cl_cols = [
        "cell",
        "seed",
        "concurrency",
        "records",
        "window_records",
        "window_s",
        "achieved_rps",
        "output_tok_per_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "tpot_p50_s",
        "tpot_p95_s",
        "attainment",
        "power_mean_w",
        "tok_per_wh",
    ]
    ol_cols = [
        "cell",
        "seed",
        "offered_rps",
        "records",
        "achieved_rps",
        "ttft_p50_s",
        "ttft_p95_s",
        "tpot_p50_s",
        "tpot_p95_s",
        "attainment",
        "attainment_ci95",
        "goodput_rps",
    ]
    md = (
        "# Closed-loop\n\n"
        + _md_table(closed["rows"], cl_cols)
        + "\n"
        + json.dumps(closed["per_cell"], indent=2)
        + "\n\n# Open-loop\n\n"
        + _md_table(open_["rows"], ol_cols)
        + "\n"
        + json.dumps(open_["per_cell"], indent=2, default=str)
        + "\n"
    )
    (out / "tables.md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "manifests": len(manifests),
                "closed_rows": len(closed["rows"]),
                "open_rows": len(open_["rows"]),
                "closed_per_cell": closed["per_cell"],
                "open_per_cell": {
                    k: (v or {}).get("r_slo") if v else None for k, v in open_["per_cell"].items()
                },
            },
            indent=1,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
