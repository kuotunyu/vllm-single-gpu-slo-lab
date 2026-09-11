"""Paired speculative-decoding analysis for W4 (spec §3.1; protocol in ADR 0015).

Cells are labelled ``<family>-<specdec>`` and every W4 stage manifest carries ``specdec`` and
``family``. Within a family the ``none`` cell is the baseline; every other cell is compared with
it at equal (seed, concurrency) for closed-loop stages and equal (seed, offered rate) for
open-loop stages. That is a paired design: the open-loop rates of a family are all scaled from
the baseline's r_sat and a seed fixes the prompt sequence (ADR 0015 item 3). Everything here is
derived from the closed-loop / open-loop tables that :mod:`slo_lab.batch_analysis` builds from
the committed manifests, so ``make reproduce`` rebuilds these tables too. A stage flagged as
suspect on either side of a pair is left out of the pairing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from slo_lab.batch_analysis import md_table

BASELINE = "none"
CLOSED_PAIR_KEYS = ("concurrency", "seed")
OPEN_PAIR_KEYS = ("offered_rps", "seed")
# accelerated minus baseline (seconds) / accelerated over baseline (ratio); the key names are the
# manifest row fields of batch_analysis
DIFF_METRICS = {
    "tpot_p50_diff_s": "tpot_p50_s",
    "tpot_p95_diff_s": "tpot_p95_s",
    "ttft_p95_diff_s": "ttft_p95_s",
}
OPEN_DIFF_METRICS = {"attainment_diff": "attainment", **DIFF_METRICS}
RATIO_METRICS = {
    "rps_ratio": "achieved_rps",
    "output_tok_per_s_ratio": "output_tok_per_s",
    "tok_per_wh_ratio": "tok_per_wh",
}
OPEN_BY_RATE_METRICS = (*OPEN_DIFF_METRICS, *RATIO_METRICS)


def _r(v: float | None, nd: int = 6) -> float | None:
    return None if v is None else round(float(v), nd)


def _mean(values: list[float]) -> float | None:
    return _r(sum(values) / len(values)) if values else None


def _consistent(metric: str, values: list[float]) -> bool:
    """Every seed on the same side of 0 (differences) or of 1 (ratios)."""
    if not values:
        return False
    pivot = 1.0 if metric.endswith("_ratio") else 0.0
    return all(v > pivot for v in values) or all(v < pivot for v in values)


def _pair(row: dict[str, Any], base: dict[str, Any], diffs: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "acceptance_rate": row.get("acceptance_rate"),
        "mean_acceptance_length": row.get("mean_acceptance_length"),
    }
    for name, key in diffs.items():
        a, b = row.get(key), base.get(key)
        out[name] = _r(a - b) if a is not None and b is not None else None
    for name, key in RATIO_METRICS.items():
        a, b = row.get(key), base.get(key)
        out[name] = _r(a / b) if a is not None and b else None
    return out


def _clean_by_key(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple, dict[str, Any]]:
    return {tuple(r.get(k) for k in keys): r for r in rows if not r.get("suspect")}


def _by_rate(open_paired: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for p in open_paired:
        grouped[p["offered_rps"]].append(p)
    out: dict[str, Any] = {}
    for rate in sorted(grouped):
        pairs = sorted(grouped[rate], key=lambda p: p["seed"])
        entry: dict[str, Any] = {"offered_rps": rate, "seeds": [p["seed"] for p in pairs]}
        for metric in OPEN_BY_RATE_METRICS:
            values = [p[metric] for p in pairs if p.get(metric) is not None]
            entry[metric] = {
                "per_seed": values,
                "mean": _mean(values),
                "consistent_sign": _consistent(metric, values),
            }
        entry["acceptance_rate_mean"] = _mean(
            [p["acceptance_rate"] for p in pairs if p.get("acceptance_rate") is not None]
        )
        out[f"{rate:g}"] = entry
    return out


def analyze_specdec(closed: dict[str, Any], open_: dict[str, Any]) -> dict[str, Any]:
    """Per family: the baseline cell and, per cell, its own capacity numbers plus the pairs."""
    labels: dict[str, dict[str, str]] = {}
    for r in closed["rows"] + open_["rows"]:
        if r.get("specdec") is None or r.get("family") is None:
            continue
        labels.setdefault(r["cell"], {"family": r["family"], "specdec": r["specdec"]})
    members: dict[str, dict[str, str]] = defaultdict(dict)
    for cell, info in sorted(labels.items()):
        members[info["family"]][cell] = info["specdec"]

    families: dict[str, Any] = {}
    for family, cells in sorted(members.items()):
        base = next((c for c, s in cells.items() if s == BASELINE), None)
        base_closed = _clean_by_key(
            [r for r in closed["rows"] if r["cell"] == base], CLOSED_PAIR_KEYS
        )
        base_open = _clean_by_key([r for r in open_["rows"] if r["cell"] == base], OPEN_PAIR_KEYS)
        base_r_sat = (closed["per_cell"].get(base) or {}).get("r_sat_rps") if base else None
        entries: dict[str, Any] = {}
        for cell, specdec in cells.items():
            cl_rows = [r for r in closed["rows"] if r["cell"] == cell]
            ol_rows = [r for r in open_["rows"] if r["cell"] == cell]
            cl_cell = closed["per_cell"].get(cell) or {}
            ol_cell = open_["per_cell"].get(cell) or {}
            clean = [r for r in cl_rows + ol_rows if not r.get("suspect")]
            closed_paired: list[dict[str, Any]] = []
            open_paired: list[dict[str, Any]] = []
            if cell != base:
                for key, r in sorted(_clean_by_key(cl_rows, CLOSED_PAIR_KEYS).items()):
                    b = base_closed.get(key)
                    if b is not None:
                        closed_paired.append(
                            {"concurrency": key[0], "seed": key[1], **_pair(r, b, DIFF_METRICS)}
                        )
                for key, r in sorted(_clean_by_key(ol_rows, OPEN_PAIR_KEYS).items()):
                    b = base_open.get(key)
                    if b is not None:
                        open_paired.append(
                            {
                                "offered_rps": key[0],
                                "seed": key[1],
                                **_pair(r, b, OPEN_DIFF_METRICS),
                            }
                        )
            r_sat = cl_cell.get("r_sat_rps")
            entries[cell] = {
                "specdec": specdec,
                "is_baseline": cell == base,
                "r_sat_rps": r_sat,
                "r_sat_is_lower_bound": cl_cell.get("r_sat_is_lower_bound"),
                "r_sat_vs_base": _r(r_sat / base_r_sat)
                if cell != base and r_sat is not None and base_r_sat
                else None,
                "capacity_C": cl_cell.get("capacity_C"),
                "r_slo": ol_cell.get("r_slo"),
                "acceptance_rate_mean": _mean(
                    [r["acceptance_rate"] for r in clean if r.get("acceptance_rate") is not None]
                ),
                "mean_acceptance_length_mean": _mean(
                    [
                        r["mean_acceptance_length"]
                        for r in clean
                        if r.get("mean_acceptance_length") is not None
                    ]
                ),
                "closed_paired": closed_paired,
                "open_paired": open_paired,
                "open_paired_by_rate": _by_rate(open_paired),
            }
        families[family] = {"base_cell": base, "cells": entries}
    return {"families": families}


CELL_COLS = [
    "cell",
    "specdec",
    "r_sat_rps",
    "r_sat_vs_base",
    "capacity_C",
    "r_slo",
    "acceptance_rate_mean",
    "mean_acceptance_length_mean",
]
CLOSED_PAIR_COLS = [
    "concurrency",
    "seed",
    "rps_ratio",
    "output_tok_per_s_ratio",
    "tpot_p50_diff_s",
    "tpot_p95_diff_s",
    "ttft_p95_diff_s",
    "tok_per_wh_ratio",
    "acceptance_rate",
    "mean_acceptance_length",
]


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def write_specdec_tables(out: Path, result: dict[str, Any]) -> None:
    """``specdec.json`` plus a ``# Speculative decoding`` section appended to ``tables.md``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "specdec.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    lines = ["", "# Speculative decoding", ""]
    for family, entry in result["families"].items():
        lines += [f"## {family}: baseline {entry['base_cell'] or 'missing'}", ""]
        rows = [{"cell": cell, **e} for cell, e in entry["cells"].items()]
        lines.append(md_table(rows, CELL_COLS))
        for cell, e in entry["cells"].items():
            if e["is_baseline"] or not entry["base_cell"]:
                continue
            base = entry["base_cell"]
            if e["closed_paired"]:
                lines += ["", f"### {cell} vs {base}: closed-loop, per (concurrency, seed)", ""]
                lines.append(md_table(e["closed_paired"], CLOSED_PAIR_COLS))
            if e["open_paired_by_rate"]:
                lines += [
                    "",
                    f"### {cell} vs {base}: open-loop, mean over seeds "
                    "(per-seed values in brackets; sign = same side in every seed)",
                    "",
                    "| offered rps | n | "
                    + " | ".join(OPEN_BY_RATE_METRICS)
                    + " | acceptance rate |",
                    "|---|---|" + "---|" * (len(OPEN_BY_RATE_METRICS) + 1),
                ]
                for rate in e["open_paired_by_rate"].values():
                    cells = [
                        f"{_fmt(rate[m]['mean'])} ({'/'.join(_fmt(v) for v in rate[m]['per_seed'])})"
                        f"{' sign' if rate[m]['consistent_sign'] else ''}"
                        for m in OPEN_BY_RATE_METRICS
                    ]
                    lines.append(
                        f"| {_fmt(rate['offered_rps'])} | {len(rate['seeds'])} | "
                        + " | ".join(cells)
                        + f" | {_fmt(rate['acceptance_rate_mean'])} |"
                    )
        lines.append("")
    with (out / "tables.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
