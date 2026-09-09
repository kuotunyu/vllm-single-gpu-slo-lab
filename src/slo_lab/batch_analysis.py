"""Aggregate stage manifests from batch directories into closed-loop / open-loop tables.

Spec §3.2-§3.3. Reads every ``*/manifest.json`` under the given directories (one per stage,
written by ``slo-lab run-stage``) and produces:

- closed-loop: per (cell, seed, concurrency) achieved rps, TTFT/TPOT percentiles, attainment,
  window length, mean power and output tokens per Wh; per cell r_sat (maximum mean rps over the
  grid, flagged ``r_sat_is_lower_bound`` unless the top grid point gains < 5% over the one below
  it) and capacity C (highest concurrency whose window attainment >= 95%);
- open-loop: per (cell, seed, offered rate) attainment with Wilson CI, percentiles, goodput; per
  cell r_SLO across seeds via ``slo_lab.slo.r_slo``.

Tenancy: a stage whose GPU power signature (``w_per_util_point``) or single-stream TPOT probe
says the card was shared with another process is marked ``suspect`` and excluded from r_sat, C
and r_SLO (ADR 0006, 2026-09-09). ``make reproduce`` rebuilds every table listed in
``analysis/tables/index.json`` through :func:`write_tables` and diffs the result, so the output
must stay deterministic for a given evidence tree.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from slo_lab.slo import SweepCell, r_slo

PROBE_DRIFT_MAX = 0.15  # re-warm TPOT more than 15% above the batch's best probe -> suspect
W_PER_UTIL_MIN = 2.0  # host-calibrated (RTX 4090, ADR 0006): clean >= 2.3, foreign tenant ~1.8
PLATEAU_GAIN_MAX = 0.05  # top grid point must add < 5% over the previous one to count as a plateau

CLOSED_COLS = [
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
    "mean_util_pct",
    "w_per_util_point",
    "probe_tpot_median_s",
    "suspect",
]
OPEN_COLS = [
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
    "w_per_util_point",
    "probe_tpot_median_s",
    "suspect",
]


def signature_from_power_csv(path: Path, start_s: float) -> dict[str, float | None]:
    """Fallback for manifests written before ``power_window`` carried the tenancy signature."""
    empty: dict[str, float | None] = {"mean_util_pct": None, "w_per_util_point": None}
    if not path.exists():
        return empty
    utils: list[float] = []
    watts: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                if float(row["t_s"]) < start_s:
                    continue
                utils.append(float(row["util_gpu_pct"]))
                watts.append(float(row["power_w"]))
            except (KeyError, ValueError):
                continue
    if not utils:
        return empty
    mean_util = sum(utils) / len(utils)
    mean_w = sum(watts) / len(watts)
    return {
        "mean_util_pct": round(mean_util, 1),
        "w_per_util_point": round(mean_w / mean_util, 2) if mean_util else None,
    }


def load_manifests(dirs: list[Path]) -> list[dict[str, Any]]:
    """Every ``manifest.json`` below the given directories, in a deterministic order."""
    out: list[dict[str, Any]] = []
    for d in dirs:
        for path in sorted(d.rglob("manifest.json")):
            m = json.loads(path.read_text(encoding="utf-8"))
            rel = path.relative_to(d.parent) if path.is_relative_to(d.parent) else path
            m["_path"] = rel.as_posix()
            pw = m.get("power_window") or {}
            if pw.get("w_per_util_point") is None:
                sig = signature_from_power_csv(
                    path.parent / "power.csv", float(m.get("discard_first_s") or 0.0)
                )
                m["power_window"] = {**pw, **sig, "signature_source": "power.csv (fallback)"}
            out.append(m)
    return out


def flag_suspects(rows: list[dict[str, Any]]) -> None:
    """Mark stages whose tenancy probe or power signature says the card was shared."""
    probes = [r["probe_tpot_median_s"] for r in rows if r.get("probe_tpot_median_s")]
    best = min(probes) if probes else None
    for r in rows:
        reasons: list[str] = []
        p = r.get("probe_tpot_median_s")
        if p is not None and best is not None and p > best * (1 + PROBE_DRIFT_MAX):
            reasons.append(f"probe TPOT {p * 1000:.1f} ms > best {best * 1000:.1f} ms +15%")
        w = r.get("w_per_util_point")
        if w is not None and w < W_PER_UTIL_MIN:
            reasons.append(f"W per util point {w} < {W_PER_UTIL_MIN}")
        r["suspect"] = bool(reasons)
        r["suspect_reasons"] = reasons


def _row(m: dict[str, Any]) -> dict[str, Any]:
    s = m.get("summary") or {}
    pw = m.get("power_window") or {}
    hist = (m.get("server_histograms") or {}).get("time_to_first_token_seconds") or {}
    return {
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
        "power_mean_w": pw.get("mean_w"),
        "tok_per_wh": pw.get("output_tok_per_wh"),
        "mean_util_pct": pw.get("mean_util_pct"),
        "w_per_util_point": pw.get("w_per_util_point"),
        "probe_tpot_median_s": m.get("probe_tpot_median_s"),
        "server_ttft_p95_bounds_s": hist.get("p95_bounds_s"),
        "path": m.get("_path"),
    }


def analyze(manifests: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    closed: dict[str, Any] = {"rows": [], "per_cell": {}}
    open_: dict[str, Any] = {"rows": [], "per_cell": {}}
    by_cell_cl: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell_ol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in manifests:
        row = _row(m)
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
    for rows in list(by_cell_cl.values()) + list(by_cell_ol.values()):
        flag_suspects(rows)

    for cell, rows in sorted(by_cell_cl.items()):
        rows.sort(key=lambda r: (r["seed"], r["concurrency"]))
        by_conc: dict[int, list[float]] = defaultdict(list)
        att_by_conc: dict[int, list[float]] = defaultdict(list)
        suspects = sorted({r["concurrency"] for r in rows if r["suspect"]})
        for r in rows:
            if r["suspect"]:
                continue  # a shared card says nothing about the engine
            if r["achieved_rps"] is not None:
                by_conc[r["concurrency"]].append(r["achieved_rps"])
            if r["attainment"] is not None:
                att_by_conc[r["concurrency"]].append(r["attainment"])
        mean_rps = {c: sum(v) / len(v) for c, v in sorted(by_conc.items())}
        r_max = max(mean_rps.values()) if mean_rps else None
        plateau = [c for c, v in mean_rps.items() if r_max and v >= 0.95 * r_max]
        cap = [c for c, v in sorted(att_by_conc.items()) if v and min(v) >= 0.95]
        concs = sorted(mean_rps)
        gain = {
            cur: (mean_rps[cur] - mean_rps[prev]) / mean_rps[prev] if mean_rps[prev] else None
            for prev, cur in pairwise(concs)
        }
        top_gain = gain.get(concs[-1]) if concs else None
        plateau_reached = top_gain is not None and top_gain < PLATEAU_GAIN_MAX
        closed["per_cell"][cell] = {
            "r_sat_rps": r_max,
            "r_sat_is_lower_bound": not plateau_reached,
            "plateau_reached": plateau_reached,
            "plateau_first_concurrency": plateau[0] if plateau else None,
            "capacity_C": max(cap) if cap else None,
            "suspect_concurrencies_excluded": suspects,
            "mean_rps_by_concurrency": mean_rps,
            "gain_vs_prev_by_concurrency": gain,
            "min_attainment_by_concurrency": {c: min(v) for c, v in sorted(att_by_conc.items())},
        }
    for cell, rows in sorted(by_cell_ol.items()):
        rows.sort(key=lambda r: (r["seed"], r["offered_rps"]))
        cells = [
            SweepCell(offered_rate=r["offered_rps"], seed=r["seed"], attainment=r["attainment"])
            for r in rows
            if r["attainment"] is not None and not r["suspect"]
        ]
        result = r_slo(cells) if cells else None
        open_["per_cell"][cell] = {
            **(result.model_dump() if result else {}),
            "suspect_rates_excluded": sorted({r["offered_rps"] for r in rows if r["suspect"]}),
        }
    return closed, open_


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return "" if v is None else str(v)

    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n" for r in rows)


def write_tables(out: Path, closed: dict[str, Any], open_: dict[str, Any]) -> None:
    """``closed_loop.json``, ``open_loop.json`` and ``tables.md`` under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    # LF on every platform so `make reproduce` diffs cleanly whether run on Windows or CI
    (out / "closed_loop.json").write_text(
        json.dumps(closed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "open_loop.json").write_text(
        json.dumps(open_, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md = (
        "# Closed-loop\n\n"
        + md_table(closed["rows"], CLOSED_COLS)
        + "\n"
        + json.dumps(closed["per_cell"], indent=2)
        + "\n\n# Open-loop\n\n"
        + md_table(open_["rows"], OPEN_COLS)
        + "\n"
        + json.dumps(open_["per_cell"], indent=2, default=str)
        + "\n"
    )
    (out / "tables.md").write_text(md, encoding="utf-8", newline="\n")


def run(batch_dirs: list[Path], out: Path) -> dict[str, Any]:
    """Load, analyze, write; returns a compact summary for the console."""
    manifests = load_manifests(batch_dirs)
    closed, open_ = analyze(manifests)
    write_tables(out, closed, open_)
    return {
        "manifests": len(manifests),
        "closed_rows": len(closed["rows"]),
        "open_rows": len(open_["rows"]),
        "closed_per_cell": closed["per_cell"],
        "open_per_cell": {k: v.get("r_slo") for k, v in open_["per_cell"].items()},
    }


def rebuild_from_index(root: Path, index_path: Path) -> list[str]:
    """Rebuild every table listed in ``analysis/tables/index.json``; returns the names rebuilt."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rebuilt: list[str] = []
    for name, dirs in index.items():
        if name.startswith("_"):
            continue
        run([root / d for d in dirs], index_path.parent / name)
        rebuilt.append(name)
    return rebuilt
