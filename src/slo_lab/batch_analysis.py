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

from slo_lab.slo import (
    SweepCell,
    SweepRun,
    evidence_path,
    grid_markdown,
    open_evidence_text,
    r_slo,
    read_records_jsonl,
    sensitivity_grid,
)

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
    "windows_committed_mb",
    "suspect",
]
OPEN_COLS = [
    "cell",
    "seed",
    "offered_rps",
    "records",
    "achieved_rps",
    "served_rps",
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


def served_rps(records_path: Path, *, discard_first_s: float, window_end_s: float) -> float | None:
    """Completions per second inside the measurement window (open-loop service rate).

    ``achieved_rps`` counts a record at the moment it was *offered*, so an overloaded open-loop
    stage reports the load generator's rate: the FP8 2 x r_sat point of 2026-09-09 reads
    "87.5 rps achieved" while the server was actually finishing about 43 per second and the
    queue grew without bound. This counts a record when it *finished*, which is what the server
    delivered. Requests offered during the discard period but completing inside the window count,
    because they are service performed inside it; requests still running at the end do not.
    Closed-loop stages do not need this: a new request is only offered when one completes, so
    offered and served rates coincide by construction.
    """
    if evidence_path(records_path) is None or window_end_s <= discard_first_s:
        return None
    served = 0
    with open_evidence_text(records_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("outcome") != "ok":
                continue
            e2e = record.get("e2e_s")
            if e2e is None:
                continue
            finished = float(record["offered_at_s"]) + float(e2e)
            if discard_first_s <= finished < window_end_s:
                served += 1
    return served / (window_end_s - discard_first_s)


def _physical_vram_mib(stage_dir: Path) -> float | None:
    """Physical card size from the batch's ``quiet_gpu.json`` (NVML ``memory_total_mib``)."""
    for candidate in (stage_dir.parent / "quiet_gpu.json", stage_dir / "quiet_gpu.json"):
        if candidate.exists():
            try:
                total = json.loads(candidate.read_text(encoding="utf-8")).get("memory_total_mib")
                return float(total) if total else None
            except (OSError, ValueError):
                return None
    return None


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
            m["_physical_vram_mib"] = _physical_vram_mib(path.parent)
            m["_dir"] = str(path.parent)
            out.append(m)
    return out


def _sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """r_SLO for the TTFT x TPOT threshold grid, recomputed from the raw records of every
    non-suspect open-loop stage (spec §3.6: zero-cost re-analysis of the same evidence)."""
    runs: list[SweepRun] = []
    window_start = 0.0
    for r in rows:
        if r["suspect"] or not r.get("_dir"):
            continue
        records_path = Path(r["_dir"]) / "records.jsonl"
        if evidence_path(records_path) is None:
            continue
        runs.append(
            SweepRun(
                offered_rate=r["offered_rps"],
                seed=r["seed"],
                records=read_records_jsonl(records_path),
            )
        )
        window_start = float(r.get("discard_first_s") or 0.0)
    if not runs:
        return None
    grid = sensitivity_grid(runs, window_start_s=window_start)
    return {
        "r_slo_by_threshold": {f"ttft_s={t:g}|tpot_s={p:g}": v for (t, p), v in grid.items()},
        "markdown": grid_markdown(grid),
        "runs": len(runs),
    }


def _committed_vram_mb(m: dict[str, Any]) -> float | None:
    """Largest Windows-side ``committed_mb`` seen around the stage (host_before / host_after)."""
    values = [
        float(((m.get(key) or {}).get("windows_gpu_memory") or {}).get("committed_mb"))
        for key in ("host_before", "host_after")
        if ((m.get(key) or {}).get("windows_gpu_memory") or {}).get("committed_mb") is not None
    ]
    return max(values) if values else None


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
        committed, physical = r.get("windows_committed_mb"), r.get("physical_vram_mib")
        if committed is not None and physical is not None and committed > physical:
            reasons.append(
                f"Windows committed VRAM {committed:.0f} MB > physical {physical:.0f} MiB "
                "(VidMm paging, ADR 0007)"
            )
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
        "windows_committed_mb": _committed_vram_mb(m),
        "physical_vram_mib": m.get("_physical_vram_mib"),
        "server_ttft_p95_bounds_s": hist.get("p95_bounds_s"),
        "discard_first_s": m.get("discard_first_s"),
        "path": m.get("_path"),
        "_dir": m.get("_dir"),
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
            stage_dir = m.get("_dir")
            row["served_rps"] = (
                served_rps(
                    Path(stage_dir) / "records.jsonl",
                    discard_first_s=float(m.get("discard_first_s") or 0.0),
                    window_end_s=float(m.get("duration_s") or 0.0),
                )
                if stage_dir
                else None
            )
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
            "sensitivity": _sensitivity(rows),
        }
    for row in closed["rows"] + open_["rows"]:
        row.pop("_dir", None)  # filesystem detail, not evidence
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
        + json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "sensitivity"}
                for k, v in open_["per_cell"].items()
            },
            indent=2,
            default=str,
        )
        + "\n"
        + "".join(
            f"\n## SLO sensitivity: {cell} (r_SLO per TTFT x TPOT threshold, {v['sensitivity']['runs']} runs)\n\n"
            + v["sensitivity"]["markdown"]
            + "\n"
            for cell, v in open_["per_cell"].items()
            if v.get("sensitivity")
        )
    )
    (out / "tables.md").write_text(md, encoding="utf-8", newline="\n")


def run(batch_dirs: list[Path], out: Path) -> dict[str, Any]:
    """Load, analyze, write; returns a compact summary for the console."""
    manifests = load_manifests(batch_dirs)
    if manifests and all(m.get("kind") == "trace" for m in manifests):
        # W3 admission traces get their own table (ADR 0012); imported here because
        # admission_analysis itself builds on this module.
        from slo_lab.admission_analysis import analyze_trace, write_trace_tables

        result = analyze_trace(manifests)
        write_trace_tables(out, result)
        return {"manifests": len(manifests), "trace_rows": len(result["rows"])}
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
