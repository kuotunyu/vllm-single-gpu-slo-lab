"""Admission-trace analysis for W3 (spec §3.3, §3.5, §8; protocol in ADR 0012).

One trace stage = one (cell, policy, seed): the 25-minute burst trace replayed through the shim.
Everything here is recomputed from the committed evidence of a stage (``records.jsonl[.gz]``,
``metrics.csv``, ``shim.csv``, ``manifest.json``), so ``make reproduce`` rebuilds the tables.

- Phase summaries: attainment over offered requests (429 / timeout / 5xx are misses), goodput,
  rejection rate, and TTFT / TPOT p95 of served requests, per phase and for the whole trace. The
  pre-burst phase skips its first 60 s, the same discard as the W2 sweeps; burst and recovery do
  not, because the transition is what they measure.
- Time-to-recover (spec §3.3): seconds from the end of the burst to the first 5 s grid point where
  the total queue (vLLM waiting + shim waiting) is empty and the TTFT p95 of requests offered in
  the next 60 s is within the SLO. A cap-and-reject policy can satisfy that at once while having
  turned half the burst away, so the attainment-based variant (first point where offered
  attainment over the next 60 s reaches 95 %) is reported beside it.
- Policy table: per (cell, policy) the per-seed values and their mean, and per seed the paired
  difference against native queueing (``passthrough``), since the three policies of a seed replay
  the identical trace file.
"""

from __future__ import annotations

import bisect
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slo_lab.batch_analysis import PROBE_DRIFT_MAX, _committed_vram_mb, md_table
from slo_lab.slo import (
    DEFAULT_SLO,
    Outcome,
    RequestRecord,
    Slo,
    evidence_path,
    meets_slo,
    read_records_jsonl,
    summarise,
)
from slo_lab.stats import percentile

POLICY_ORDER = ("passthrough", "hard_cap", "bounded_queue")
PRE_DISCARD_S = 60.0
RECOVERY_WINDOW_S = 60.0
RECOVERY_STEP_S = 5.0
RECOVERY_ATTAINMENT = 0.95
PAIRED_METRICS = (
    "attainment",
    "goodput_rps",
    "rejection_rate",
    "attainment_burst",
    "attainment_recovery",
    "ttft_p95_burst_s",
    "time_to_recover_s",
    "time_to_recover_attainment_s",
)


def _r(v: float | None, nd: int = 6) -> float | None:
    return None if v is None else round(float(v), nd)


def phase_summary(
    records: Sequence[RequestRecord], start_s: float, end_s: float, slo: Slo = DEFAULT_SLO
) -> dict[str, Any]:
    """Offered-denominator summary of requests offered in ``[start_s, end_s)``."""
    sel = [r for r in records if start_s <= r.offered_at_s < end_s]
    base: dict[str, Any] = {"start_s": start_s, "end_s": end_s, "offered": len(sel)}
    if not sel:
        return base | {
            "met": 0,
            "rejected_429": 0,
            "timeouts": 0,
            "errors": 0,
            "attainment": None,
            "attainment_ci95": None,
            "goodput_rps": None,
            "rejection_rate": None,
            "ttft_p95_s": None,
            "tpot_p95_s": None,
        }
    s = summarise(sel, slo, window_s=end_s - start_s)
    ok = [r for r in sel if r.outcome is Outcome.OK and r.ttft_s is not None]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpots = [t for t in (r.tpot_s for r in ok) if t is not None]
    return base | {
        "met": s.met,
        "rejected_429": s.rejected_429,
        "timeouts": s.timeouts,
        "errors": s.errors,
        "attainment": _r(s.attainment_offered),
        "attainment_ci95": [_r(s.attainment_offered_ci95[0]), _r(s.attainment_offered_ci95[1])],
        "goodput_rps": _r(s.goodput_rps),
        "rejection_rate": _r(s.rejection_rate),
        "ttft_p95_s": _r(percentile(ttfts, 95)) if ttfts else None,
        "tpot_p95_s": _r(percentile(tpots, 95)) if tpots else None,
    }


def _read_csv_series(path: Path, t_col: str, v_col: str) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    out: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append((float(row[t_col]), float(row[v_col])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out)


def _latest_at_or_before(series: list[tuple[float, float]], t: float) -> float | None:
    i = bisect.bisect_right(series, (t, float("inf"))) - 1
    return series[i][1] if i >= 0 else None


def _has_column(path: Path, column: str) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as fh:
        return column in fh.readline().strip().split(",")


def waiting_series(
    stage_dir: Path, *, origin_mono: float | None = None, origin_unix: float | None = None
) -> list[tuple[float, float]]:
    """``(seconds from trace start, vLLM waiting + shim waiting)`` at each ``/metrics`` sample.

    Samples are put on the records' axis with the monotonic clock inference-perf uses
    (``t_mono``). In WSL2 the wall clock drifted ~7 s against it within one 2.5-minute stage
    (W3 dry run, 2026-09-11), so ``t_unix`` plus a clock offset is only a fallback for files
    written before ``t_mono`` existed.
    """
    metrics, shim_path = stage_dir / "metrics.csv", stage_dir / "shim.csv"
    if origin_mono is not None and _has_column(metrics, "t_mono"):
        t_col, origin = "t_mono", origin_mono
    elif origin_unix is not None:
        t_col, origin = "t_unix", origin_unix
    else:
        return []
    engine = _read_csv_series(metrics, t_col, "num_requests_waiting")
    shim = _read_csv_series(shim_path, t_col, "waiting")
    out = []
    for t, waiting in engine:
        queued = _latest_at_or_before(shim, t) or 0.0
        out.append((round(t - origin, 3), waiting + queued))
    return out


def _grid(burst_end_s: float, trace_end_s: float) -> list[float]:
    points, t = [], burst_end_s
    while t + RECOVERY_WINDOW_S <= trace_end_s + 1e-9:
        points.append(round(t, 6))
        t += RECOVERY_STEP_S
    return points


def time_to_recover(
    records: Sequence[RequestRecord],
    waiting: Sequence[tuple[float, float]],
    *,
    burst_end_s: float,
    trace_end_s: float,
    slo: Slo = DEFAULT_SLO,
) -> float | None:
    """Spec §3.3: queue empty and next-60 s TTFT p95 within the SLO; None if never."""
    served = sorted(
        (r.offered_at_s, r.ttft_s)
        for r in records
        if r.outcome is Outcome.OK and r.ttft_s is not None
    )
    offered = [o for o, _ in served]
    series = sorted(waiting)
    for t in _grid(burst_end_s, trace_end_s):
        queued = _latest_at_or_before(series, t)
        if queued is None or queued > 0:
            continue
        lo = bisect.bisect_left(offered, t)
        hi = bisect.bisect_left(offered, t + RECOVERY_WINDOW_S)
        block = [ttft for _, ttft in served[lo:hi]]
        if block and percentile(block, 95) <= slo.ttft_s:
            return round(t - burst_end_s, 6)
    return None


def time_to_recover_attainment(
    records: Sequence[RequestRecord],
    *,
    burst_end_s: float,
    trace_end_s: float,
    slo: Slo = DEFAULT_SLO,
) -> float | None:
    """First grid point after the burst whose next 60 s reach 95 % offered attainment."""
    ordered = sorted(records, key=lambda r: r.offered_at_s)
    offered = [r.offered_at_s for r in ordered]
    met = [meets_slo(r, slo) for r in ordered]
    for t in _grid(burst_end_s, trace_end_s):
        lo = bisect.bisect_left(offered, t)
        hi = bisect.bisect_left(offered, t + RECOVERY_WINDOW_S)
        if hi > lo and sum(met[lo:hi]) / (hi - lo) >= RECOVERY_ATTAINMENT:
            return round(t - burst_end_s, 6)
    return None


def _phases(m: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((m.get("trace") or {}).get("phases")) or [])


def stage_metrics(
    records: Sequence[RequestRecord],
    phases: Sequence[dict[str, Any]],
    waiting: Sequence[tuple[float, float]] | None,
) -> dict[str, Any]:
    """Whole-trace and per-phase summaries plus both recovery times for one trace stage."""
    trace_end = max(float(p["end_s"]) for p in phases)
    by_name = {p["phase"]: p for p in phases}
    out: dict[str, Any] = {"all": phase_summary(records, 0.0, trace_end)}
    for p in phases:
        start = float(p["start_s"])
        if p["phase"] == "pre":  # 60 s like W2; half the phase for a short smoke profile
            start += min(PRE_DISCARD_S, (float(p["end_s"]) - start) / 2)
        out[p["phase"]] = phase_summary(records, start, float(p["end_s"]))
    burst_end = float(by_name["burst"]["end_s"]) if "burst" in by_name else None
    ttr = ttr_att = None
    if burst_end is not None:
        if waiting:
            ttr = time_to_recover(records, waiting, burst_end_s=burst_end, trace_end_s=trace_end)
        ttr_att = time_to_recover_attainment(records, burst_end_s=burst_end, trace_end_s=trace_end)
    out["time_to_recover_s"] = ttr
    out["time_to_recover_attainment_s"] = ttr_att
    return out


def _origin_unix(m: dict[str, Any]) -> float | None:
    origin = m.get("records_origin_monotonic_s")
    offsets = [v for v in (m.get("clock_offset_s") or {}).values() if v is not None]
    if origin is None or not offsets:
        return None
    return float(origin) + sum(offsets) / len(offsets)


def _stage_row(m: dict[str, Any]) -> dict[str, Any] | None:
    stage_dir = Path(m["_dir"])
    records_path = stage_dir / "records.jsonl"
    phases = _phases(m)
    if evidence_path(records_path) is None or not phases:
        return None
    records = read_records_jsonl(records_path)
    waiting = waiting_series(
        stage_dir, origin_mono=m.get("records_origin_monotonic_s"), origin_unix=_origin_unix(m)
    )
    sm = stage_metrics(records, phases, waiting or None)

    def ph(name: str, key: str) -> Any:
        return (sm.get(name) or {}).get(key)

    wall = m.get("load_wall_s")
    cpu = m.get("shim_cpu_s")
    return {
        "cell": m.get("cell"),
        "policy": m.get("policy"),
        "seed": m.get("seed"),
        "records": len(records),
        "offered": ph("all", "offered"),
        "attainment": ph("all", "attainment"),
        "attainment_ci95": ph("all", "attainment_ci95"),
        "goodput_rps": ph("all", "goodput_rps"),
        "rejection_rate": ph("all", "rejection_rate"),
        "attainment_pre": ph("pre", "attainment"),
        "attainment_burst": ph("burst", "attainment"),
        "attainment_recovery": ph("recovery", "attainment"),
        "goodput_burst_rps": ph("burst", "goodput_rps"),
        "rejection_burst": ph("burst", "rejection_rate"),
        "ttft_p95_pre_s": ph("pre", "ttft_p95_s"),
        "ttft_p95_burst_s": ph("burst", "ttft_p95_s"),
        "tpot_p95_burst_s": ph("burst", "tpot_p95_s"),
        "ttft_p95_recovery_s": ph("recovery", "ttft_p95_s"),
        "time_to_recover_s": sm["time_to_recover_s"],
        "time_to_recover_attainment_s": sm["time_to_recover_attainment_s"],
        "shim_cpu_util": _r(cpu / wall, 4) if cpu is not None and wall else None,
        "probe_tpot_median_s": m.get("probe_tpot_median_s"),
        "windows_committed_mb": _committed_vram_mb(m),
        "physical_vram_mib": m.get("_physical_vram_mib"),
        "path": m.get("_path"),
    }


def _flag(rows: list[dict[str, Any]]) -> None:
    """Probe-TPOT drift and Windows VRAM oversubscription (W/util is not a trace-stage flag)."""
    for cell in {r["cell"] for r in rows}:
        mine = [r for r in rows if r["cell"] == cell]
        probes = [r["probe_tpot_median_s"] for r in mine if r.get("probe_tpot_median_s")]
        best = min(probes) if probes else None
        for r in mine:
            reasons: list[str] = []
            p = r.get("probe_tpot_median_s")
            if p is not None and best is not None and p > best * (1 + PROBE_DRIFT_MAX):
                reasons.append(f"probe TPOT {p * 1000:.1f} ms > best {best * 1000:.1f} ms +15%")
            committed, physical = r.get("windows_committed_mb"), r.get("physical_vram_mib")
            if committed is not None and physical is not None and committed > physical:
                reasons.append(f"Windows committed VRAM {committed:.0f} MB > physical")
            r["suspect"] = bool(reasons)
            r["suspect_reasons"] = reasons


def _policy_key(policy: str | None) -> int:
    return POLICY_ORDER.index(policy) if policy in POLICY_ORDER else len(POLICY_ORDER)


def _mean(values: list[float]) -> float | None:
    return _r(sum(values) / len(values)) if values else None


def _per_cell_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [r for r in rows if not r.get("suspect")]
    out: dict[str, Any] = {}
    for cell in sorted({r["cell"] for r in clean}):
        cell_rows = [r for r in clean if r["cell"] == cell]
        native = {r["seed"]: r for r in cell_rows if r["policy"] == "passthrough"}
        policies: dict[str, Any] = {}
        for policy in sorted({r["policy"] for r in cell_rows}, key=_policy_key):
            mine = sorted((r for r in cell_rows if r["policy"] == policy), key=lambda r: r["seed"])
            entry: dict[str, Any] = {
                "seeds": [r["seed"] for r in mine],
                "n": len(mine),
                "per_seed": {k: [r.get(k) for r in mine] for k in PAIRED_METRICS},
                "mean": {
                    k: _mean([r[k] for r in mine if r.get(k) is not None]) for k in PAIRED_METRICS
                },
            }
            if policy != "passthrough" and native:
                paired: dict[str, Any] = {}
                for k in PAIRED_METRICS:
                    diffs = [
                        _r(r[k] - native[r["seed"]][k])
                        for r in mine
                        if r["seed"] in native
                        and r.get(k) is not None
                        and native[r["seed"]].get(k) is not None
                    ]
                    values = [d for d in diffs if d is not None]
                    paired[k] = {
                        "per_seed": values,
                        "mean": _mean(values),
                        "consistent_sign": bool(values)
                        and (all(d > 0 for d in values) or all(d < 0 for d in values)),
                    }
                entry["paired_vs_passthrough"] = paired
            policies[policy] = entry
        out[cell] = policies
    return out


def analyze_trace(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows per (cell, policy, seed) and the per-policy table with paired differences."""
    rows = [row for m in manifests if m.get("kind") == "trace" for row in [_stage_row(m)] if row]
    rows.sort(key=lambda r: (str(r["cell"]), _policy_key(r["policy"]), r["seed"]))
    _flag(rows)
    return {"rows": rows, "per_cell_policy": _per_cell_policy(rows)}


STAGE_COLS = [
    "cell",
    "policy",
    "seed",
    "offered",
    "attainment",
    "attainment_pre",
    "attainment_burst",
    "attainment_recovery",
    "goodput_rps",
    "rejection_rate",
    "rejection_burst",
    "ttft_p95_burst_s",
    "tpot_p95_burst_s",
    "time_to_recover_s",
    "time_to_recover_attainment_s",
    "shim_cpu_util",
    "probe_tpot_median_s",
    "suspect",
]


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def write_trace_tables(out: Path, result: dict[str, Any]) -> None:
    """``admission.json`` and ``tables.md`` under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "admission.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    lines = ["## Admission trace: one row per (cell, policy, seed)", ""]
    lines.append(md_table(result["rows"], STAGE_COLS))
    for cell, policies in result["per_cell_policy"].items():
        lines += ["", f"## {cell}: mean over seeds (per-seed values in brackets)", ""]
        lines.append("| policy | n | " + " | ".join(PAIRED_METRICS) + " |")
        lines.append("|---|---|" + "---|" * len(PAIRED_METRICS))
        for policy, e in policies.items():
            cells = [
                f"{_fmt(e['mean'][k])} ({'/'.join(_fmt(v) for v in e['per_seed'][k])})"
                for k in PAIRED_METRICS
            ]
            lines.append(f"| {policy} | {e['n']} | " + " | ".join(cells) + " |")
        diffs = [
            (p, e["paired_vs_passthrough"])
            for p, e in policies.items()
            if "paired_vs_passthrough" in e
        ]
        if diffs:
            lines += ["", f"## {cell}: paired difference vs passthrough (same trace per seed)", ""]
            lines.append(
                "| policy | metric | per-seed difference | mean | same sign in every seed |"
            )
            lines.append("|---|---|---|---|---|")
            for policy, paired in diffs:
                for k in PAIRED_METRICS:
                    d = paired[k]
                    per = "/".join(_fmt(v) for v in d["per_seed"])
                    lines.append(
                        f"| {policy} | {k} | {per or 'n/a'} | {_fmt(d['mean'])} | "
                        f"{'yes' if d['consistent_sign'] else 'no'} |"
                    )
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
