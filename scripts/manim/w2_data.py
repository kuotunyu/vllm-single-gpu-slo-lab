"""Everything the W2 knee scene draws, read once from analysis/tables (no manim import)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slo_lab.plots import attainment_series, tpot_series

CELLS = ("bf16", "fp8", "awq", "gptq")
LABELS = {"bf16": "BF16", "fp8": "FP8", "awq": "AWQ", "gptq": "GPTQ-Int4"}
OPEN_TABLES = {c: f"w2-{c}-open-loop" for c in CELLS}
CLOSED_TABLES = {
    "bf16": "w2-bf16-closed-loop",
    "fp8": "w2-fp8-closed-loop-v3",
    "awq": "w2-awq-closed-loop",
    "gptq": "w2-gptq-closed-loop",
}


@dataclass(frozen=True)
class Cell:
    name: str
    label: str
    attainment: list[tuple[float, float]]  # (offered rps, min attainment over seeds)
    tpot_p95_ms: list[tuple[float, float]]  # (offered rps, mean over seeds)
    ttft_p95_s: list[tuple[float, float]]
    r_slo: float
    r_sat: float
    wh_per_m_tokens: float
    tmmlu_accuracy: float
    tmmlu_delta: float | None  # vs BF16, None for the baseline
    mcnemar_p: float | None


@dataclass(frozen=True)
class KneeData:
    cells: dict[str, Cell]
    sensitivity_fp8: dict[tuple[float, float], float]  # (ttft_s, tpot_s) -> r_slo
    rate_max: float


def _json(root: Path, table: str, name: str) -> dict[str, Any]:
    return json.loads((root / "analysis" / "tables" / table / name).read_text(encoding="utf-8"))


def _ttft_series(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    by_rate: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if not r.get("suspect") and r.get("ttft_p95_s") is not None:
            by_rate[float(r["offered_rps"])].append(float(r["ttft_p95_s"]))
    return [(rate, statistics.mean(v)) for rate, v in sorted(by_rate.items())]


def _wh_per_m_tokens(rows: list[dict[str, Any]], r_slo: float) -> float:
    at = [float(r["tok_per_wh"]) for r in rows if abs(float(r["offered_rps"]) - r_slo) < 0.01]
    return 1e6 / statistics.mean(at)


def _sensitivity(per_cell: dict[str, Any]) -> dict[tuple[float, float], float]:
    out: dict[tuple[float, float], float] = {}
    for key, value in per_cell["sensitivity"]["r_slo_by_threshold"].items():
        ttft, tpot = (float(part.split("=")[1]) for part in key.split("|"))
        out[(ttft, tpot)] = float(value)
    return out


def load_knee(root: Path) -> KneeData:
    quality = {r["b"]: r for r in _json(root, "w2-quality-paired", "paired.json")["rows"]}
    baseline_acc = next(iter(quality.values()))["a_accuracy"]
    cells: dict[str, Cell] = {}
    sensitivity: dict[tuple[float, float], float] = {}
    rate_max = 0.0
    for name in CELLS:
        table_dir = root / "analysis" / "tables" / OPEN_TABLES[name]
        open_json = _json(root, OPEN_TABLES[name], "open_loop.json")
        per_cell = open_json["per_cell"][name]
        att = next(s for s in attainment_series([table_dir]) if s.name == name).points
        tpot = next(s for s in tpot_series([table_dir]) if s.name == name).points
        rows = open_json["rows"]
        r_slo = float(per_cell["r_slo"])
        closed = _json(root, CLOSED_TABLES[name], "closed_loop.json")["per_cell"][name]
        q = quality.get(name)
        cells[name] = Cell(
            name,
            LABELS[name],
            att,
            tpot,
            _ttft_series(rows),
            r_slo,
            float(closed["r_sat_rps"]),
            _wh_per_m_tokens(rows, r_slo),
            float(q["b_accuracy"]) if q else float(baseline_acc),
            float(q["delta_b_minus_a"]) if q else None,
            float(q["mcnemar_exact_p"]) if q else None,
        )
        rate_max = max(rate_max, max(r for r, _ in att))
        if name == "fp8":
            sensitivity = _sensitivity(per_cell)
    return KneeData(cells, sensitivity, rate_max)
