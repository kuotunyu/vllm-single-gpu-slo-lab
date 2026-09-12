"""Everything the W4 spec-decode scene draws, read once from analysis/tables (no manim import)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CELLS = (
    ("fp8", "fp8-ngram", "8B FP8 + n-gram"),
    ("q4b", "q4b-eagle3", "4B + EAGLE-3"),
    ("q4b", "q4b-ngram", "4B + n-gram"),
)
CONCURRENCIES = (1, 8, 32, 128, 256)
OPEN_MAX_MULTIPLIER = 0.5  # paired open-loop points are shown up to 0.5 x the base r_sat


@dataclass(frozen=True)
class OpenPoint:
    offered_rps: float
    tpot_p50_diff_ms: float
    tpot_p95_diff_ms: float
    attainment_diff: float
    p50_consistent: bool
    p95_consistent: bool


@dataclass(frozen=True)
class SpecCell:
    family: str
    name: str
    label: str
    acceptance: float
    mean_acceptance_length: float
    r_sat: float
    base_r_sat: float
    closed: dict[int, float | None]  # concurrency -> rps ratio vs none (None: excluded)
    open: list[OpenPoint]


@dataclass(frozen=True)
class SpecData:
    cells: list[SpecCell]


def _family(root: Path, family: str) -> dict[str, Any]:
    path = root / "analysis" / "tables" / f"w4-{family}-specdec" / "specdec.json"
    return json.loads(path.read_text(encoding="utf-8"))["families"][family]


def _open_points(cell: dict[str, Any], base_r_sat: float) -> list[OpenPoint]:
    out: list[OpenPoint] = []
    for _rate, v in sorted(cell["open_paired_by_rate"].items(), key=lambda kv: float(kv[0])):
        offered = float(v["offered_rps"])
        if offered > OPEN_MAX_MULTIPLIER * base_r_sat * 1.02:
            continue
        p50, p95, att = v["tpot_p50_diff_s"], v["tpot_p95_diff_s"], v["attainment_diff"]
        if p50["mean"] is None or p95["mean"] is None:
            continue
        out.append(
            OpenPoint(
                offered,
                1000.0 * float(p50["mean"]),
                1000.0 * float(p95["mean"]),
                float(att["mean"] or 0.0),
                bool(p50["consistent_sign"]),
                bool(p95["consistent_sign"]),
            )
        )
    return out


def load_specdec(root: Path) -> SpecData:
    cells: list[SpecCell] = []
    for family, name, label in CELLS:
        fam = _family(root, family)
        base = fam["cells"][fam["base_cell"]]
        cell = fam["cells"][name]
        closed: dict[int, float | None] = {c: None for c in CONCURRENCIES}
        for e in cell["closed_paired"]:
            if e.get("rps_ratio") is not None and int(e["concurrency"]) in closed:
                closed[int(e["concurrency"])] = float(e["rps_ratio"])
        cells.append(
            SpecCell(
                family,
                name,
                label,
                float(cell["acceptance_rate_mean"]),
                float(cell["mean_acceptance_length_mean"]),
                float(cell["r_sat_rps"]),
                float(base["r_sat_rps"]),
                closed,
                _open_points(cell, float(base["r_sat_rps"])),
            )
        )
    return SpecData(cells)
