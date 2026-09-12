"""Everything the W3 explainer scene draws, computed once from the committed evidence.

Seed 1's replay through each policy gives the queue series and the 10 s buckets; the 3-seed
means come from ``analysis/tables``. No manim import, so ``tests/test_w3_anim_data.py`` checks
in CI that the values the animation shows are the tables' values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slo_lab.admission_analysis import waiting_series
from slo_lab.slo import read_records_jsonl
from slo_lab.timeline import Bucket, Phase, bucket_records, downsample, phases_of, rolling

POLICIES = ("passthrough", "hard_cap", "bounded_queue")
TRACE_END_S = 1500.0
QUEUE_STEP_S = 5.0
BUCKET_S = 10.0
ROLL = 3
_METRICS = (
    "attainment",
    "goodput_rps",
    "rejection_rate",
    "attainment_burst",
    "attainment_recovery",
    "ttft_p95_burst_s",
    "time_to_recover_s",
)


@dataclass(frozen=True)
class Lane:
    policy: str
    queue: list[tuple[float, float]]  # every QUEUE_STEP_S from 0 to TRACE_END_S
    buckets: list[Bucket]  # BUCKET_S buckets rolled over ROLL buckets (readouts)
    raw_buckets: list[Bucket]  # unrolled (rejection sparks)

    def queue_at(self, t: float) -> float:
        i = min(len(self.queue) - 1, max(0, int(t // QUEUE_STEP_S)))
        return self.queue[i][1]

    def raw_bucket_index(self, t: float) -> int:
        return min(len(self.raw_buckets) - 1, max(0, int(t // BUCKET_S)))

    def bucket_at(self, t: float) -> Bucket:
        return self.buckets[self.raw_bucket_index(t)]


@dataclass(frozen=True)
class ReplayData:
    lanes: dict[str, Lane]
    phases: list[Phase]
    rate_rps: dict[str, float]
    scoreboard: dict[str, dict[str, float | None]]  # policy -> 3-seed means
    per_seed: dict[str, dict[str, list[float | None]]]  # policy -> metric -> per-seed values
    coda: dict[str, dict[str, float | None]]  # "c256" / "c192" -> hard-cap means
    queue_max: float


def _admission_json(root: Path, name: str) -> dict[str, Any]:
    path = root / "analysis" / "tables" / name / "admission.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _means(entry: dict[str, Any]) -> dict[str, float | None]:
    return {k: entry["mean"].get(k) for k in _METRICS}


def _tpot_burst_mean(table: dict[str, Any], cell: str) -> float | None:
    """3-seed mean of the burst-stage TPOT p95 from the stage rows (not among the means)."""
    vals = [
        r["tpot_p95_burst_s"]
        for r in table["rows"]
        if r["cell"] == cell and r["policy"] == "hard_cap" and r["tpot_p95_burst_s"] is not None
    ]
    return sum(vals) / len(vals) if vals else None


def load_replay(root: Path, seed: int = 1) -> ReplayData:
    base = root / "evidence" / "raw" / "w3" / "fp8" / "trace" / f"seed-{seed}"
    lanes: dict[str, Lane] = {}
    phases: list[Phase] = []
    for policy in POLICIES:
        d = base / f"trace-{policy}"
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if not phases:
            phases = phases_of(m)
        series = waiting_series(d, origin_mono=m.get("records_origin_monotonic_s"))
        queue = downsample(series, QUEUE_STEP_S, TRACE_END_S)
        raw = bucket_records(
            read_records_jsonl(d / "records.jsonl"), bucket_s=BUCKET_S, end_s=TRACE_END_S
        )
        lanes[policy] = Lane(policy, queue, rolling(raw, ROLL), raw)
    trace = json.loads((base / f"trace-seed-{seed}.json").read_text(encoding="utf-8"))
    rate_rps = {p["phase"]: float(p["rate_rps"]) for p in trace["phases"]}
    w3_table = _admission_json(root, "w3-fp8-admission")
    c192_table = _admission_json(root, "w3-fp8-c192-admission")
    w3 = w3_table["per_cell_policy"]["fp8"]
    c192 = c192_table["per_cell_policy"]["fp8-c192"]["hard_cap"]
    scoreboard = {p: _means(w3[p]) for p in POLICIES}
    per_seed = {
        p: {k: list(w3[p]["per_seed"][k]) for k in _METRICS if k in w3[p]["per_seed"]}
        for p in POLICIES
    }
    coda = {"c256": _means(w3["hard_cap"]), "c192": _means(c192)}
    coda["c256"]["tpot_p95_burst_s"] = _tpot_burst_mean(w3_table, "fp8")
    coda["c192"]["tpot_p95_burst_s"] = _tpot_burst_mean(c192_table, "fp8-c192")
    queue_max = max(v for lane in lanes.values() for _, v in lane.queue)
    return ReplayData(lanes, phases, rate_rps, scoreboard, per_seed, coda, queue_max)
