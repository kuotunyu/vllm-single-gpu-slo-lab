"""Re-parse raw inference-perf output with the current adapter and compare with committed records (W5).

W2's ``records.jsonl`` were adapted while the adapter still took inference-perf's re-tokenized
``output_tokens``; 1.45 % of W2's requests were counted short, which inflates their TPOT (ADR 0012
appendix). The raw ``per_request_lifecycle_metrics.json`` files stayed on the measurement host, so
every W2 stage can be re-adapted with the server's ``completion_tokens`` and compared, stage by
stage and at the r_SLO level, before any committed number changes (HANDOFF, W5 item 1).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from slo_lab.harness.ipf_adapter import adapt_file, write_records_jsonl
from slo_lab.slo import (
    DEFAULT_SLO,
    Outcome,
    RequestRecord,
    Slo,
    SweepCell,
    evidence_path,
    filter_window,
    r_slo,
    read_records_jsonl,
    summarise,
)
from slo_lab.stats import percentile


def reparse_stage(raw_path: Path, out_path: Path) -> int:
    """Adapt one raw per-request file with the current adapter; returns the record count."""
    records = adapt_file(raw_path)
    write_records_jsonl(records, out_path)
    return len(records)


def _tpot_p95(records: list[RequestRecord]) -> float | None:
    tpots = [r.tpot_s for r in records if r.outcome is Outcome.OK and r.tpot_s is not None]
    return float(percentile(tpots, 95)) if tpots else None


def _attainment(records: list[RequestRecord], slo: Slo) -> float | None:
    return summarise(records, slo).attainment_offered if records else None


def compare_stage(
    old_path: Path, new_path: Path, *, discard_first_s: float = 60.0, slo: Slo = DEFAULT_SLO
) -> dict[str, Any]:
    """Token-count changes, attainment and TPOT p95 of one stage, old records against new."""
    old = read_records_jsonl(old_path)
    new = read_records_jsonl(new_path)
    old_by_id = {r.request_id: r for r in old}
    changed = sum(
        1
        for r in new
        if r.request_id in old_by_id and old_by_id[r.request_id].output_tokens != r.output_tokens
    )
    old_w = filter_window(old, discard_first_s)
    new_w = filter_window(new, discard_first_s)
    return {
        "records": len(new),
        "records_old": len(old),
        "changed_token_counts": changed,
        "attainment_old": _attainment(old_w, slo),
        "attainment_new": _attainment(new_w, slo),
        "tpot_p95_old_s": _tpot_p95(old_w),
        "tpot_p95_new_s": _tpot_p95(new_w),
    }


def compare_tree(
    evidence_root: Path,
    new_root: Path,
    *,
    discard_first_s: float = 60.0,
    slo: Slo = DEFAULT_SLO,
) -> dict[str, Any]:
    """Every committed stage under ``evidence_root`` against ``new_root/<same path>/records.jsonl``.

    Open-loop stages of a cell are also reduced to r_SLO (frozen rule) on both sides, so a
    changed capacity number is visible before any table is regenerated.
    """
    stages: list[dict[str, Any]] = []
    missing_new: list[str] = []
    for manifest in sorted(evidence_root.rglob("manifest.json")):
        stage_dir = manifest.parent
        rel = stage_dir.relative_to(evidence_root).as_posix()
        if evidence_path(stage_dir / "records.jsonl") is None:
            continue
        if evidence_path(new_root / rel / "records.jsonl") is None:
            missing_new.append(rel)
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "stage": rel,
            "cell": m.get("cell"),
            "seed": m.get("seed"),
            "kind": m.get("kind"),
            "rate_rps": m.get("rate_rps"),
            "concurrency": m.get("concurrency"),
        }
        row.update(
            compare_stage(
                stage_dir / "records.jsonl",
                new_root / rel / "records.jsonl",
                discard_first_s=float(m.get("discard_first_s") or discard_first_s),
                slo=slo,
            )
        )
        stages.append(row)
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in stages:
        if s["kind"] == "open_loop" and s["rate_rps"] is not None and s["seed"] is not None:
            by_cell[str(s["cell"])].append(s)
    per_cell: dict[str, Any] = {}
    for cell, rows in sorted(by_cell.items()):
        old_cells = [
            SweepCell(offered_rate=r["rate_rps"], seed=r["seed"], attainment=r["attainment_old"])
            for r in rows
            if r["attainment_old"] is not None
        ]
        new_cells = [
            SweepCell(offered_rate=r["rate_rps"], seed=r["seed"], attainment=r["attainment_new"])
            for r in rows
            if r["attainment_new"] is not None
        ]
        r_old = r_slo(old_cells).r_slo if old_cells else None
        r_new = r_slo(new_cells).r_slo if new_cells else None
        per_cell[cell] = {
            "r_slo_old": r_old,
            "r_slo_new": r_new,
            "changed": r_old != r_new,
            "open_loop_stages": len(rows),
            "changed_token_counts": sum(r["changed_token_counts"] for r in rows),
            "records": sum(r["records"] for r in rows),
        }
    return {"stages": stages, "per_cell": per_cell, "missing_new": missing_new}
