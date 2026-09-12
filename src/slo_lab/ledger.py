"""Run ledger (``analysis/ledger/runs.csv``): one row per measured stage, rebuilt from evidence.

The design spec's evidence contract (§9) has ``make reproduce`` recompute three ledgers. Two of
them stay header-only by decision: ``cost.csv`` because the owned 4090 carries no $ figure
(ADR 0010) and ``spend.csv`` because no paid compute was ever used (A1 cancelled, ADR 0014).
``runs.csv`` is rebuilt here from every ``manifest.json`` under the batch directories listed in
``analysis/tables/index.json``; the status column comes from the rebuilt tables' suspect flags, so
a stage the analyzer excluded is visible in the ledger with its reasons. Rows are sorted by
``run_id`` and written with a fixed column order, so the file is deterministic and diffs clean.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RUNS_COLUMNS = (
    "run_id",
    "date",
    "gpu_class",
    "precision",
    "admission",
    "specdec",
    "traffic",
    "seed",
    "status",
    "evidence_path",
)
GPU_CLASS = "RTX 4090 24 GB (WSL2, desktop-shared)"
PRECISION_BY_MODEL = {
    "Qwen/Qwen3-8B": "bf16",
    "Qwen/Qwen3-8B-FP8": "fp8",
    "Qwen/Qwen3-8B-AWQ": "awq",
    "JunHowie/Qwen3-8B-GPTQ-Int4": "gptq_int4",
    "Qwen/Qwen3-4B": "bf16-4b",
}
TABLE_FILES = ("closed_loop.json", "open_loop.json", "admission.json")


def precision_of(model: str | None) -> str:
    return PRECISION_BY_MODEL.get(model or "", model or "")


def admission_of(manifest: dict[str, Any]) -> str:
    """The shim policy the stage went through; ``direct`` when no shim stats were recorded."""
    policy = manifest.get("policy")
    if policy:
        return str(policy)
    return "passthrough" if manifest.get("shim_final") else "direct"


def traffic_of(manifest: dict[str, Any]) -> str:
    kind = str(manifest.get("kind") or "")
    if kind == "closed_loop":
        return f"closed_loop c={manifest.get('concurrency')}"
    if kind == "open_loop":
        return f"open_loop {manifest.get('rate_rps')} rps"
    if kind == "trace":
        trace = manifest.get("trace")
        profile = trace.get("profile") if isinstance(trace, dict) else None
        return f"trace {Path(str(profile)).stem}" if profile else "trace"
    return kind


def _status_by_path(table_dir: Path) -> dict[str, str]:
    """Map a table row's relative manifest path to ``clean`` or ``suspect: <reasons>``."""
    status: dict[str, str] = {}
    for name in TABLE_FILES:
        f = table_dir / name
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            if row.get("suspect"):
                reasons = "; ".join(str(r) for r in row.get("suspect_reasons") or [])
                status[row["path"]] = f"suspect: {reasons}" if reasons else "suspect"
            else:
                status[row["path"]] = "clean"
    return status


def ledger_rows(root: Path, index: dict[str, list[str]]) -> list[dict[str, str]]:
    """One row per manifest under the batch directories of every table in ``index``."""
    raw = root / "evidence" / "raw"
    rows: dict[str, dict[str, str]] = {}
    for table, dirs in index.items():
        if table.startswith("_"):
            continue
        status = _status_by_path(root / "analysis" / "tables" / table)
        for d in dirs:
            batch = root / d
            if not batch.exists():
                continue
            for mf in sorted(batch.rglob("manifest.json")):
                m = json.loads(mf.read_text(encoding="utf-8"))
                stage = mf.parent
                run_id = stage.relative_to(raw).as_posix()
                if run_id in rows:
                    continue
                key = mf.relative_to(batch.parent).as_posix()
                started = str(m.get("started_at") or "")
                rows[run_id] = {
                    "run_id": run_id,
                    "date": started[:10],
                    "gpu_class": GPU_CLASS,
                    "precision": precision_of(m.get("model")),
                    "admission": admission_of(m),
                    "specdec": str(m.get("specdec") or "none"),
                    "traffic": traffic_of(m),
                    "seed": str(m.get("seed") if m.get("seed") is not None else ""),
                    "status": status.get(key, "not in table"),
                    "evidence_path": stage.relative_to(root).as_posix(),
                }
    return [rows[k] for k in sorted(rows)]


def write_runs_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUNS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rebuild_runs_ledger(root: Path, index_path: Path) -> int:
    """Rewrite ``analysis/ledger/runs.csv`` from the index; returns the number of rows."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = ledger_rows(root, index)
    write_runs_ledger(root / "analysis" / "ledger" / "runs.csv", rows)
    return len(rows)
