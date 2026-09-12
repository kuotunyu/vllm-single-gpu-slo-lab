"""Run ledger rebuilt from manifests and the tables' suspect flags (slo_lab.ledger)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from slo_lab.ledger import (
    RUNS_COLUMNS,
    admission_of,
    ledger_rows,
    precision_of,
    rebuild_runs_ledger,
    traffic_of,
)


def _manifest(path: Path, **fields: object) -> None:
    path.mkdir(parents=True, exist_ok=True)
    base = {
        "cell": "fp8",
        "kind": "open_loop",
        "seed": 1,
        "model": "Qwen/Qwen3-8B-FP8",
        "rate_rps": 10.5,
        "concurrency": None,
        "started_at": "2026-09-10T05:44:07+00:00",
    }
    base.update(fields)
    (path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")


def _root(tmp_path: Path) -> Path:
    root = tmp_path
    raw = root / "evidence" / "raw"
    _manifest(raw / "w2" / "fp8" / "open-loop" / "seed-1" / "ol-rate-10.5")
    _manifest(raw / "w2" / "fp8" / "open-loop" / "seed-2" / "ol-rate-10.5", seed=2)
    _manifest(
        raw / "w4" / "fp8-ngram" / "closed-loop" / "seed-1" / "cl-conc-256",
        cell="fp8-ngram",
        kind="closed_loop",
        concurrency=256,
        rate_rps=None,
        specdec="ngram",
        family="fp8",
        shim_final={"admitted": 1},
        started_at="2026-09-12T00:10:00+00:00",
    )
    _manifest(
        raw / "w3" / "fp8" / "trace" / "seed-1" / "trace-hard_cap",
        kind="trace",
        policy="hard_cap",
        rate_rps=None,
        trace={"profile": "/mnt/d/repo/config/traffic/burst25.yaml"},
        started_at="2026-09-11T20:00:00+00:00",
    )
    tables = root / "analysis" / "tables"
    (tables / "w2-fp8-open-loop").mkdir(parents=True)
    (tables / "w2-fp8-open-loop" / "open_loop.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "path": "open-loop/seed-1/ol-rate-10.5/manifest.json",
                        "suspect": False,
                        "suspect_reasons": [],
                    },
                    {
                        "path": "open-loop/seed-2/ol-rate-10.5/manifest.json",
                        "suspect": True,
                        "suspect_reasons": ["windows committed 25000 MB > physical 24564 MiB"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (tables / "w4-fp8-specdec").mkdir(parents=True)
    (tables / "w4-fp8-specdec" / "closed_loop.json").write_text(
        json.dumps({"rows": []}), encoding="utf-8"
    )
    (tables / "w3-fp8-admission").mkdir(parents=True)
    (tables / "w3-fp8-admission" / "admission.json").write_text(
        json.dumps(
            {"rows": [{"path": "trace/seed-1/trace-hard_cap/manifest.json", "suspect": False}]}
        ),
        encoding="utf-8",
    )
    (tables / "index.json").write_text(
        json.dumps(
            {
                "_comment": "ignored",
                "w2-fp8-open-loop": ["evidence/raw/w2/fp8/open-loop"],
                "w4-fp8-specdec": ["evidence/raw/w4/fp8-ngram", "evidence/raw/w4/missing"],
                "w3-fp8-admission": ["evidence/raw/w3/fp8/trace"],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_rows_cover_every_manifest_with_status_from_tables(tmp_path: Path) -> None:
    root = _root(tmp_path)
    index = json.loads((root / "analysis" / "tables" / "index.json").read_text())
    rows = ledger_rows(root, index)
    by_id = {r["run_id"]: r for r in rows}
    assert [r["run_id"] for r in rows] == sorted(by_id)
    assert set(by_id) == {
        "w2/fp8/open-loop/seed-1/ol-rate-10.5",
        "w2/fp8/open-loop/seed-2/ol-rate-10.5",
        "w3/fp8/trace/seed-1/trace-hard_cap",
        "w4/fp8-ngram/closed-loop/seed-1/cl-conc-256",
    }
    clean = by_id["w2/fp8/open-loop/seed-1/ol-rate-10.5"]
    assert clean["status"] == "clean"
    assert clean["date"] == "2026-09-10"
    assert clean["precision"] == "fp8"
    assert clean["admission"] == "direct"
    assert clean["specdec"] == "none"
    assert clean["traffic"] == "open_loop 10.5 rps"
    assert clean["seed"] == "1"
    assert clean["evidence_path"] == "evidence/raw/w2/fp8/open-loop/seed-1/ol-rate-10.5"
    suspect = by_id["w2/fp8/open-loop/seed-2/ol-rate-10.5"]
    assert suspect["status"] == "suspect: windows committed 25000 MB > physical 24564 MiB"
    w4 = by_id["w4/fp8-ngram/closed-loop/seed-1/cl-conc-256"]
    assert w4["admission"] == "passthrough"
    assert w4["specdec"] == "ngram"
    assert w4["traffic"] == "closed_loop c=256"
    assert w4["status"] == "not in table"
    w3 = by_id["w3/fp8/trace/seed-1/trace-hard_cap"]
    assert w3["admission"] == "hard_cap"
    assert w3["traffic"] == "trace burst25"
    assert w3["status"] == "clean"


def test_rebuild_writes_deterministic_csv_with_the_frozen_header(tmp_path: Path) -> None:
    root = _root(tmp_path)
    index_path = root / "analysis" / "tables" / "index.json"
    n = rebuild_runs_ledger(root, index_path)
    out = root / "analysis" / "ledger" / "runs.csv"
    first = out.read_bytes()
    assert n == 4
    assert first.startswith(",".join(RUNS_COLUMNS).encode() + b"\n")
    assert b"\r\n" not in first
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4 and list(rows[0]) == list(RUNS_COLUMNS)
    rebuild_runs_ledger(root, index_path)
    assert out.read_bytes() == first


def test_field_helpers() -> None:
    assert precision_of("Qwen/Qwen3-4B") == "bf16-4b"
    assert precision_of("other/model") == "other/model"
    assert precision_of(None) == ""
    assert admission_of({"policy": "bounded_queue"}) == "bounded_queue"
    assert admission_of({"shim_final": {}}) == "direct"
    assert traffic_of({"kind": "trace"}) == "trace"
    assert traffic_of({"kind": "weird"}) == "weird"
