"""Evidence compression (ADR 0011): deterministic, lossless, idempotent, and invisible to readers."""

import gzip
import importlib.util
import json
from pathlib import Path

from slo_lab.batch_analysis import served_rps
from slo_lab.slo import evidence_path, open_evidence_text, read_records_jsonl

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compress_evidence", REPO / "scripts" / "compress_evidence.py"
)
assert _spec is not None and _spec.loader is not None
compress_evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compress_evidence)


def _record(i: int, offered: float, e2e: float, outcome: str = "ok") -> dict:
    return {
        "request_id": f"r{i}",
        "offered_at_s": offered,
        "outcome": outcome,
        "ttft_s": 0.05,
        "e2e_s": e2e,
        "output_tokens": 132,
    }


def _write_stage(d: Path) -> Path:
    d.mkdir(parents=True)
    lines = [json.dumps(_record(i, 60.0 + i, 1.0)) for i in range(5)]
    (d / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "vllm.log").write_text("INFO engine started\n" * 50, encoding="utf-8")
    (d / "vllm-refine-0.80.log").write_text("INFO refine session\n", encoding="utf-8")
    (d / "manifest.json").write_text("{}\n", encoding="utf-8")
    (d / "inference-perf.log").write_text("ipf\n", encoding="utf-8")
    return d


def test_compress_replaces_targets_losslessly_and_leaves_other_files(tmp_path):
    stage = _write_stage(tmp_path / "seed-1" / "ol-rate-1.00")
    original = (stage / "records.jsonl").read_bytes()
    files, before, after = compress_evidence.compress_tree([tmp_path])
    assert files == 3 and after < before
    names = sorted(p.name for p in stage.iterdir())
    assert names == [
        "inference-perf.log",
        "manifest.json",
        "records.jsonl.gz",
        "vllm-refine-0.80.log.gz",
        "vllm.log.gz",
    ]
    assert gzip.decompress((stage / "records.jsonl.gz").read_bytes()) == original


def test_compression_is_deterministic_and_idempotent(tmp_path):
    a = _write_stage(tmp_path / "a" / "s")
    b = _write_stage(tmp_path / "b" / "s")
    compress_evidence.compress_tree([tmp_path / "a"])
    compress_evidence.compress_tree([tmp_path / "b"])
    assert (a / "records.jsonl.gz").read_bytes() == (b / "records.jsonl.gz").read_bytes()
    snapshot = (a / "records.jsonl.gz").read_bytes()
    assert compress_evidence.compress_tree([tmp_path / "a"]) == (0, 0, 0)
    assert (a / "records.jsonl.gz").read_bytes() == snapshot


def test_readers_see_the_same_records_plain_or_compressed(tmp_path):
    plain = _write_stage(tmp_path / "plain")
    packed = _write_stage(tmp_path / "packed")
    compress_evidence.compress_tree([packed])
    assert evidence_path(packed / "records.jsonl") == packed / "records.jsonl.gz"
    assert evidence_path(plain / "records.jsonl") == plain / "records.jsonl"
    assert evidence_path(tmp_path / "nowhere" / "records.jsonl") is None
    assert read_records_jsonl(plain / "records.jsonl") == read_records_jsonl(
        packed / "records.jsonl"
    )
    with open_evidence_text(packed / "vllm.log") as fh:
        assert fh.readline() == "INFO engine started\n"
    kwargs = {"discard_first_s": 60.0, "window_end_s": 300.0}
    assert served_rps(plain / "records.jsonl", **kwargs) == served_rps(
        packed / "records.jsonl", **kwargs
    )


def test_main_reports_missing_directory(tmp_path, capsys):
    assert compress_evidence.main([str(tmp_path / "absent")]) == 2
    assert "not found" in capsys.readouterr().err
