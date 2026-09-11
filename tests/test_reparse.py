"""Re-parse W2's raw loadgen output with the server token count and compare with the committed records (W5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slo_lab.slo import Outcome, RequestRecord, read_records_jsonl, write_records_jsonl


def _raw(start: float, n_tokens: int, tpot: float, *, retok: int, server: int | None) -> dict:
    chunks = [start + 0.05 + tpot * i for i in range(n_tokens)]
    resp = {"output_tokens": retok, "chunk_times": chunks}
    if server is not None:
        resp["server_usage"] = {
            "prompt_tokens": 108,
            "completion_tokens": server,
            "total_tokens": 108 + server,
        }
    return {
        "start_time": start,
        "end_time": chunks[-1] + 0.001,
        "request": "{}",
        "response": "",
        "info": {
            "request_metrics": {"text": {"input_tokens": 108}},
            "response_metrics": resp,
            "input_tokens": 108,
        },
        "error": None,
    }


def test_reparse_stage_takes_the_server_token_count(tmp_path: Path) -> None:
    from slo_lab.reparse import reparse_stage

    raws = [
        _raw(10.0, 132, 0.02, retok=128, server=132),  # re-tokenized short by 4
        _raw(20.0, 132, 0.02, retok=132, server=132),
    ]
    raw_path = tmp_path / "per_request_lifecycle_metrics.json"
    raw_path.write_text(json.dumps(raws), encoding="utf-8")
    n = reparse_stage(raw_path, tmp_path / "records.jsonl")
    assert n == 2
    records = read_records_jsonl(tmp_path / "records.jsonl")
    assert [r.output_tokens for r in records] == [132, 132]


def _rec(i: int, offered: float, ttft: float, tpot: float, tokens: int) -> RequestRecord:
    return RequestRecord(
        request_id=f"ipf-{i:06d}",
        offered_at_s=offered,
        outcome=Outcome.OK,
        ttft_s=ttft,
        e2e_s=ttft + tpot * 131,  # what the server actually took for 132 tokens
        output_tokens=tokens,
        input_tokens=108,
    )


def test_compare_stage_reports_the_tpot_and_attainment_change(tmp_path: Path) -> None:
    from slo_lab.reparse import compare_stage

    # old: a quarter of the requests were counted as 100 tokens, which inflates their TPOT past 50 ms
    old = [_rec(i, 60.0 + i, 0.1, 0.045, 100 if i % 4 == 0 else 132) for i in range(40)]
    new = [_rec(i, 60.0 + i, 0.1, 0.045, 132) for i in range(40)]
    write_records_jsonl(tmp_path / "old.jsonl", old)
    write_records_jsonl(tmp_path / "new.jsonl", new)
    out = compare_stage(tmp_path / "old.jsonl", tmp_path / "new.jsonl", discard_first_s=60.0)
    assert out["records"] == 40 and out["changed_token_counts"] == 10
    assert out["attainment_old"] == pytest.approx(0.75) and out["attainment_new"] == 1.0
    assert out["tpot_p95_old_s"] > 0.05 > out["tpot_p95_new_s"]
    assert out["tpot_p95_new_s"] == pytest.approx(0.045)


def _stage(root: Path, cell: str, seed: int, rate: float, records: list[RequestRecord]) -> None:
    d = root / cell / "open-loop" / f"seed-{seed}" / f"ol-rate-{rate:.2f}"
    d.mkdir(parents=True)
    write_records_jsonl(d / "records.jsonl", records)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "cell": cell,
                "seed": seed,
                "kind": "open_loop",
                "rate_rps": rate,
                "duration_s": 300,
                "discard_first_s": 60.0,
            }
        ),
        encoding="utf-8",
    )


def test_compare_tree_recomputes_r_slo_per_cell(tmp_path: Path) -> None:
    from slo_lab.reparse import compare_tree

    old_root, new_root = tmp_path / "evidence", tmp_path / "new"
    for seed in (1, 2):
        for rate, tpot in ((10.0, 0.02), (20.0, 0.045)):
            old = [_rec(i, 60.0 + i, 0.1, tpot, 132) for i in range(40)]
            # the re-parse reveals that at 20 rps a fifth of seed 2's requests really took longer
            new = [
                _rec(
                    i,
                    60.0 + i,
                    0.1,
                    0.06 if (seed == 2 and rate == 20.0 and i % 5 == 0) else tpot,
                    132,
                )
                for i in range(40)
            ]
            _stage(old_root, "fp8", seed, rate, old)
            _stage(new_root, "fp8", seed, rate, new)
    out = compare_tree(old_root, new_root)
    assert len(out["stages"]) == 4
    cell = out["per_cell"]["fp8"]
    assert cell["r_slo_old"] == 20.0 and cell["r_slo_new"] == 10.0
    assert cell["changed"] is True
    stage = next(s for s in out["stages"] if s["seed"] == 2 and s["rate_rps"] == 20.0)
    assert stage["attainment_new"] == pytest.approx(0.8)
    # a stage missing on the new side is reported, not silently skipped
    _stage(old_root, "awq", 1, 5.0, [_rec(i, 60.0 + i, 0.1, 0.02, 132) for i in range(10)])
    out = compare_tree(old_root, new_root)
    assert out["missing_new"] == ["awq/open-loop/seed-1/ol-rate-5.00"]
