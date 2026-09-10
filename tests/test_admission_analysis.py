"""Admission-trace analysis (W3): phase summaries, time-to-recover, policy tables, paired diffs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from slo_lab.admission_analysis import (
    analyze_trace,
    phase_summary,
    time_to_recover,
    time_to_recover_attainment,
    waiting_series,
    write_trace_tables,
)
from slo_lab.batch_analysis import load_manifests
from slo_lab.batch_analysis import run as run_tables
from slo_lab.slo import RequestRecord, write_records_jsonl

PHASES = [
    {"phase": "pre", "start_s": 0.0, "end_s": 300.0},
    {"phase": "burst", "start_s": 300.0, "end_s": 600.0},
    {"phase": "recovery", "start_s": 600.0, "end_s": 1500.0},
]


def _ok(i: int, t: float, ttft: float = 0.1, tpot: float = 0.02) -> RequestRecord:
    return RequestRecord(
        request_id=f"r{i}",
        offered_at_s=t,
        outcome="ok",
        ttft_s=ttft,
        e2e_s=ttft + tpot * 131,
        output_tokens=132,
    )


def _rej(i: int, t: float) -> RequestRecord:
    return RequestRecord(request_id=f"x{i}", offered_at_s=t, outcome="rejected_429")


def test_phase_summary_counts_rejections_as_misses_but_keeps_them_out_of_latency():
    recs = [_ok(i, 310.0 + i) for i in range(90)] + [_rej(i, 320.0 + i) for i in range(10)]
    s = phase_summary(recs, 300.0, 600.0)
    assert s["offered"] == 100 and s["met"] == 90 and s["rejected_429"] == 10
    assert s["attainment"] == pytest.approx(0.9)
    assert s["rejection_rate"] == pytest.approx(0.1)
    assert s["goodput_rps"] == pytest.approx(90 / 300)
    assert s["ttft_p95_s"] == pytest.approx(0.1)
    assert s["tpot_p95_s"] == pytest.approx(0.02)
    empty = phase_summary(recs, 0.0, 300.0)
    assert empty["offered"] == 0 and empty["attainment"] is None


def _backlog_records() -> list[RequestRecord]:
    slow = [_ok(i, 600.0 + i * 0.5, ttft=5.0) for i in range(200)]  # offered 600.0 .. 699.5
    fast = [_ok(1000 + i, 700.0 + i * 0.5) for i in range(1600)]  # offered 700.0 .. 1499.5
    return slow + fast


def test_time_to_recover_needs_an_empty_queue_and_a_good_ttft_window():
    recs = _backlog_records()
    drained_at_690 = [(float(t), 50.0 if t < 690 else 0.0) for t in range(0, 1500, 5)]
    # the queue is empty from 690 but [690, 750) still holds 20 slow of 120 (p95 = 5 s)
    assert time_to_recover(recs, drained_at_690, burst_end_s=600.0, trace_end_s=1500.0) == 100.0
    drained_at_800 = [(float(t), 50.0 if t < 800 else 0.0) for t in range(0, 1500, 5)]
    assert time_to_recover(recs, drained_at_800, burst_end_s=600.0, trace_end_s=1500.0) == 200.0
    never = [(float(t), 1.0) for t in range(0, 1500, 5)]
    assert time_to_recover(recs, never, burst_end_s=600.0, trace_end_s=1500.0) is None


def test_attainment_recovery_sees_rejections_that_ttft_cannot():
    rejected = [_rej(i, 600.0 + i * 0.5) for i in range(100)]  # 600.0 .. 649.5
    served = [_ok(1000 + i, 650.0 + i * 0.5) for i in range(1700)]  # 650.0 .. 1499.5
    recs = rejected + served
    empty_queue = [(float(t), 0.0) for t in range(0, 1500, 5)]
    # admitted requests are fast throughout, so the spec's TTFT criterion is met at once ...
    assert time_to_recover(recs, empty_queue, burst_end_s=600.0, trace_end_s=1500.0) == 0.0
    # ... while offered attainment only reaches 95 % once the rejections leave the window
    assert time_to_recover_attainment(recs, burst_end_s=600.0, trace_end_s=1500.0) == 50.0


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def test_waiting_series_adds_the_shim_queue_to_the_engine_queue(tmp_path: Path):
    origin = 1_000_000.0
    _write_csv(
        tmp_path / "metrics.csv",
        ["t_unix", "num_requests_running", "num_requests_waiting"],
        [[origin + 0.0, 10, 0], [origin + 5.0, 256, 40], [origin + 10.0, 100, 0]],
    )
    _write_csv(
        tmp_path / "shim.csv",
        ["t_unix", "in_flight", "waiting"],
        [[origin + 1.0, 10, 0], [origin + 6.0, 256, 7], [origin + 11.0, 90, 0]],
    )
    # files without t_mono fall back to the wall clock
    expected = [(0.0, 0.0), (5.0, 40.0), (10.0, 7.0)]
    assert waiting_series(tmp_path, origin_mono=123.0, origin_unix=origin) == expected
    (tmp_path / "shim.csv").unlink()
    assert waiting_series(tmp_path, origin_unix=origin) == [(0.0, 0.0), (5.0, 40.0), (10.0, 0.0)]
    assert waiting_series(tmp_path) == []


def test_waiting_series_aligns_on_the_monotonic_clock_when_present(tmp_path: Path):
    # the wall clock drifts 7 s over the stage (seen in WSL2); t_mono does not
    mono0, unix0 = 500.0, 1_000_000.0
    _write_csv(
        tmp_path / "metrics.csv",
        ["t_unix", "num_requests_running", "num_requests_waiting", "t_mono"],
        [
            [unix0 + 0.0, 1, 0, mono0 + 0.0],
            [unix0 + 1.5, 1, 30, mono0 + 5.0],
            [unix0 + 3.0, 1, 0, mono0 + 10.0],
        ],
    )
    _write_csv(
        tmp_path / "shim.csv",
        ["t_unix", "in_flight", "waiting", "t_mono"],
        [[unix0 + 2.0, 1, 4, mono0 + 6.0]],
    )
    got = waiting_series(tmp_path, origin_mono=mono0, origin_unix=unix0)
    assert got == [(0.0, 0.0), (5.0, 30.0), (10.0, 4.0)]


def _stage(root: Path, cell: str, policy: str, seed: int, recs: list[RequestRecord], probe=0.019):
    d = root / cell / "trace" / f"seed-{seed}" / f"trace-{policy}"
    d.mkdir(parents=True)
    write_records_jsonl(d / "records.jsonl", recs)
    origin_mono, offset = 5000.0, 1_700_000_000.0
    rows = [[offset + origin_mono + t, 0, 0] for t in range(0, 1500, 5)]
    _write_csv(d / "metrics.csv", ["t_unix", "num_requests_running", "num_requests_waiting"], rows)
    manifest = {
        "cell": cell,
        "kind": "trace",
        "policy": policy,
        "seed": seed,
        "records": len(recs),
        "trace": {"file": f"trace-seed-{seed}.csv", "sha256": "0" * 64, "phases": PHASES},
        "records_origin_monotonic_s": origin_mono,
        "clock_offset_s": {"before": offset, "after": offset},
        "probe_tpot_median_s": probe,
        "shim_cpu_s": 60.0,
        "load_wall_s": 1600.0,
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _trace_records(reject_every: int) -> list[RequestRecord]:
    recs: list[RequestRecord] = []
    for i in range(1500):
        t = float(i)
        in_burst = 300.0 <= t < 600.0
        if in_burst and reject_every and i % reject_every == 0:
            recs.append(_rej(i, t))
        else:
            recs.append(_ok(i, t, ttft=2.0 if (in_burst and not reject_every) else 0.1))
    return recs


def test_analyze_trace_builds_policy_rows_and_paired_differences(tmp_path: Path):
    for seed in (1, 2):
        _stage(tmp_path, "fp8", "passthrough", seed, _trace_records(reject_every=0))
        _stage(tmp_path, "fp8", "hard_cap", seed, _trace_records(reject_every=2))
    manifests = load_manifests([tmp_path / "fp8"])
    result = analyze_trace(manifests)
    rows = result["rows"]
    assert [(r["policy"], r["seed"]) for r in rows] == [
        ("passthrough", 1),
        ("passthrough", 2),
        ("hard_cap", 1),
        ("hard_cap", 2),
    ]
    native = rows[0]
    assert native["attainment_burst"] == pytest.approx(0.0)  # every burst request waited 2 s
    assert native["attainment"] == pytest.approx(1200 / 1500)
    cap = rows[2]
    assert cap["rejection_burst"] == pytest.approx(0.5)
    assert cap["attainment_burst"] == pytest.approx(0.5)
    per = result["per_cell_policy"]["fp8"]
    assert per["hard_cap"]["n"] == 2
    diff = per["hard_cap"]["paired_vs_passthrough"]["attainment_burst"]
    assert diff["per_seed"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert diff["consistent_sign"] is True
    assert "paired_vs_passthrough" not in per["passthrough"]

    out = tmp_path / "tables"
    write_trace_tables(out, result)
    md = (out / "tables.md").read_text(encoding="utf-8")
    assert "hard_cap" in md and "passthrough" in md
    assert (
        json.loads((out / "admission.json").read_text(encoding="utf-8"))["rows"][0]["cell"] == "fp8"
    )


def test_batch_run_writes_admission_tables_for_trace_manifests(tmp_path: Path):
    _stage(tmp_path, "bf16", "passthrough", 1, _trace_records(reject_every=0))
    out = tmp_path / "tables"
    summary = run_tables([tmp_path / "bf16"], out)
    assert summary["trace_rows"] == 1
    assert (out / "admission.json").exists() and (out / "tables.md").exists()
    assert not (out / "closed_loop.json").exists()
