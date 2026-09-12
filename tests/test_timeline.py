"""Time-bucketed views of a replayed trace (slo_lab.timeline)."""

from __future__ import annotations

import pytest

from slo_lab.slo import RequestRecord
from slo_lab.timeline import Bucket, Phase, bucket_records, downsample, phases_of, rolling


def _ok(i: int, offered: float, ttft: float = 0.1) -> RequestRecord:
    return RequestRecord(
        request_id=f"r{i}",
        offered_at_s=offered,
        outcome="ok",
        ttft_s=ttft,
        e2e_s=ttft + 1.0,
        output_tokens=132,
    )


def _bad(i: int, offered: float, outcome: str) -> RequestRecord:
    return RequestRecord(request_id=f"r{i}", offered_at_s=offered, outcome=outcome)


RECORDS = [
    _ok(0, 1.0, 0.1),
    _ok(1, 2.0, 0.2),
    _ok(2, 5.0, 2.0),  # TTFT above the 1 s threshold: offered, not met
    _bad(3, 12.0, "rejected_429"),
    _bad(4, 15.0, "timeout"),
    _ok(5, 25.0, 0.3),
]


def test_bucket_records_counts_attainment_and_ttft() -> None:
    b = bucket_records(RECORDS, bucket_s=10.0, end_s=30.0)
    assert [x.start_s for x in b] == [0.0, 10.0, 20.0]
    assert (b[0].offered, b[0].met, b[0].rejected, b[0].timeouts) == (3, 2, 0, 0)
    assert b[0].attainment == pytest.approx(2 / 3)
    assert b[0].ttft_p95_s == pytest.approx(1.82)  # linear-interpolation p95 of [0.1, 0.2, 2.0]
    assert (b[1].offered, b[1].met, b[1].rejected, b[1].timeouts) == (2, 0, 1, 1)
    assert b[1].attainment == 0.0 and b[1].ttft_p95_s is None
    assert (b[2].offered, b[2].met) == (1, 1)


def test_bucket_records_default_end_and_explicit_cut() -> None:
    assert len(bucket_records(RECORDS, bucket_s=10.0)) == 3  # last offer at 25 s -> end 30 s
    cut = bucket_records(RECORDS, bucket_s=10.0, end_s=20.0)
    assert len(cut) == 2 and sum(x.offered for x in cut) == 5
    empty = bucket_records([], bucket_s=10.0, end_s=30.0)
    assert len(empty) == 3 and empty[0].attainment is None and empty[0].ttft_p95_s is None


def test_bucket_records_rejects_bad_width() -> None:
    with pytest.raises(ValueError):
        bucket_records(RECORDS, bucket_s=0.0)


def test_rolling_merges_a_window_of_buckets() -> None:
    b = bucket_records(RECORDS, bucket_s=10.0, end_s=30.0)
    r = rolling(b, 2)
    assert (r[0].start_s, r[0].end_s, r[0].offered) == (0.0, 10.0, 3)
    assert (r[1].start_s, r[1].end_s, r[1].offered, r[1].met, r[1].rejected) == (
        0.0,
        20.0,
        5,
        2,
        1,
    )
    assert r[1].ttft_p95_s == pytest.approx(1.82)
    assert (r[2].start_s, r[2].end_s, r[2].offered, r[2].met) == (10.0, 30.0, 3, 1)
    assert r[2].ok_ttfts == (0.3,)
    with pytest.raises(ValueError):
        rolling(b, 0)


def test_downsample_holds_the_latest_sample() -> None:
    series = [(7.0, 3.0), (2.0, 1.0), (12.0, 0.0)]
    assert downsample(series, 5.0, 15.0) == [(0.0, 0.0), (5.0, 1.0), (10.0, 3.0), (15.0, 0.0)]
    assert downsample([], 5.0, 10.0) == [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    with pytest.raises(ValueError):
        downsample(series, 0.0, 10.0)


def test_phases_of_reads_the_manifest_trace() -> None:
    m = {
        "trace": {
            "phases": [
                {"phase": "pre", "start_s": 0, "end_s": 300},
                {"phase": "burst", "start_s": 300, "end_s": 600},
            ]
        }
    }
    assert phases_of(m) == [Phase("pre", 0.0, 300.0), Phase("burst", 300.0, 600.0)]
    assert phases_of({}) == []
    assert isinstance(Bucket(0.0, 10.0, 0, 0, 0, 0, ()), Bucket)
