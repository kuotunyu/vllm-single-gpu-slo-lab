import pytest
from pydantic import ValidationError

from slo_lab.slo import (
    Outcome,
    RequestRecord,
    Slo,
    SweepCell,
    SweepRun,
    filter_window,
    grid_markdown,
    meets_slo,
    r_slo,
    read_records_jsonl,
    sensitivity_grid,
    summarise,
    write_records_jsonl,
)


def rec(rid, offered_at, outcome="ok", ttft=None, e2e=None, out=None):
    return RequestRecord(
        request_id=rid,
        offered_at_s=offered_at,
        outcome=outcome,
        ttft_s=ttft,
        e2e_s=e2e,
        output_tokens=out,
    )


# a: TPOT = (1.5 - 0.5) / 10 = 0.100 -> misses TPOT
# b: TPOT = (0.7 - 0.2) / 10 = 0.050 -> meets both (boundary inclusive)
# c: 429 rejection -> miss, counts in offered
# d: TPOT 0.050 but TTFT 1.2 -> misses TTFT
RECORDS = [
    rec("a", 0.0, ttft=0.5, e2e=1.5, out=11),
    rec("b", 1.0, ttft=0.2, e2e=0.7, out=11),
    rec("c", 2.0, outcome="rejected_429"),
    rec("d", 3.0, ttft=1.2, e2e=1.7, out=11),
]


def test_tpot_and_ntpot():
    assert RECORDS[0].tpot_s == pytest.approx(0.1)
    assert RECORDS[0].ntpot_s == pytest.approx(1.5 / 11)
    assert RECORDS[2].tpot_s is None
    assert rec("one", 0, ttft=0.1, e2e=0.1, out=1).tpot_s is None


def test_meets_slo_is_joint_and_inclusive():
    assert [meets_slo(r) for r in RECORDS] == [False, True, False, False]
    loose = Slo(ttft_s=2.0, tpot_s=0.1)
    assert [meets_slo(r, loose) for r in RECORDS] == [True, True, False, True]


def test_summarise_uses_offered_denominator():
    s = summarise(RECORDS, window_s=10.0)
    assert (s.offered, s.admitted, s.met, s.rejected_429) == (4, 3, 1, 1)
    assert s.attainment_offered == 0.25
    assert s.attainment_admitted == pytest.approx(1 / 3)
    assert s.rejection_rate == 0.25
    assert s.goodput_rps == pytest.approx(0.1)
    lo, hi = s.attainment_offered_ci95
    assert 0.0 <= lo <= 0.25 <= hi <= 1.0


def test_summarise_counts_timeout_and_error_as_misses():
    records = [
        rec("t", 0.0, outcome="timeout"),
        rec("e", 1.0, outcome="error"),
        rec("ok", 2.0, ttft=0.1, e2e=0.6, out=11),
    ]
    s = summarise(records)
    assert (s.offered, s.admitted, s.met, s.timeouts, s.errors) == (3, 3, 1, 1, 1)
    assert s.rejection_rate == pytest.approx(1 / 3)
    assert s.attainment_offered == pytest.approx(1 / 3)
    assert s.goodput_rps is None


def test_summarise_all_rejected_has_no_admitted_attainment():
    s = summarise([rec("x", 0.0, outcome="rejected_429")])
    assert s.attainment_admitted is None
    assert s.attainment_offered == 0.0


def test_summarise_empty_raises():
    with pytest.raises(ValueError):
        summarise([])


def test_ok_record_requires_latency_fields():
    with pytest.raises(ValidationError):
        RequestRecord(request_id="x", offered_at_s=0.0, outcome=Outcome.OK)
    with pytest.raises(ValidationError):
        rec("bad", 0.0, ttft=1.0, e2e=0.5, out=11)


def test_filter_window_drops_warmup():
    assert [r.request_id for r in filter_window(RECORDS, 1.0)] == ["b", "c", "d"]
    assert [r.request_id for r in filter_window(RECORDS, 1.0, 3.0)] == ["b", "c"]


def cells(table):
    return [
        SweepCell(offered_rate=rate, seed=seed, attainment=att)
        for rate, per_seed in table.items()
        for seed, att in per_seed.items()
    ]


def test_r_slo_highest_rate_where_all_seeds_pass():
    res = r_slo(cells({1.0: {0: 0.99, 1: 0.98}, 2.0: {0: 0.96, 1: 0.95}, 3.0: {0: 0.97, 1: 0.90}}))
    assert res.r_slo == 2.0
    assert res.limiting_rate == 3.0
    assert res.seeds == [0, 1]
    assert res.min_attainment_by_rate[2.0] == 0.95


def test_r_slo_stops_at_first_failing_rate():
    res = r_slo(cells({1.0: {0: 0.99, 1: 0.99}, 2.0: {0: 0.90, 1: 0.99}, 3.0: {0: 0.99, 1: 0.99}}))
    assert res.r_slo == 1.0
    assert res.limiting_rate == 2.0


def test_r_slo_none_when_lowest_rate_fails():
    res = r_slo(cells({1.0: {0: 0.5, 1: 0.99}, 2.0: {0: 0.99, 1: 0.99}}))
    assert res.r_slo is None
    assert res.limiting_rate == 1.0


def test_r_slo_missing_seed_counts_as_failure():
    res = r_slo(cells({1.0: {0: 0.99, 1: 0.99}, 2.0: {0: 0.99}, 3.0: {0: 0.99, 1: 0.99}}))
    assert res.r_slo == 1.0
    assert res.min_attainment_by_rate[2.0] is None


def test_r_slo_empty_raises():
    with pytest.raises(ValueError):
        r_slo([])


def _run(rate, seed, ttft, tpot, n=20):
    e2e = ttft + tpot * 10  # 11 output tokens -> TPOT = (e2e - ttft) / 10
    return SweepRun(
        offered_rate=rate,
        seed=seed,
        records=[rec(f"{rate}-{seed}-{i}", float(i), ttft=ttft, e2e=e2e, out=11) for i in range(n)],
    )


def test_sensitivity_grid_recomputes_r_slo_per_threshold_pair():
    runs = []
    for seed in (0, 1):
        runs.append(_run(1.0, seed, ttft=0.3, tpot=0.02))
        runs.append(_run(2.0, seed, ttft=0.8, tpot=0.04))
        runs.append(_run(3.0, seed, ttft=1.5, tpot=0.04))
    grid = sensitivity_grid(runs)
    assert grid[(0.5, 0.03)] == 1.0
    assert grid[(0.5, 0.05)] == 1.0
    assert grid[(1.0, 0.03)] == 1.0
    assert grid[(1.0, 0.05)] == 2.0
    assert grid[(2.0, 0.05)] == 3.0
    assert grid[(2.0, 0.10)] == 3.0
    assert len(grid) == 9


def test_sensitivity_grid_window_can_empty_runs():
    runs = [_run(1.0, 0, ttft=0.3, tpot=0.02, n=5)]
    assert sensitivity_grid(runs, window_start_s=60.0) == dict.fromkeys(
        [(t, p) for t in (0.5, 1.0, 2.0) for p in (0.03, 0.05, 0.1)]
    )


def test_grid_markdown_renders_values_and_na():
    text = grid_markdown({(0.5, 0.03): None, (0.5, 0.05): 2.0})
    assert "| 0.5 s | n/a | 2 req/s |" in text
    assert "30 ms" in text and "50 ms" in text


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "records.jsonl"
    write_records_jsonl(path, RECORDS)
    back = read_records_jsonl(path)
    assert back == RECORDS
