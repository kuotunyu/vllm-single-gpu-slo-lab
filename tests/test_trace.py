"""Burst trace generation (ADR 0012): seeded, on inference-perf's 10 ms grid, phase-correct."""

from datetime import UTC, datetime
from pathlib import Path

from slo_lab.harness.trace import burst_arrivals, load_profile, phase_bounds, write_azure_trace

REPO = Path(__file__).resolve().parents[1]
PROFILE = [(0.5, 300), (1.5, 300), (0.5, 900)]


def test_profile_and_phase_bounds_from_burst25():
    profile = load_profile(REPO / "config" / "traffic" / "burst25.yaml")
    assert profile == PROFILE
    assert phase_bounds(profile) == [
        ("pre", 0.0, 300.0),
        ("burst", 300.0, 600.0),
        ("recovery", 600.0, 1500.0),
    ]


def test_arrivals_are_seeded_sorted_on_grid_and_start_at_zero():
    a = burst_arrivals(43.67, PROFILE, seed=1)
    assert a == burst_arrivals(43.67, PROFILE, seed=1)
    assert a != burst_arrivals(43.67, PROFILE, seed=2)
    assert a[0] == 0.0
    assert a == sorted(a)
    assert all(abs(t * 100 - round(t * 100)) < 1e-6 for t in a)
    assert a[-1] < 1500.0


def test_phase_counts_match_rates_within_four_sigma():
    a = burst_arrivals(43.67, PROFILE, seed=3)
    for (name, lo, hi), (mult, dur) in zip(phase_bounds(PROFILE), PROFILE, strict=True):
        n = sum(1 for t in a if lo <= t < hi)
        expected = mult * 43.67 * dur
        assert abs(n - expected) < 4 * expected**0.5, (name, n, expected)


def _ipf_offset(stamp: str) -> float:
    """inference-perf 0.6.1 AzurePublicDatasetReader.parse_timestamp: two fractional digits."""
    head, frac = stamp.split(".", 1)
    clean = f"{head}.{frac[:2].ljust(2, '0')}"
    return datetime.strptime(clean, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=UTC).timestamp()


def test_written_trace_replays_to_the_same_offsets(tmp_path):
    a = burst_arrivals(11.4, PROFILE, seed=1)
    path = tmp_path / "trace.csv"
    digest = write_azure_trace(path, a)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "TIMESTAMP,ContextTokens,GeneratedTokens"
    assert len(lines) == len(a) + 1
    rows = [line.split(",") for line in lines[1:]]
    assert all(r[1] == "108" and r[2] == "132" for r in rows)
    t0 = _ipf_offset(rows[0][0])
    assert [round(_ipf_offset(r[0]) - t0, 2) for r in rows] == [round(t, 2) for t in a]
    assert len(digest) == 64
    assert write_azure_trace(tmp_path / "again.csv", a) == digest


def test_short_profiles_get_positional_names():
    assert [p[0] for p in phase_bounds([(0.5, 60), (1.5, 60), (0.5, 120)])] == [
        "pre",
        "burst",
        "recovery",
    ]
    assert [p[0] for p in phase_bounds([(1.0, 10), (2.0, 10)])] == ["phase-0", "phase-1"]
