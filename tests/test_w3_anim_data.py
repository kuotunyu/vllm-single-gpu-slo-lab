"""The explainer animation's loader reads exactly the committed evidence and tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("w3_data", REPO / "scripts" / "manim" / "w3_data.py")
assert _spec is not None and _spec.loader is not None
w3_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w3_data)


@pytest.fixture(scope="module")
def replay():
    return w3_data.load_replay(REPO)


def test_lanes_cover_the_trace_on_fixed_grids(replay) -> None:
    assert list(replay.lanes) == list(w3_data.POLICIES)
    for lane in replay.lanes.values():
        assert lane.queue[0][0] == 0.0
        assert lane.queue[-1][0] == 1500.0 and len(lane.queue) == 301
        assert len(lane.buckets) == 150 and len(lane.raw_buckets) == 150
        assert sum(b.offered for b in lane.raw_buckets) == 45624
    assert replay.queue_max == pytest.approx(9479.0)
    assert replay.lanes["hard_cap"].queue_at(450.0) == 0.0
    assert replay.lanes["passthrough"].queue_at(600.0) > 5000
    assert replay.lanes["passthrough"].raw_bucket_index(1499.9) == 149
    assert replay.lanes["hard_cap"].bucket_at(450.0).rejected > 0


def test_phases_rates_and_scoreboard_match_the_tables(replay) -> None:
    assert [p.name for p in replay.phases] == ["pre", "burst", "recovery"]
    assert replay.rate_rps == {"pre": 21.835, "burst": 65.505, "recovery": 21.835}
    sb = replay.scoreboard
    assert sb["passthrough"]["attainment"] == pytest.approx(0.1918, abs=5e-4)
    assert sb["hard_cap"]["attainment"] == pytest.approx(0.5729, abs=5e-4)
    assert sb["bounded_queue"]["goodput_rps"] == pytest.approx(18.01, abs=0.01)
    assert sb["hard_cap"]["rejection_rate"] == pytest.approx(0.2092, abs=5e-4)
    assert replay.per_seed["passthrough"]["time_to_recover_s"] == [820.0, None, 745.0]
    assert replay.coda["c256"]["tpot_p95_burst_s"] == pytest.approx(0.0578, abs=5e-4)
    assert replay.coda["c192"]["tpot_p95_burst_s"] == pytest.approx(0.0452, abs=5e-4)
    assert replay.coda["c192"]["attainment_burst"] == pytest.approx(0.491, abs=1e-3)
    assert replay.coda["c192"]["attainment"] == pytest.approx(0.7804, abs=5e-4)
