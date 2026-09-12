"""The W4 spec-decode animation's loader reads exactly the committed tables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("w4_data", REPO / "scripts" / "manim" / "w4_data.py")
assert _spec is not None and _spec.loader is not None
w4_data = importlib.util.module_from_spec(_spec)
sys.modules["w4_data"] = w4_data
_spec.loader.exec_module(w4_data)


@pytest.fixture(scope="module")
def spec():
    return w4_data.load_specdec(REPO)


def test_acceptance_closed_loop_and_r_sat(spec) -> None:
    by = {c.name: c for c in spec.cells}
    assert list(by) == ["fp8-ngram", "q4b-eagle3", "q4b-ngram"]
    assert by["fp8-ngram"].acceptance == pytest.approx(0.5197, abs=1e-3)
    assert by["q4b-eagle3"].acceptance == pytest.approx(0.2654, abs=1e-3)
    assert by["q4b-ngram"].mean_acceptance_length == pytest.approx(2.599, abs=1e-3)
    assert by["fp8-ngram"].closed[1] == pytest.approx(1.724, abs=1e-3)
    assert by["fp8-ngram"].closed[128] == pytest.approx(0.85, abs=1e-3)
    assert by["fp8-ngram"].closed[256] is None  # paged, excluded by the suspect rule
    assert by["q4b-eagle3"].closed[256] == pytest.approx(0.604, abs=1e-3)
    assert set(by["q4b-ngram"].closed) == set(w4_data.CONCURRENCIES)
    assert by["fp8-ngram"].r_sat == pytest.approx(26.67, abs=0.01)
    assert by["fp8-ngram"].base_r_sat == pytest.approx(40.17, abs=0.01)
    assert by["q4b-ngram"].base_r_sat == pytest.approx(47.62, abs=0.01)


def test_open_loop_points_stop_at_half_of_the_base_r_sat(spec) -> None:
    by = {c.name: c for c in spec.cells}
    fp8 = by["fp8-ngram"].open
    assert [p.offered_rps for p in fp8] == [4.02, 10.04, 20.09]
    assert fp8[2].tpot_p95_diff_ms == pytest.approx(3.81, abs=0.01)
    assert fp8[0].tpot_p50_diff_ms == pytest.approx(-3.56, abs=0.01)
    assert all(abs(p.attainment_diff) < 0.002 for p in fp8)
    eagle = by["q4b-eagle3"].open
    assert [p.offered_rps for p in eagle] == [4.76, 11.9, 23.81]
    assert eagle[0].tpot_p95_diff_ms == pytest.approx(-2.21, abs=0.01)
    assert eagle[2].tpot_p95_diff_ms == pytest.approx(10.2, abs=0.01)
    assert eagle[2].p95_consistent is True
