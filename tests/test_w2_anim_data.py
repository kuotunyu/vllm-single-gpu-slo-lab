"""The W2 knee animation's loader reads exactly the committed tables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("w2_data", REPO / "scripts" / "manim" / "w2_data.py")
assert _spec is not None and _spec.loader is not None
w2_data = importlib.util.module_from_spec(_spec)
sys.modules["w2_data"] = w2_data
_spec.loader.exec_module(w2_data)


@pytest.fixture(scope="module")
def knee():
    return w2_data.load_knee(REPO)


def test_curves_and_r_slo_match_the_tables(knee) -> None:
    assert list(knee.cells) == ["bf16", "fp8", "awq", "gptq"]
    fp8 = knee.cells["fp8"]
    assert dict(fp8.attainment)[28.39] == pytest.approx(0.9102, abs=1e-4)
    assert dict(fp8.attainment)[26.2] == 1.0 and dict(fp8.attainment)[43.67] == 0.0
    assert [c.r_slo for c in knee.cells.values()] == [10.26, 26.2, 22.61, 22.7]
    assert knee.cells["fp8"].r_sat == pytest.approx(41.35, abs=0.01)
    assert knee.cells["bf16"].r_sat == pytest.approx(11.40, abs=0.01)
    assert dict(fp8.tpot_p95_ms)[26.2] < 50 < dict(fp8.tpot_p95_ms)[32.75]
    # TTFT p95 (3-seed mean) stays well under the 1 s threshold up to the knee (0.40 s at 30.57)
    assert max(t for r, t in fp8.ttft_p95_s if r <= 30.57) < 0.5
    assert knee.rate_max >= 87.34


def test_energy_quality_and_sensitivity(knee) -> None:
    wh = {c.name: c.wh_per_m_tokens for c in knee.cells.values()}
    assert wh["bf16"] == pytest.approx(74.6, abs=0.1)
    assert wh["fp8"] == pytest.approx(31.2, abs=0.1)
    assert wh["awq"] == pytest.approx(36.2, abs=0.1)
    assert wh["gptq"] == pytest.approx(36.3, abs=0.1)
    assert knee.cells["bf16"].tmmlu_accuracy == pytest.approx(0.5911, abs=1e-4)
    assert knee.cells["bf16"].tmmlu_delta is None and knee.cells["bf16"].mcnemar_p is None
    assert knee.cells["fp8"].tmmlu_delta == pytest.approx(-0.0002, abs=1e-4)
    assert knee.cells["fp8"].mcnemar_p == pytest.approx(0.945, abs=0.005)
    assert knee.cells["gptq"].tmmlu_delta == pytest.approx(-0.0208, abs=1e-4)
    s = knee.sensitivity_fp8
    assert s[(1.0, 0.03)] == 24.02 and s[(1.0, 0.05)] == 26.2 and s[(2.0, 0.1)] == 30.57
    assert len(s) == 9
