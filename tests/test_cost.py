from pathlib import Path

import pytest

from slo_lab.cost import (
    POWER_CSV_COLUMNS,
    CostConfig,
    OperatingPoint,
    PowerSample,
    blended_usd_per_million_output_tokens,
    cost_report,
    gpu_usd_per_hour,
    idle_baseline_w,
    integrate,
    load_cost_config,
    power_usd_per_hour,
    read_power_csv,
    select_window,
    usd_per_million_output_tokens,
    wh_per_million_output_tokens,
)

REPO = Path(__file__).resolve().parents[1]

# 175200 TWD / 32 = 5475 USD; 5 y * 365 d * 24 h = 43800 h -> 0.125 USD/h
CFG = CostConfig(
    status="owner_provided",
    gpu_purchase_price_twd=175200,
    purchase_date="2024-01-01",
    amortisation_years=5,
    duty_hours_per_day=24,
    electricity_twd_per_kwh=3.2,  # 0.1 USD/kWh
    twd_per_usd=32,
)


def test_gpu_usd_per_hour_and_duty_sensitivity():
    assert gpu_usd_per_hour(CFG) == pytest.approx(0.125)
    assert gpu_usd_per_hour(CFG, duty_hours_per_day=8) == pytest.approx(0.375)


def test_power_usd_per_hour():
    assert power_usd_per_hour(300.0, CFG) == pytest.approx(0.03)


def test_usd_per_million_output_tokens_hand_computed():
    # (0.125 + 0.03) / 3600 / 100 * 1e6 = 0.430556
    assert usd_per_million_output_tokens(0.125, 0.03, 100.0) == pytest.approx(0.430556, abs=1e-6)
    with pytest.raises(ValueError):
        usd_per_million_output_tokens(0.125, 0.03, 0.0)


def test_wh_per_million_output_tokens():
    # 300 W at 100 tok/s = 3 J/token = 3e6 J per M = 833.33 Wh
    assert wh_per_million_output_tokens(300.0, 100.0) == pytest.approx(833.333, abs=1e-3)


def test_blended_google_formula():
    # 0.155/3600 USD/s / (100/4 + 100) tok/s * 1e6 = 0.344444
    assert blended_usd_per_million_output_tokens(0.125, 0.03, 100.0, 100.0, 4.0) == pytest.approx(
        0.344444, abs=1e-6
    )


def test_cost_report_measured_and_naive_with_caveat():
    at = OperatingPoint(output_tok_per_s=100.0, p_avg_w=300.0, input_tok_per_s=100.0)
    peak = OperatingPoint(output_tok_per_s=400.0, p_avg_w=350.0)
    cfg = CFG.model_copy(update={"input_output_price_ratio": 4.0})
    rep = cost_report(cfg, at, peak)
    assert rep.gpu_usd_per_h == pytest.approx(0.125)
    assert rep.usd_per_m_output_tok_at_r_slo == pytest.approx(0.430556, abs=1e-6)
    assert rep.usd_per_m_output_tok_at_r_slo_duty8 == pytest.approx(1.125, abs=1e-6)
    assert rep.usd_per_m_output_tok_naive == pytest.approx(
        0.111111, abs=1e-6
    )  # (0.125+0.035)/3600/400*1e6
    assert rep.utilisation_u == pytest.approx(0.25)
    assert rep.wh_per_m_output_tok_at_r_slo == pytest.approx(833.333, abs=1e-3)
    assert rep.wh_per_m_output_tok_naive == pytest.approx(243.056, abs=1e-3)
    assert rep.blended_usd_per_m_output_tok_at_r_slo == pytest.approx(0.344444, abs=1e-6)
    assert any("naive / U" in c for c in rep.caveats)
    assert any("lower bound" in c for c in rep.caveats)
    assert not any("placeholder" in c for c in rep.caveats)


def test_cost_report_without_peak_and_with_placeholder_config():
    cfg = CFG.model_copy(update={"status": "owner_input_pending"})
    rep = cost_report(cfg, OperatingPoint(output_tok_per_s=50.0, p_avg_w=200.0))
    assert rep.usd_per_m_output_tok_naive is None
    assert rep.utilisation_u is None
    assert rep.blended_usd_per_m_output_tok_at_r_slo is None
    assert rep.caveats[0].startswith("cost config still holds owner-input placeholders")


def test_shipped_cost_example_is_marked_placeholder():
    example = load_cost_config(REPO / "config" / "cost.yaml.example")
    assert example.is_placeholder


def samples(power, phase="measure", t0=0.0):
    return [
        PowerSample(t_s=t0 + i, power_w=p, temp_c=60 + i, clocks_sm_mhz=2500 - i, phase=phase)
        for i, p in enumerate(power)
    ]


def test_integrate_constant_power_trapezoid():
    s = integrate(samples([100.0] * 5))
    assert s.duration_s == 4.0
    assert s.energy_wh == pytest.approx(400.0 / 3600.0)
    assert s.p_avg_w == pytest.approx(100.0)
    assert s.p_max_w == 100.0
    assert s.temp_max_c == 64
    assert s.clocks_sm_min_mhz == 2496


def test_integrate_linear_ramp_and_idle_baseline():
    measure = samples([0.0, 100.0, 200.0, 300.0, 400.0], t0=60.0)
    idle = samples([20.0, 20.0, 20.0], phase="idle")
    s = integrate(measure)
    assert s.energy_wh == pytest.approx(800.0 / 3600.0)  # 50 + 150 + 250 + 350 J
    assert s.p_avg_w == pytest.approx(200.0)
    baseline = idle_baseline_w(idle)
    assert baseline == 20.0
    assert s.p_avg_w - baseline == pytest.approx(180.0)
    both = idle + measure
    assert select_window(both, phase="measure") == measure
    assert select_window(both, start_s=61.0, end_s=62.0) == measure[1:3]


def test_integrate_requires_two_samples_and_monotone_time():
    with pytest.raises(ValueError):
        integrate(samples([1.0]))
    with pytest.raises(ValueError):
        integrate([PowerSample(0.0, 1.0), PowerSample(0.0, 1.0)])
    with pytest.raises(ValueError):
        idle_baseline_w([])


def test_read_power_csv_uses_sampler_columns(tmp_path):
    path = tmp_path / "power.csv"
    lines = [",".join(POWER_CSV_COLUMNS)]
    lines.append("2026-09-03T00:00:00.000+00:00,1.0,110,50,1024,2500,61,measure")
    lines.append("2026-09-03T00:00:01.000+00:00,0.0,100,,,,,measure")  # out of order + blanks
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = read_power_csv(path)
    assert [r.t_s for r in rows] == [0.0, 1.0]
    assert rows[0].util_gpu_pct is None
    assert rows[1].mem_used_mib == 1024.0
    assert integrate(rows).energy_wh == pytest.approx(105.0 / 3600.0)
