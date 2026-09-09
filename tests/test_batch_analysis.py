"""Batch analysis: r_sat plateau flag, capacity C, tenancy suspects, deterministic tables."""

from __future__ import annotations

import json
from pathlib import Path

from slo_lab.batch_analysis import (
    analyze,
    flag_suspects,
    load_manifests,
    rebuild_from_index,
    run,
    signature_from_power_csv,
)


def _manifest(
    concurrency: int,
    rps: float,
    attainment: float,
    *,
    w_per_util: float | None = 3.0,
    probe: float | None = 0.019,
    kind: str = "closed_loop",
    rate: float | None = None,
) -> dict:
    return {
        "cell": "fp8",
        "seed": 1,
        "kind": kind,
        "concurrency": concurrency if kind == "closed_loop" else None,
        "rate_rps": rate,
        "records": 100,
        "window_records": 80,
        "window_s": 120.0,
        "achieved_rps": rps,
        "output_tok_per_s": rps * 132,
        "ttft_p50_s": 0.05,
        "ttft_p95_s": 0.2,
        "tpot_p50_s": 0.016,
        "tpot_p95_s": 0.02,
        "summary": {
            "attainment_offered": attainment,
            "attainment_offered_ci95": [attainment - 0.02, min(1.0, attainment + 0.02)],
            "rejection_rate": 0.0,
            "goodput_rps": rps * attainment,
        },
        "power_window": {"mean_w": 300.0, "output_tok_per_wh": 40000.0, "mean_util_pct": 80.0}
        | ({"w_per_util_point": w_per_util} if w_per_util is not None else {}),
        "probe_tpot_median_s": probe,
        "discard_first_s": 60.0,
    }


def test_flag_suspects_uses_probe_drift_and_power_signature() -> None:
    rows = [
        {"probe_tpot_median_s": 0.019, "w_per_util_point": 3.0},
        {"probe_tpot_median_s": 0.027, "w_per_util_point": 3.0},  # +42% probe drift
        {"probe_tpot_median_s": 0.019, "w_per_util_point": 1.8},  # shared-card signature
        {"probe_tpot_median_s": None, "w_per_util_point": None},  # nothing to judge on
    ]
    flag_suspects(rows)
    assert [r["suspect"] for r in rows] == [False, True, True, False]
    assert rows[1]["suspect_reasons"][0].startswith("probe TPOT 27.0 ms > best 19.0 ms")
    assert rows[2]["suspect_reasons"] == ["W per util point 1.8 < 2.0"]
    # desktop VRAM oversubscription (ADR 0007): committed past the physical card
    paged = [
        {"windows_committed_mb": 25263.0, "physical_vram_mib": 24564.0},
        {"windows_committed_mb": 23183.0, "physical_vram_mib": 24564.0},
        {"windows_committed_mb": 25263.0, "physical_vram_mib": None},
    ]
    flag_suspects(paged)
    assert [r["suspect"] for r in paged] == [True, False, False]
    assert paged[0]["suspect_reasons"][0].startswith("Windows committed VRAM 25263 MB > physical")


def test_analyze_excludes_suspects_and_flags_lower_bound_r_sat() -> None:
    manifests = [
        _manifest(1, 0.4, 1.0),
        _manifest(2, 0.9, 1.0),
        _manifest(4, 1.1, 0.5, w_per_util=1.8),  # contaminated: would otherwise drop C to 2
        _manifest(8, 3.8, 1.0),
        _manifest(16, 7.6, 0.98),
        _manifest(32, 13.8, 0.9),
    ]
    closed, open_ = analyze(manifests)
    cell = closed["per_cell"]["fp8"]
    assert cell["suspect_concurrencies_excluded"] == [4]
    assert cell["r_sat_rps"] == 13.8 and cell["r_sat_is_lower_bound"] is True
    assert cell["capacity_C"] == 16  # 32 fails the 95% rule, 4 is excluded not counted
    assert cell["gain_vs_prev_by_concurrency"][8] > 3  # 0.9 -> 3.8 skips the excluded point
    assert [r["concurrency"] for r in closed["rows"]] == [1, 2, 4, 8, 16, 32]
    assert open_ == {"rows": [], "per_cell": {}}
    # a flat top makes it a plateau
    manifests.append(_manifest(64, 14.2, 0.9))
    cell = analyze(manifests)[0]["per_cell"]["fp8"]
    assert cell["plateau_reached"] is True and cell["r_sat_is_lower_bound"] is False
    assert cell["plateau_first_concurrency"] == 32


def test_open_loop_r_slo_skips_suspect_rates() -> None:
    manifests = [
        _manifest(0, 10.0, 1.0, kind="open_loop", rate=10.0),
        _manifest(0, 20.0, 0.97, kind="open_loop", rate=20.0),
        _manifest(0, 30.0, 0.6, kind="open_loop", rate=30.0, w_per_util=1.5),
        _manifest(0, 40.0, 0.5, kind="open_loop", rate=40.0),
    ]
    _, open_ = analyze(manifests)
    cell = open_["per_cell"]["fp8"]
    assert cell["suspect_rates_excluded"] == [30.0]
    assert cell["r_slo"] == 20.0


def test_load_manifests_falls_back_to_power_csv_and_run_is_deterministic(tmp_path: Path) -> None:
    batch = tmp_path / "evidence" / "raw" / "w2" / "fp8" / "closed-loop"
    for c, rps, util, w in ((8, 3.8, 95.0, 250.0), (16, 3.7, 97.0, 175.0)):
        d = batch / "seed-1" / f"cl-conc-{c}"
        d.mkdir(parents=True)
        m = _manifest(c, rps, 1.0, w_per_util=None, probe=None)
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        lines = ["timestamp_iso,t_s,power_w,util_gpu_pct,mem_used_mib,clocks_sm_mhz,temp_c,phase"]
        lines += [f"x,{t},{w},{util},24000,2500,60,measure" for t in range(0, 130)]
        (d / "power.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (batch / "seed-1" / "quiet_gpu.json").write_text(
        json.dumps({"memory_total_mib": 24564.0}), encoding="utf-8"
    )
    sig = signature_from_power_csv(batch / "seed-1" / "cl-conc-16" / "power.csv", 60.0)
    assert sig == {"mean_util_pct": 97.0, "w_per_util_point": 1.8}
    loaded = load_manifests([batch])
    assert [m["power_window"]["w_per_util_point"] for m in loaded] == [1.8, 2.63]
    assert loaded[0]["_path"] == "closed-loop/seed-1/cl-conc-16/manifest.json"
    assert loaded[0]["_physical_vram_mib"] == 24564.0

    index = tmp_path / "analysis" / "tables" / "index.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps({"_comment": "x", "w2": ["evidence/raw/w2/fp8/closed-loop"]}), encoding="utf-8"
    )
    assert rebuild_from_index(tmp_path, index) == ["w2"]
    first = (index.parent / "w2" / "tables.md").read_text(encoding="utf-8")
    summary = run([batch], index.parent / "w2")
    assert summary["closed_per_cell"]["fp8"]["suspect_concurrencies_excluded"] == [16]
    assert summary["closed_per_cell"]["fp8"]["r_sat_rps"] == 3.8
    assert (index.parent / "w2" / "tables.md").read_text(encoding="utf-8") == first
    assert "| fp8 | 1 | 16 |" in first and "True |" in first
