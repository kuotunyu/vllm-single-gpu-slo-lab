"""Dependency-free SVG figures rebuilt from the committed tables and evidence (spec §8, W5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_line_chart_is_deterministic_svg_with_axes_series_and_legend() -> None:
    from slo_lab.plots import Series, line_chart

    series = [
        Series("fp8", [(10.0, 1.0), (20.0, 1.0), (30.0, 0.5)]),
        Series("bf16", [(5.0, 1.0), (10.0, 0.9)], dashed=True),
    ]
    svg = line_chart(
        series,
        title="attainment vs offered rate",
        x_label="offered rps",
        y_label="attainment",
        hlines=[(0.95, "95 % target")],
    )
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<polyline") == 2 and "stroke-dasharray" in svg
    assert "fp8" in svg and "bf16" in svg and "95 % target" in svg
    assert "offered rps" in svg and "attainment" in svg
    assert svg == line_chart(
        series,
        title="attainment vs offered rate",
        x_label="offered rps",
        y_label="attainment",
        hlines=[(0.95, "95 % target")],
    )
    # coordinates are rounded, so floating-point noise cannot change the bytes
    assert ".0000" not in svg


def test_line_chart_rejects_empty_input() -> None:
    from slo_lab.plots import line_chart

    with pytest.raises(ValueError):
        line_chart([], title="t", x_label="x", y_label="y")


def _open_loop_table(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    out = tmp_path / "analysis" / "tables" / name
    out.mkdir(parents=True)
    per_cell: dict[str, dict] = {}
    for r in rows:
        cell = per_cell.setdefault(r["cell"], {"min_attainment_by_rate": {}})
        key = str(r["offered_rps"])
        prev = cell["min_attainment_by_rate"].get(key)
        cell["min_attainment_by_rate"][key] = (
            r["attainment"] if prev is None else min(prev, r["attainment"])
        )
    (out / "open_loop.json").write_text(
        json.dumps({"rows": rows, "per_cell": per_cell}), encoding="utf-8"
    )
    return out


def test_attainment_and_tpot_charts_read_open_loop_tables(tmp_path: Path) -> None:
    from slo_lab.plots import attainment_vs_rate, tpot_vs_rate

    rows = []
    for seed in (1, 2):
        for rate, att, tpot in ((10.0, 1.0, 0.02), (20.0, 0.97, 0.03), (30.0, 0.4, 0.06)):
            rows.append(
                {
                    "cell": "fp8",
                    "seed": seed,
                    "offered_rps": rate,
                    "attainment": att - 0.01 * (seed - 1),
                    "tpot_p95_s": tpot,
                    "suspect": False,
                }
            )
    rows.append(
        {
            "cell": "fp8",
            "seed": 1,
            "offered_rps": 40.0,
            "attainment": 0.9,
            "tpot_p95_s": 0.01,
            "suspect": True,
        }
    )
    table = _open_loop_table(tmp_path, "w2-fp8-open-loop", rows)
    from slo_lab.plots import attainment_series, tpot_series

    (att_series,) = attainment_series([table])
    assert att_series.name == "fp8"
    assert att_series.points == [(10.0, 0.99), (20.0, 0.96), (30.0, 0.39)]  # min over seeds
    assert all(x != 40.0 for x, _ in att_series.points)  # suspect stage left out
    (tp_series,) = tpot_series([table])
    assert tp_series.points == [(10.0, 20.0), (20.0, 30.0), (30.0, 60.0)]  # ms, mean over seeds
    att = attainment_vs_rate([table], title="W2")
    assert att.count("<polyline") == 1  # one line per cell
    tpot = tpot_vs_rate([table], title="W4", slo_tpot_s=0.05)
    assert "50 ms" in tpot and tpot.count("<polyline") == 1


def test_queue_timeline_aligns_scraped_queues_on_the_records_origin(tmp_path: Path) -> None:
    from slo_lab.plots import queue_timeline

    stage = tmp_path / "trace-passthrough"
    stage.mkdir()
    rows = ["t_unix,num_requests_running,num_requests_waiting,t_mono"]
    rows += [f"{1e9 + t},10,{50 if 300 <= t < 700 else 0},{5000 + t}" for t in range(0, 1500, 5)]
    (stage / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (stage / "shim.csv").write_text(
        "t_unix,in_flight,waiting,admitted,completed,upstream_errors,rejected_cap_full,"
        "rejected_queue_full,rejected_queue_timeout,t_mono\n"
        f"{1e9 + 400},256,7,1,1,0,0,0,0,5400.0\n",
        encoding="utf-8",
    )
    (stage / "manifest.json").write_text(
        json.dumps(
            {
                "policy": "passthrough",
                "records_origin_monotonic_s": 5000.0,
                "clock_offset_s": {"before": 1e9 - 5000.0, "after": 1e9 - 5000.0},
                "trace": {
                    "phases": [
                        {"phase": "pre", "start_s": 0, "end_s": 300},
                        {"phase": "burst", "start_s": 300, "end_s": 600},
                        {"phase": "recovery", "start_s": 600, "end_s": 1500},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    svg = queue_timeline([stage], title="fp8 seed 1")
    assert svg.count("<polyline") == 1 and "passthrough" in svg
    assert "burst" in svg  # phase boundaries are drawn
    assert svg == queue_timeline([stage], title="fp8 seed 1")


def test_rebuild_plots_writes_only_the_figures_whose_inputs_exist(tmp_path: Path) -> None:
    from slo_lab.plots import rebuild_plots

    rows = [
        {
            "cell": "fp8",
            "seed": 1,
            "offered_rps": 10.0,
            "attainment": 1.0,
            "tpot_p95_s": 0.02,
            "suspect": False,
        }
    ]
    _open_loop_table(tmp_path, "w2-fp8-open-loop", rows)
    (tmp_path / "analysis" / "tables" / "index.json").write_text(
        json.dumps({"w2-fp8-open-loop": ["evidence/raw/w2/fp8/open-loop"]}), encoding="utf-8"
    )
    written = rebuild_plots(tmp_path)
    assert written == ["w2-attainment-vs-rate.svg"]
    assert (tmp_path / "evidence" / "plots" / "w2-attainment-vs-rate.svg").exists()
    assert rebuild_plots(tmp_path) == written  # idempotent, same bytes
    first = (tmp_path / "evidence" / "plots" / "w2-attainment-vs-rate.svg").read_bytes()
    rebuild_plots(tmp_path)
    assert (tmp_path / "evidence" / "plots" / "w2-attainment-vs-rate.svg").read_bytes() == first


def test_repo_plots_rebuild_from_committed_evidence() -> None:
    from slo_lab.plots import rebuild_plots

    if not (REPO / "analysis" / "tables" / "w2-fp8-open-loop" / "open_loop.json").exists():
        pytest.skip("W2 tables not present")
    names = rebuild_plots(REPO)
    assert "w2-attainment-vs-rate.svg" in names
    assert any(n.startswith("w3-queue-timeline") for n in names)
