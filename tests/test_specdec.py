"""W4 speculative-decoding pieces (ADR 0015): counters, stage summary, config flag, paired analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

REPO = Path(__file__).resolve().parents[1]

METRICS_TEXT = """# HELP vllm:spec_decode_num_drafts_total x
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:num_preemptions_total{engine="0",model_name="m"} 4.0
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 300.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 150.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 80.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 50.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="2"} 20.0
vllm:spec_decode_num_drafts_created{engine="0",model_name="m"} 1.7e9
"""

NGRAM_FLAG = (
    "--speculative-config "
    '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'
)


def test_tracked_metrics_include_spec_decode_and_preemption_counters() -> None:
    from slo_lab.harness.metrics_scraper import TRACKED, parse_metrics

    for name in (
        "vllm:num_preemptions_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    ):
        assert name in TRACKED
    assert TRACKED[0] == "vllm:num_requests_running"  # existing CSV columns keep their place
    values = parse_metrics(METRICS_TEXT)
    assert values["vllm:spec_decode_num_draft_tokens_total"] == 300.0
    assert values["vllm:num_preemptions_total"] == 4.0
    assert "vllm:spec_decode_num_drafts_created" not in values


def test_parse_labelled_keys_a_counter_by_one_label() -> None:
    from slo_lab.harness.metrics_scraper import SPEC_DECODE_PER_POS, parse_labelled

    assert parse_labelled(METRICS_TEXT, SPEC_DECODE_PER_POS, "position") == {
        "0": 80.0,
        "1": 50.0,
        "2": 20.0,
    }
    assert parse_labelled(METRICS_TEXT, "vllm:nothing", "position") == {}


def test_spec_decode_summary_reports_rates_from_counter_deltas() -> None:
    from slo_lab.harness.stage import counter_delta, spec_decode_summary

    before = {
        "vllm:spec_decode_num_drafts_total": 10.0,
        "vllm:spec_decode_num_draft_tokens_total": 30.0,
        "vllm:spec_decode_num_accepted_tokens_total": 5.0,
        "vllm:num_preemptions_total": 1.0,
    }
    after = {
        "vllm:spec_decode_num_drafts_total": 110.0,
        "vllm:spec_decode_num_draft_tokens_total": 330.0,
        "vllm:spec_decode_num_accepted_tokens_total": 155.0,
        "vllm:num_preemptions_total": 1.0,
    }
    out = spec_decode_summary(
        before, after, {"0": 5.0, "1": 0.0, "2": 0.0}, {"0": 85.0, "1": 50.0, "2": 20.0}
    )
    assert out == {
        "drafts": 100.0,
        "draft_tokens": 300.0,
        "accepted_tokens": 150.0,
        "acceptance_rate": 0.5,
        "mean_acceptance_length": 2.5,
        "accepted_per_pos": [80.0, 50.0, 20.0],
        "acceptance_per_pos": [0.8, 0.5, 0.2],
    }
    assert counter_delta(before, after, "vllm:num_preemptions_total") == 0.0
    assert counter_delta({}, after, "vllm:num_preemptions_total") is None
    # a stage with the counters present but no drafts (idle server) has undefined rates
    idle = spec_decode_summary(after, after, {"0": 85.0}, {"0": 85.0})
    assert idle["drafts"] == 0.0
    assert idle["acceptance_rate"] is None and idle["mean_acceptance_length"] is None
    assert idle["acceptance_per_pos"] == [None]
    # a server without speculative decoding exposes no counters at all
    plain = {"vllm:num_requests_running": 0.0}
    assert spec_decode_summary(plain, plain) is None
    assert spec_decode_summary(None, after) is None


def test_speculative_config_flag_renders_the_yaml_as_one_compact_argument() -> None:
    from slo_lab.harness.specdec import speculative_config_flag

    assert speculative_config_flag(REPO / "config/specdec/ngram.yaml") == NGRAM_FLAG
    eagle = speculative_config_flag(REPO / "config/specdec/eagle3_4b.yaml")
    flag, _, payload = eagle.partition(" ")
    assert flag == "--speculative-config" and " " not in payload  # one shell word for batch.sh
    assert json.loads(payload) == {
        "method": "eagle3",
        "model": "AngelSlim/Qwen3-4B_eagle3",
        "num_speculative_tokens": 3,
    }
    assert speculative_config_flag(REPO / "config/specdec/none.yaml") == ""


def test_specdec_flags_command_prints_the_server_argument() -> None:
    from slo_lab.cli import app

    result = CliRunner().invoke(app, ["specdec-flags", str(REPO / "config/specdec/ngram.yaml")])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == NGRAM_FLAG
    none = CliRunner().invoke(app, ["specdec-flags", str(REPO / "config/specdec/none.yaml")])
    assert none.exit_code == 0 and none.output.strip() == ""


def test_specdec_configs_are_frozen_for_w4() -> None:
    ngram = yaml.safe_load((REPO / "config/specdec/ngram.yaml").read_text(encoding="utf-8"))
    assert ngram["speculative_config"] == {
        "method": "ngram",
        "num_speculative_tokens": 3,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    }
    eagle = yaml.safe_load((REPO / "config/specdec/eagle3_4b.yaml").read_text(encoding="utf-8"))
    assert eagle["speculative_config"]["num_speculative_tokens"] == 3
    assert eagle["speculative_config"]["model"] == "AngelSlim/Qwen3-4B_eagle3"


def _manifest(
    cell: str,
    family: str,
    specdec: str,
    *,
    kind: str,
    rps: float,
    tpot_p50: float,
    tpot_p95: float,
    seed: int = 1,
    concurrency: int | None = None,
    rate: float | None = None,
    ttft_p95: float = 0.2,
    attainment: float = 1.0,
    tok_per_wh: float = 40000.0,
    acceptance: float | None = None,
    w_per_util: float = 3.0,
) -> dict:
    return {
        "cell": cell,
        "family": family,
        "specdec": specdec,
        "seed": seed,
        "kind": kind,
        "concurrency": concurrency,
        "rate_rps": rate,
        "records": 100,
        "window_records": 80,
        "window_s": 120.0,
        "achieved_rps": rps,
        "output_tok_per_s": rps * 132,
        "ttft_p50_s": 0.05,
        "ttft_p95_s": ttft_p95,
        "tpot_p50_s": tpot_p50,
        "tpot_p95_s": tpot_p95,
        "summary": {
            "attainment_offered": attainment,
            "attainment_offered_ci95": [attainment - 0.02, min(1.0, attainment + 0.02)],
            "rejection_rate": 0.0,
            "goodput_rps": rps * attainment,
        },
        "power_window": {
            "mean_w": 300.0,
            "output_tok_per_wh": tok_per_wh,
            "mean_util_pct": 80.0,
            "w_per_util_point": w_per_util,
        },
        # the re-warm single-stream probe is a per-host constant, not the stage's TPOT
        "probe_tpot_median_s": 0.019,
        "discard_first_s": 60.0,
        "preemptions": 0.0,
        "spec_decode": None
        if acceptance is None
        else {
            "drafts": 100.0,
            "draft_tokens": 300.0,
            "accepted_tokens": 300.0 * acceptance,
            "acceptance_rate": acceptance,
            "mean_acceptance_length": 1 + 3 * acceptance,
            "accepted_per_pos": [],
            "acceptance_per_pos": [],
        },
    }


def test_rows_carry_specdec_fields_only_when_the_manifest_has_them() -> None:
    from slo_lab.batch_analysis import _row

    old = _row({"cell": "fp8", "seed": 1, "summary": {}, "power_window": {}})
    assert "specdec" not in old and "acceptance_rate" not in old and "preemptions" not in old
    new = _row(
        _manifest(
            "fp8-ngram",
            "fp8",
            "ngram",
            kind="closed_loop",
            concurrency=1,
            rps=0.4,
            tpot_p50=0.016,
            tpot_p95=0.017,
            acceptance=0.5,
        )
    )
    assert new["specdec"] == "ngram" and new["family"] == "fp8"
    assert new["acceptance_rate"] == 0.5 and new["mean_acceptance_length"] == 2.5
    assert new["preemptions"] == 0.0


def _w4_manifests() -> list[dict]:
    manifests = []
    for c, rps, tpot in ((1, 0.4, 0.019), (256, 40.0, 0.040)):
        manifests.append(
            _manifest(
                "fp8-none",
                "fp8",
                "none",
                kind="closed_loop",
                concurrency=c,
                rps=rps,
                tpot_p50=tpot,
                tpot_p95=tpot * 1.1,
            )
        )
        manifests.append(
            _manifest(
                "fp8-ngram",
                "fp8",
                "ngram",
                kind="closed_loop",
                concurrency=c,
                rps=rps * 1.2,
                tpot_p50=tpot * 0.8,
                tpot_p95=tpot * 0.9,
                acceptance=0.4,
            )
        )
    for seed in (1, 2, 3):
        for rate in (4.0, 20.0):
            manifests.append(
                _manifest(
                    "fp8-none",
                    "fp8",
                    "none",
                    kind="open_loop",
                    seed=seed,
                    rate=rate,
                    rps=rate,
                    tpot_p50=0.022,
                    tpot_p95=0.025,
                )
            )
            manifests.append(
                _manifest(
                    "fp8-ngram",
                    "fp8",
                    "ngram",
                    kind="open_loop",
                    seed=seed,
                    rate=rate,
                    rps=rate,
                    tpot_p50=0.018,
                    tpot_p95=0.021,
                    attainment=1.0 if rate < 10 else 0.9,
                    acceptance=0.4,
                )
            )
    # an accelerated cell whose baseline is not in this batch
    manifests.append(
        _manifest(
            "q4b-eagle3",
            "q4b",
            "eagle3",
            kind="closed_loop",
            concurrency=1,
            rps=0.8,
            tpot_p50=0.009,
            tpot_p95=0.010,
            acceptance=0.7,
        )
    )
    return manifests


def test_specdec_analysis_pairs_each_accelerated_cell_with_its_family_baseline() -> None:
    from slo_lab.batch_analysis import analyze
    from slo_lab.specdec_analysis import analyze_specdec

    closed, open_ = analyze(_w4_manifests())
    result = analyze_specdec(closed, open_)
    fp8 = result["families"]["fp8"]
    assert fp8["base_cell"] == "fp8-none"
    ngram = fp8["cells"]["fp8-ngram"]
    assert ngram["specdec"] == "ngram"
    assert ngram["r_sat_rps"] == pytest.approx(48.0)
    assert ngram["r_sat_vs_base"] == pytest.approx(1.2)
    assert ngram["acceptance_rate_mean"] == pytest.approx(0.4)
    assert ngram["r_slo"] == 4.0  # 20 rps fails the 95 % rule in every seed
    closed_pairs = {(p["concurrency"], p["seed"]): p for p in ngram["closed_paired"]}
    assert closed_pairs[(1, 1)]["tpot_p50_diff_s"] == pytest.approx(0.019 * 0.8 - 0.019)
    assert closed_pairs[(256, 1)]["rps_ratio"] == pytest.approx(1.2)
    assert closed_pairs[(256, 1)]["acceptance_rate"] == 0.4
    assert len(ngram["open_paired"]) == 6
    by_rate = ngram["open_paired_by_rate"]
    assert by_rate["4"]["tpot_p95_diff_s"]["mean"] == pytest.approx(0.021 - 0.025)
    assert by_rate["4"]["tpot_p95_diff_s"]["consistent_sign"] is True
    assert by_rate["20"]["attainment_diff"]["mean"] == pytest.approx(-0.1)
    assert by_rate["4"]["attainment_diff"]["consistent_sign"] is False  # every difference is 0
    assert by_rate["4"]["tok_per_wh_ratio"]["mean"] == pytest.approx(1.0)
    q4b = result["families"]["q4b"]
    assert q4b["base_cell"] is None
    assert q4b["cells"]["q4b-eagle3"]["closed_paired"] == []
    assert q4b["cells"]["q4b-eagle3"]["r_sat_vs_base"] is None


def test_specdec_analysis_skips_pairs_with_a_suspect_stage() -> None:
    from slo_lab.batch_analysis import analyze
    from slo_lab.specdec_analysis import analyze_specdec

    manifests = _w4_manifests()
    # the baseline's c = 256 stage was measured on a shared card
    for m in manifests:
        if m["cell"] == "fp8-none" and m.get("concurrency") == 256:
            m["power_window"]["w_per_util_point"] = 1.5
    closed, open_ = analyze(manifests)
    ngram = analyze_specdec(closed, open_)["families"]["fp8"]["cells"]["fp8-ngram"]
    assert [p["concurrency"] for p in ngram["closed_paired"]] == [1]


def test_run_writes_specdec_tables_beside_the_closed_and_open_tables(tmp_path: Path) -> None:
    from slo_lab.batch_analysis import run

    dirs = []
    for cell, specdec, tpot in (("fp8-none", "none", 0.02), ("fp8-ngram", "ngram", 0.016)):
        d = tmp_path / "evidence" / "raw" / "w4" / cell / "closed-loop" / "seed-1" / "cl-conc-1"
        d.mkdir(parents=True)
        m = _manifest(
            cell,
            "fp8",
            specdec,
            kind="closed_loop",
            concurrency=1,
            rps=0.4,
            tpot_p50=tpot,
            tpot_p95=tpot,
            acceptance=None if specdec == "none" else 0.5,
        )
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        dirs.append(tmp_path / "evidence" / "raw" / "w4" / cell)
    out = tmp_path / "tables" / "w4-fp8-specdec"
    summary = run(dirs, out)
    assert summary["specdec_families"] == ["fp8"]
    spec = json.loads((out / "specdec.json").read_text(encoding="utf-8"))
    pair = spec["families"]["fp8"]["cells"]["fp8-ngram"]["closed_paired"][0]
    assert pair["tpot_p50_diff_s"] == pytest.approx(-0.004)
    md = (out / "tables.md").read_text(encoding="utf-8")
    assert "# Speculative decoding" in md and "fp8-ngram" in md
    run(dirs, out)
    assert (out / "tables.md").read_text(encoding="utf-8") == md  # deterministic rebuild
    # W2-style manifests (no specdec label) get no speculative-decoding table
    plain_dir = (
        tmp_path / "evidence" / "raw" / "w2" / "fp8" / "closed-loop" / "seed-1" / "cl-conc-1"
    )
    plain_dir.mkdir(parents=True)
    plain = _manifest(
        "fp8",
        "fp8",
        "none",
        kind="closed_loop",
        concurrency=1,
        rps=0.4,
        tpot_p50=0.02,
        tpot_p95=0.02,
    )
    for key in ("family", "specdec", "spec_decode", "preemptions"):
        plain.pop(key)
    (plain_dir / "manifest.json").write_text(json.dumps(plain), encoding="utf-8")
    plain_out = tmp_path / "tables" / "w2"
    plain_summary = run([tmp_path / "evidence" / "raw" / "w2" / "fp8" / "closed-loop"], plain_out)
    assert "specdec_families" not in plain_summary
    assert not (plain_out / "specdec.json").exists()
    assert "Speculative decoding" not in (plain_out / "tables.md").read_text(encoding="utf-8")


def test_physical_vram_is_read_from_the_shared_session_seed_dir(tmp_path: Path) -> None:
    """W4 shares one server session across seeds: quiet_gpu.json lives in seed-1 only, and the
    committed-VRAM rule must still apply to the stages under seed-2 and seed-3 (2026-09-12: the
    n-gram cell's overload stages of seeds 2 and 3 went unflagged because physical was None)."""
    from slo_lab.batch_analysis import _physical_vram_mib

    cell = tmp_path / "fp8-ngram"
    (cell / "seed-1" / "ol-rate-4.02").mkdir(parents=True)
    (cell / "seed-2" / "ol-rate-4.02").mkdir(parents=True)
    (cell / "seed-1" / "quiet_gpu.json").write_text(
        json.dumps({"memory_total_mib": 24564.0}), encoding="utf-8"
    )
    assert _physical_vram_mib(cell / "seed-1" / "ol-rate-4.02") == 24564.0
    assert _physical_vram_mib(cell / "seed-2" / "ol-rate-4.02") == 24564.0
    assert _physical_vram_mib(tmp_path / "elsewhere" / "seed-2" / "stage") is None
