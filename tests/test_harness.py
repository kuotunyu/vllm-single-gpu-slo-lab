"""Harness pieces: inference-perf adapter, metrics scraper parsing, config generation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from slo_lab.harness.ipf_adapter import adapt_file, adapt_record, write_records_jsonl
from slo_lab.harness.ipf_config import closed_loop_config, open_loop_config, write_config
from slo_lab.harness.metrics_scraper import TRACKED, MetricsScraper, parse_metrics
from slo_lab.slo import Outcome, read_records_jsonl, summarise

SMOKE = (
    Path(__file__).resolve().parents[1]
    / "evidence/raw/w1/inference-perf-smoke/per_request_lifecycle_metrics.json"
)


def test_adapter_reads_the_w1_smoke_file_end_to_end(tmp_path: Path) -> None:
    records = adapt_file(SMOKE)
    assert len(records) == 20
    assert all(r.outcome is Outcome.OK for r in records)
    assert min(r.offered_at_s for r in records) == 0.0
    assert all(r.ttft_s is not None and 0 < r.ttft_s < 2 for r in records)
    assert all(r.e2e_s is not None and r.e2e_s >= r.ttft_s for r in records)  # type: ignore[operator]
    assert all(r.output_tokens and r.output_tokens > 0 for r in records)
    assert all(r.input_tokens and r.input_tokens > 0 for r in records)
    out = tmp_path / "records.jsonl"
    write_records_jsonl(records, out)
    back = read_records_jsonl(out)
    assert [r.request_id for r in back] == [r.request_id for r in records]
    summary = summarise(back, window_s=20.0)
    assert summary.offered == 20 and summary.rejected_429 == 0
    assert 0.0 <= summary.attainment_offered <= 1.0


def _raw(start: float, chunks: list[float], error: str | None = None, tokens: int = 5) -> dict:
    return {
        "start_time": start,
        "end_time": (chunks[-1] if chunks else start) + 0.001,
        "request": "{}",
        "response": "",
        "info": {
            "request_metrics": {"text": {"input_tokens": 108}},
            "response_metrics": {"output_tokens": tokens, "chunk_times": chunks},
            "input_tokens": 108,
        },
        "error": error,
    }


def test_adapter_maps_errors_to_outcomes() -> None:
    assert (
        adapt_record(_raw(10.0, [], "HTTP 429 Too Many Requests"), index=0, origin_s=10.0).outcome
        is Outcome.REJECTED_429
    )
    assert (
        adapt_record(_raw(10.0, [], "request timed out"), index=1, origin_s=10.0).outcome
        is Outcome.TIMEOUT
    )
    assert adapt_record(_raw(10.0, [], "boom"), index=2, origin_s=10.0).outcome is Outcome.ERROR
    assert adapt_record(_raw(10.0, [], None), index=3, origin_s=10.0).outcome is Outcome.ERROR
    ok = adapt_record(_raw(12.5, [12.6, 12.7, 12.8]), index=4, origin_s=10.0)
    assert ok.outcome is Outcome.OK and ok.offered_at_s == 2.5
    assert abs(ok.ttft_s - 0.1) < 1e-9 and ok.e2e_s >= ok.ttft_s  # type: ignore[operator]


METRICS_TEXT = """# HELP vllm:num_requests_running x
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 7.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 5.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 12.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0
vllm:time_to_first_token_seconds_bucket{le="0.1"} 5.0
"""


def test_metrics_parser_collapses_labels_and_ignores_untracked() -> None:
    values = parse_metrics(METRICS_TEXT)
    assert values["vllm:num_requests_running"] == 3.0
    assert values["vllm:num_requests_waiting"] == 7.0
    assert values["vllm:kv_cache_usage_perc"] == 0.42
    assert "vllm:num_requests_waiting_by_reason" not in values
    assert "vllm:time_to_first_token_seconds_bucket" not in values


def test_metrics_scraper_writes_rows_with_injected_fetch(tmp_path: Path) -> None:
    ticks = iter([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
    scraper = MetricsScraper(
        "http://x/metrics",
        tmp_path / "m.csv",
        interval_s=0.01,
        fetch=lambda url: METRICS_TEXT,
        clock=lambda: next(ticks, 999.0),
    )
    scraper.start()
    import time

    time.sleep(0.08)
    scraper.stop()
    lines = (tmp_path / "m.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[:3] == ["t_unix", "num_requests_running", "num_requests_waiting"]
    assert scraper.rows >= 2 and scraper.errors == 0
    assert lines[1].split(",")[1] == "3.0"


def test_config_generation_pins_prompt_geometry_and_seed(tmp_path: Path) -> None:
    cfg = open_loop_config(
        model="m",
        base_url="http://127.0.0.1:8013",
        report_dir="r",
        rate_rps=2.5,
        duration_s=300,
        seed=11,
    )
    assert cfg["api"] == {"type": "completion", "streaming": True}
    assert (
        cfg["data"]["input_distribution"]["mean"] == 108
        and cfg["data"]["output_distribution"]["max"] == 132
    )
    assert cfg["load"]["type"] == "poisson" and cfg["load"]["stages"] == [
        {"rate": 2.5, "duration": 300}
    ]
    assert cfg["load"]["base_seed"] == 11 and cfg["load"]["request_timeout"] == 300.0
    assert cfg["server"]["ignore_eos"] is True and "_seed" not in cfg
    closed = closed_loop_config(
        model="m", base_url="u", report_dir="r", concurrency=16, num_requests=800, seed=11
    )
    assert closed["load"]["type"] == "concurrent"
    assert closed["load"]["stages"] == [{"num_requests": 800, "concurrency_level": 16}]
    path = write_config(cfg, tmp_path / "cfg.yaml")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["load"]["stages"][0]["rate"] == 2.5
    assert TRACKED[0] == "vllm:num_requests_running"
    assert json.loads(json.dumps(cfg))  # plain data only
