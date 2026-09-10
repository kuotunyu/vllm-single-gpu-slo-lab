"""W3 harness pieces: trace-replay config, adapter origin, shim scraper, trace-stage summaries."""

from __future__ import annotations

import json
from pathlib import Path

from slo_lab.harness.ipf_adapter import adapt_file, adapt_file_with_origin
from slo_lab.harness.ipf_config import trace_replay_config

TRACE = {"file": "/t/trace-seed-2.csv", "format": "AzurePublicDataset"}


def _raw(start: float, chunks: list[float], error: str | None = None) -> dict:
    return {
        "start_time": start,
        "end_time": (chunks[-1] if chunks else start) + 0.001,
        "request": "{}",
        "response": "",
        "info": {
            "request_metrics": {"text": {"input_tokens": 108}},
            "response_metrics": {"output_tokens": len(chunks), "chunk_times": chunks},
            "input_tokens": 108,
        },
        "error": error,
    }


def test_trace_replay_config_is_one_continuous_stage_with_random_prompts() -> None:
    cfg = trace_replay_config(
        model="m",
        base_url="http://127.0.0.1:8021",
        report_dir="r",
        trace_file="/t/trace-seed-2.csv",
        duration_s=1500,
        mean_rate_rps=32.75,
        seed=2,
    )
    assert cfg["api"] == {"type": "completion", "streaming": True}
    assert cfg["server"]["base_url"] == "http://127.0.0.1:8021"
    assert cfg["server"]["ignore_eos"] is True
    # token counts come from the trace rows (108 / 132); the random generator ignores
    # distributions when a trace is given, so none are passed
    assert cfg["data"] == {"type": "random", "trace": TRACE}
    load = cfg["load"]
    assert load["type"] == "trace_replay" and load["trace"] == TRACE
    assert load["stages"] == [{"rate": 32.75, "duration": 1500}]
    assert load["interval"] == 0
    assert load["base_seed"] == 2 and load["request_timeout"] == 300.0
    # neither the client's in-flight cap nor its TCP pool may become the queue
    assert load["worker_max_concurrency"] == 4096
    assert load["worker_max_tcp_connections"] == 4096
    assert "_seed" not in cfg and "_timeout" not in cfg
    assert json.loads(json.dumps(cfg)) == cfg


def test_adapter_returns_the_raw_origin(tmp_path: Path) -> None:
    raws = [
        _raw(105.0, [105.1, 105.2]),
        _raw(100.0, [100.2, 100.3]),
        _raw(103.5, [], error="HTTP 429 Too Many Requests"),
    ]
    path = tmp_path / "per_request_lifecycle_metrics.json"
    path.write_text(json.dumps(raws), encoding="utf-8")
    records, origin = adapt_file_with_origin(path)
    assert origin == 100.0
    assert [r.offered_at_s for r in records] == [0.0, 3.5, 5.0]
    assert [r.outcome.value for r in records] == ["ok", "rejected_429", "ok"]
    assert records == adapt_file(path)
