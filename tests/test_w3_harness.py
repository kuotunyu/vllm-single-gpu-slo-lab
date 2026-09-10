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


def test_shim_scraper_writes_counter_rows_and_keeps_the_last_payload(tmp_path: Path) -> None:
    import time as _time

    from slo_lab.harness.shim_scraper import COLUMNS, ShimScraper

    payloads = iter(
        [
            {"in_flight": 3, "waiting": 0, "admitted": 3, "completed": 0, "rejected": {}},
            {
                "in_flight": 256,
                "waiting": 12,
                "admitted": 900,
                "completed": 640,
                "upstream_errors": 0,
                "rejected": {"queue_full": 5, "queue_timeout": 2},
            },
        ]
    )
    last: dict = {}

    def fetch(_url: str) -> dict:
        nonlocal last
        last = next(payloads, last)
        return last

    scraper = ShimScraper("u", tmp_path / "shim.csv", interval_s=0.05, fetch=fetch).start()
    _time.sleep(0.3)
    scraper.stop()
    lines = (tmp_path / "shim.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == ["t_unix", *COLUMNS]
    assert lines[2].split(",")[1:] == ["256", "12", "900", "640", "0", "0", "5", "2"]
    assert scraper.rows >= 2 and scraper.last == last


def test_trace_window_is_the_whole_trace() -> None:
    from slo_lab.harness.stage import stage_window
    from slo_lab.slo import RequestRecord

    recs = [
        RequestRecord(
            request_id=str(i),
            offered_at_s=float(i),
            outcome="ok",
            ttft_s=0.1,
            e2e_s=1.0,
            output_tokens=132,
        )
        for i in range(0, 1500, 10)
    ]
    window, window_s = stage_window(recs, kind="trace", discard_first_s=60.0, duration_s=1500)
    assert len(window) == len(recs) and window_s == 1500.0


def test_trace_stage_summary_aligns_scraped_queues_with_records(tmp_path: Path) -> None:
    from slo_lab.harness.stage import proc_cpu_s, trace_stage_summary
    from slo_lab.slo import RequestRecord

    offset, origin = 1_700_000_000.0, 4242.0
    rows = ["t_unix,num_requests_running,num_requests_waiting"]
    rows += [f"{offset + origin + t},10,{30 if t < 650 else 0}" for t in range(0, 1500, 5)]
    (tmp_path / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    recs = [
        RequestRecord(
            request_id=str(i),
            offered_at_s=float(i),
            outcome="ok",
            ttft_s=3.0 if 300 <= i < 650 else 0.1,
            e2e_s=6.0,
            output_tokens=132,
        )
        for i in range(1500)
    ]
    phases = [
        {"phase": "pre", "start_s": 0.0, "end_s": 300.0},
        {"phase": "burst", "start_s": 300.0, "end_s": 600.0},
        {"phase": "recovery", "start_s": 600.0, "end_s": 1500.0},
    ]
    out = trace_stage_summary(
        tmp_path, recs, origin_monotonic_s=origin, clock_offsets=[offset, offset], phases=phases
    )
    assert set(out["phase_summaries"]) == {"all", "pre", "burst", "recovery"}
    assert out["phase_summaries"]["pre"]["offered"] == 240  # first 60 s of the pre phase skipped
    assert out["phase_summaries"]["burst"]["attainment"] == 0.0
    assert out["time_to_recover_s"] == 50.0  # queue empty from 650, TTFT fast from 650
    assert out["waiting_samples"] == 300
    assert proc_cpu_s(None) is None
