"""Generate inference-perf 0.6.1 configs for one measured stage (spec §3.3).

Two shapes are produced from the same prompt settings:

- open-loop Poisson at a fixed offered rate for ``duration_s`` (the headline sweep);
- closed-loop at a fixed concurrency for ``num_requests`` (the r_sat / capacity sweep).

Fixed prompt geometry: 108 input / 132 output tokens (Google Inference Quickstart medians via
the memo), ``ignore_eos`` so output length is exact, completion API (0.6.1's synthetic datagen
rejects chat), streaming so TTFT exists. ``base_seed`` pins the synthetic corpus and arrival
process so every cell sees the same request sequence for a given seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

INPUT_TOKENS = 108
OUTPUT_TOKENS = 132


def _fixed(n: int) -> dict[str, Any]:
    return {"min": n, "max": n, "mean": n, "std_dev": 0, "total_count": 1}


def _common(
    *, model: str, base_url: str, report_dir: str, seed: int, timeout_s: float
) -> dict[str, Any]:
    return {
        "api": {"type": "completion", "streaming": True},
        "data": {
            "type": "synthetic",
            "input_distribution": _fixed(INPUT_TOKENS),
            "output_distribution": _fixed(OUTPUT_TOKENS),
        },
        "server": {"type": "vllm", "model_name": model, "base_url": base_url, "ignore_eos": True},
        "tokenizer": {"pretrained_model_name_or_path": model},
        "report": {
            "request_lifecycle": {"summary": True, "per_stage": True, "per_request": True},
            "prometheus": None,
            "goodput": {"constraints": {"ttft": 1.0, "tpot": 0.05}},
        },
        "storage": {"local_storage": {"path": report_dir}},
        "_seed": seed,
        "_timeout": timeout_s,
    }


def open_loop_config(
    *,
    model: str,
    base_url: str,
    report_dir: str,
    rate_rps: float,
    duration_s: int,
    seed: int,
    workers: int = 4,
    timeout_s: float = 300.0,
    worker_max_concurrency: int = 4096,
) -> dict[str, Any]:
    """Poisson stage. ``worker_max_concurrency`` is high on purpose: at 2 x r_sat the server's
    native queue holds ~13k requests and the client must keep offering at the Poisson rate,
    otherwise client-side queueing would hide the server latency the sweep is meant to show
    (spec §3.3, client timeout 300 s); the batch driver raises the fd limit to match."""
    cfg = _common(
        model=model, base_url=base_url, report_dir=report_dir, seed=seed, timeout_s=timeout_s
    )
    cfg["load"] = {
        "type": "poisson",
        "interval": 1.0,
        "stages": [{"rate": float(rate_rps), "duration": int(duration_s)}],
        "num_workers": workers,
        "worker_max_concurrency": int(worker_max_concurrency),
        "request_timeout": cfg.pop("_timeout"),
        "base_seed": cfg.pop("_seed"),
    }
    return cfg


def closed_loop_config(
    *,
    model: str,
    base_url: str,
    report_dir: str,
    concurrency: int,
    num_requests: int,
    seed: int,
    workers: int = 4,
    timeout_s: float = 300.0,
    worker_max_concurrency: int = 4096,
) -> dict[str, Any]:
    cfg = _common(
        model=model, base_url=base_url, report_dir=report_dir, seed=seed, timeout_s=timeout_s
    )
    cfg["load"] = {
        "type": "concurrent",
        "stages": [{"num_requests": int(num_requests), "concurrency_level": int(concurrency)}],
        "num_workers": workers,
        "worker_max_concurrency": int(worker_max_concurrency),
        "request_timeout": cfg.pop("_timeout"),
        "base_seed": cfg.pop("_seed"),
    }
    return cfg


def trace_replay_config(
    *,
    model: str,
    base_url: str,
    report_dir: str,
    trace_file: str,
    duration_s: int,
    mean_rate_rps: float,
    seed: int,
    workers: int = 4,
    timeout_s: float = 300.0,
    worker_max_concurrency: int = 4096,
    worker_max_tcp_connections: int = 4096,
) -> dict[str, Any]:
    """One continuous stage replaying a pre-generated arrival trace (W3, ADR 0012).

    inference-perf finishes every request of a stage before starting the next, so a multistage
    Poisson burst would drain its backlog with no arrivals; replaying one trace keeps arrivals
    flowing through the burst and the recovery. The random generator takes each request's
    108 / 132 token counts from the trace rows and ignores distributions, so none are passed.
    ``worker_max_tcp_connections`` is raised from inference-perf's default of 2,500 per worker so
    the client pool can never cap the native-queue arm (about 8k in flight at FP8's burst end).
    ``mean_rate_rps`` only labels the stage; arrivals come from the trace file.
    """
    cfg = _common(
        model=model, base_url=base_url, report_dir=report_dir, seed=seed, timeout_s=timeout_s
    )
    trace = {"file": str(trace_file), "format": "AzurePublicDataset"}
    cfg["data"] = {"type": "random", "trace": dict(trace)}
    cfg["load"] = {
        "type": "trace_replay",
        "trace": dict(trace),
        "interval": 0,
        "stages": [{"rate": round(float(mean_rate_rps), 4), "duration": int(duration_s)}],
        "num_workers": workers,
        "worker_max_concurrency": int(worker_max_concurrency),
        "worker_max_tcp_connections": int(worker_max_tcp_connections),
        "request_timeout": cfg.pop("_timeout"),
        "base_seed": cfg.pop("_seed"),
    }
    return cfg


def write_config(cfg: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
