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
) -> dict[str, Any]:
    cfg = _common(
        model=model, base_url=base_url, report_dir=report_dir, seed=seed, timeout_s=timeout_s
    )
    cfg["load"] = {
        "type": "poisson",
        "interval": 1.0,
        "stages": [{"rate": float(rate_rps), "duration": int(duration_s)}],
        "num_workers": workers,
        "worker_max_concurrency": 256,
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
) -> dict[str, Any]:
    cfg = _common(
        model=model, base_url=base_url, report_dir=report_dir, seed=seed, timeout_s=timeout_s
    )
    cfg["load"] = {
        "type": "concurrent",
        "stages": [{"num_requests": int(num_requests), "concurrency_level": int(concurrency)}],
        "num_workers": workers,
        "worker_max_concurrency": 256,
        "request_timeout": cfg.pop("_timeout"),
        "base_seed": cfg.pop("_seed"),
    }
    return cfg


def write_config(cfg: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
