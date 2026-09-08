"""Run one measured stage against an already-running vLLM server (spec §3.3, §3.4, §8 W2).

A stage is (cell, load point, seed). The server lifecycle and the quiet-GPU gate belong to the
batch driver, because the gate must run before the server takes the card. This module does the
part that has to be reproducible per stage:

1. warm-up: ``warmup_requests`` sequential streaming completions, TTFT of each kept;
2. power sampling (NVML, 1 s) and ``/metrics`` scraping (5 s) in background threads;
3. inference-perf with a generated config (open-loop Poisson or closed-loop concurrency);
4. adaptation to ``records.jsonl``, a summary over the measurement window, and a
   ``manifest.json`` binding versions, flags, environment switches and raw-file digests.

The bulky inference-perf outputs stay under the run directory (ignored by git); the adapted
records, summary, power CSV, metrics CSV and manifest are what get promoted to evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slo_lab.harness.ipf_adapter import adapt_file, write_records_jsonl
from slo_lab.harness.ipf_config import closed_loop_config, open_loop_config, write_config
from slo_lab.harness.metrics_scraper import MetricsScraper, fetch_metrics
from slo_lab.slo import DEFAULT_SLO, filter_window, summarise
from slo_lab.stats import percentile


@dataclass
class WarmupResult:
    n: int
    ttft_s: list[float]
    median_1_100: float | None
    median_101_200: float | None
    median_201_300: float | None
    wall_s: float


def _stream_ttft(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: float = 120.0
) -> float:
    """One streaming completion; returns TTFT in seconds (first SSE data line)."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": True,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    ttft: float | None = None
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if ttft is None and line.startswith("data:") and "[DONE]" not in line:
                ttft = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError("stream ended without a data chunk")
    return ttft


def warm_up(base_url: str, model: str, *, n: int = 100, output_tokens: int = 132) -> WarmupResult:
    """``n`` sequential requests with distinct nonce prompts; TTFT per request is kept."""
    started = time.perf_counter()
    ttfts: list[float] = []
    for i in range(n):
        prompt = (
            f"warmup nonce {i:05d} {hashlib.sha256(str(i).encode()).hexdigest()[:16]} "
            + "the quick brown fox " * 20
        )
        ttfts.append(_stream_ttft(base_url, model, prompt, output_tokens))

    def med(lo: int, hi: int) -> float | None:
        block = ttfts[lo:hi]
        return float(percentile(block, 50)) if len(block) >= 10 else None

    return WarmupResult(
        n=n,
        ttft_s=ttfts,
        median_1_100=med(0, 100),
        median_101_200=med(100, 200),
        median_201_300=med(200, 300),
        wall_s=time.perf_counter() - started,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _io_pressure() -> str | None:
    try:
        return Path("/proc/pressure/io").read_text(encoding="utf-8").splitlines()[0]
    except OSError:
        return None


class _PowerThread:
    """Wrap ``run_sampler`` with the NVML reader in a thread; degrades to a note without NVML."""

    def __init__(self, out_path: Path, phase: str) -> None:
        self.out_path = out_path
        self.phase = phase
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.note: str | None = None
        self.rows = 0

    def _run(self) -> None:
        try:
            from slo_lab.power.sampler import NvmlReader, run_sampler
        except Exception as exc:
            self.note = f"power sampler unavailable: {exc}"
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            reader = NvmlReader()
        except Exception as exc:
            self.note = f"NVML unavailable: {exc}"
            return
        with self.out_path.open("w", encoding="utf-8", newline="") as handle:
            self.rows = run_sampler(
                reader, handle, interval_s=1.0, phase=self.phase, stop=self._stop.is_set
            )

    def start(self) -> _PowerThread:
        self._thread = threading.Thread(target=self._run, name="power-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)


def run_stage(
    *,
    run_dir: Path,
    cell: str,
    model: str,
    base_url: str,
    metrics_url: str,
    seed: int,
    kind: str,
    rate_rps: float | None = None,
    duration_s: int = 300,
    concurrency: int | None = None,
    num_requests: int | None = None,
    warmup_requests: int = 100,
    discard_first_s: float = 60.0,
    inference_perf_bin: str = "inference-perf",
    engine_flags: dict[str, Any] | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    io_before = _io_pressure()

    # 1. warm-up (its own power phase so the measurement window is clean)
    power_warm = _PowerThread(run_dir / "power-warmup.csv", "warmup").start()
    warm = warm_up(base_url, model, n=warmup_requests) if warmup_requests > 0 else None
    power_warm.stop()

    # 2. background sampling for the measured window
    power = _PowerThread(run_dir / "power.csv", "measure").start()
    scraper = MetricsScraper(
        metrics_url, run_dir / "metrics.csv", interval_s=5.0, fetch=fetch_metrics
    ).start()
    metrics_before = None
    try:
        from slo_lab.harness.metrics_scraper import parse_metrics

        metrics_before = parse_metrics(fetch_metrics(metrics_url))
    except Exception:
        pass

    # 3. inference-perf
    report_dir = run_dir / "ipf"
    if kind == "open_loop":
        assert rate_rps is not None
        cfg = open_loop_config(
            model=model,
            base_url=base_url,
            report_dir=str(report_dir),
            rate_rps=rate_rps,
            duration_s=duration_s,
            seed=seed,
            workers=workers,
        )
    elif kind == "closed_loop":
        assert concurrency is not None and num_requests is not None
        cfg = closed_loop_config(
            model=model,
            base_url=base_url,
            report_dir=str(report_dir),
            concurrency=concurrency,
            num_requests=num_requests,
            seed=seed,
            workers=workers,
        )
    else:
        raise ValueError(f"unknown stage kind {kind}")
    cfg_path = write_config(cfg, run_dir / "inference-perf.yaml")
    t_load_start = time.time()
    proc = subprocess.run(
        [inference_perf_bin, "-c", str(cfg_path), "--log-level", "INFO"],
        capture_output=True,
        text=True,
        timeout=duration_s + 1800,
        check=False,
    )
    t_load_end = time.time()
    (run_dir / "inference-perf.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8"
    )

    scraper.stop()
    power.stop()
    metrics_after = None
    try:
        from slo_lab.harness.metrics_scraper import parse_metrics

        metrics_after = parse_metrics(fetch_metrics(metrics_url))
    except Exception:
        pass

    # 4. adapt + summarise
    per_request = report_dir / "per_request_lifecycle_metrics.json"
    result: dict[str, Any] = {
        "cell": cell,
        "model": model,
        "seed": seed,
        "kind": kind,
        "rate_rps": rate_rps,
        "duration_s": duration_s,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "started_at": started_at,
        "inference_perf_returncode": proc.returncode,
        "load_wall_s": round(t_load_end - t_load_start, 2),
        "warmup": asdict(warm) | {"ttft_s": None} if warm else None,
        "warmup_ttft_median_s": (float(percentile(warm.ttft_s, 50)) if warm else None),
        "io_pressure_before": io_before,
        "io_pressure_after": _io_pressure(),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "power_rows": power.rows,
        "power_note": power.note,
        "metrics_rows": scraper.rows,
        "engine_flags": engine_flags or {},
        "env": {
            k: os.environ.get(k)
            for k in (
                "VLLM_WSL2_ENABLE_PIN_MEMORY",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "HF_HUB_OFFLINE",
            )
        },
        "platform": platform.platform(),
    }
    if warm:
        (run_dir / "warmup-ttft.json").write_text(
            json.dumps({"ttft_s": warm.ttft_s}, indent=0) + "\n", encoding="utf-8"
        )
    if per_request.exists():
        records = adapt_file(per_request)
        write_records_jsonl(records, run_dir / "records.jsonl")
        window = filter_window(records, start_s=discard_first_s) if kind == "open_loop" else records
        window_s = (
            (duration_s - discard_first_s)
            if kind == "open_loop"
            else (max(r.offered_at_s for r in records) if records else None)
        )
        summary = summarise(window, DEFAULT_SLO, window_s=window_s) if window else None
        ok = [r for r in window if r.ttft_s is not None and r.e2e_s is not None]
        ttfts = [r.ttft_s for r in ok]  # type: ignore[misc]
        tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
        result.update(
            {
                "records": len(records),
                "window_records": len(window),
                "summary": summary.model_dump() if summary else None,
                "ttft_p50_s": float(percentile(ttfts, 50)) if ttfts else None,
                "ttft_p95_s": float(percentile(ttfts, 95)) if ttfts else None,
                "tpot_p50_s": float(percentile(tpots, 50)) if tpots else None,
                "tpot_p95_s": float(percentile(tpots, 95)) if tpots else None,
                "achieved_rps": (len(window) / window_s) if window_s else None,
                "output_tok_per_s": (sum(r.output_tokens or 0 for r in ok) / window_s)
                if window_s
                else None,
                "raw_sha256": {"per_request_lifecycle_metrics.json": _sha256(per_request)},
            }
        )
    else:
        result.update(
            {
                "records": 0,
                "error": "inference-perf produced no per-request file",
                "stderr_tail": proc.stderr[-800:],
            }
        )
    (run_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return result
