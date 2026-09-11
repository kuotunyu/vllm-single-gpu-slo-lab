"""Poll a vLLM ``/metrics`` endpoint and keep the few series the analysis needs as CSV.

Tracked (names frozen in ``evidence/metrics-names.txt``): ``num_requests_running``,
``num_requests_waiting``, ``kv_cache_usage_perc``, ``prefix_cache_queries_total``,
``prefix_cache_hits_total``, ``prompt_tokens_total``, ``generation_tokens_total``,
``request_success_total``; W4 (ADR 0015) adds ``num_preemptions_total`` and the three
spec-decode counters a server with speculative decoding exposes (``spec_decode_num_drafts_total``,
``spec_decode_num_draft_tokens_total``, ``spec_decode_num_accepted_tokens_total``; absent
otherwise). Parsing is a tiny Prometheus text-format reader restricted to gauge/counter sample
lines; labels are ignored because the server hosts one model, except ``parse_labelled`` for the
per-position acceptance counter.

``parse_histograms`` additionally reads the server-side latency histograms (TTFT, queue time,
time per output token, e2e). A before/after delta over a stage is the loadgen-independent view
of latency: it separates engine or API-server stalls from client artefacts (W2, 2026-09-09:
whole batches of identical multi-second client TTFTs appeared while host disk I/O was saturated).
"""

from __future__ import annotations

import csv
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

TRACKED: tuple[str, ...] = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    # W4 (ADR 0015 item 6): appended so the existing metrics.csv columns keep their place
    "vllm:num_preemptions_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
SPEC_DECODE_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def parse_metrics(text: str, tracked: tuple[str, ...] = TRACKED) -> dict[str, float]:
    """Sum the sample values of each tracked metric name (labels collapsed)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, value_part = line.rpartition(" ")
        name = name_part.split("{", 1)[0].strip()
        if name not in tracked:
            continue
        try:
            value = float(value_part.strip())
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


HISTOGRAMS: tuple[str, ...] = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
)


def _label(labels: str, key: str) -> str | None:
    for item in labels.rstrip("}").split(","):
        k, _, v = item.partition("=")
        if k.strip() == key:
            return v.strip().strip('"')
    return None


def parse_labelled(text: str, name: str, label: str) -> dict[str, float]:
    """Sample values of one metric summed per value of ``label`` (e.g. draft ``position``)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, value_part = line.rpartition(" ")
        full, _, labels = name_part.partition("{")
        if full.strip() != name:
            continue
        key = _label(labels, label)
        if key is None:
            continue
        try:
            value = float(value_part.strip())
        except ValueError:
            continue
        out[key] = out.get(key, 0.0) + value
    return out


def parse_histograms(text: str, names: tuple[str, ...] = HISTOGRAMS) -> dict[str, dict[str, Any]]:
    """Cumulative ``_bucket`` counts keyed by ``le`` plus ``_count``/``_sum`` per histogram.

    Labels other than ``le`` are collapsed, as in ``parse_metrics``. Values are cumulative since
    server start; take ``histogram_delta`` between two snapshots to isolate one stage.
    """
    out: dict[str, dict[str, Any]] = {
        name: {"buckets": {}, "count": 0.0, "sum": 0.0} for name in names
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, value_part = line.rpartition(" ")
        full, _, labels = name_part.partition("{")
        full = full.strip()
        try:
            value = float(value_part.strip())
        except ValueError:
            continue
        for name in names:
            if full == f"{name}_bucket":
                le = _label(labels, "le")
                if le is not None:
                    out[name]["buckets"][le] = out[name]["buckets"].get(le, 0.0) + value
            elif full == f"{name}_count":
                out[name]["count"] += value
            elif full == f"{name}_sum":
                out[name]["sum"] += value
            else:
                continue
            break
    return out


def histogram_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Counts accrued between two snapshots of one histogram (buckets stay cumulative in ``le``)."""
    b_buckets: dict[str, float] = before.get("buckets", {})
    a_buckets: dict[str, float] = after.get("buckets", {})
    return {
        "buckets": {le: float(n) - float(b_buckets.get(le, 0.0)) for le, n in a_buckets.items()},
        "count": float(after.get("count", 0.0)) - float(before.get("count", 0.0)),
        "sum": float(after.get("sum", 0.0)) - float(before.get("sum", 0.0)),
    }


def histogram_quantile_bounds(delta: dict[str, Any], q: float) -> tuple[float | None, float]:
    """(lower, upper) bucket edges enclosing quantile ``q``; upper is ``inf`` past the last edge."""
    buckets: dict[str, float] = delta.get("buckets", {})
    count = float(delta.get("count", 0.0))
    if count <= 0:
        return (None, float("inf"))
    prev: float | None = None
    for le, cum in sorted(((float(le), float(n)) for le, n in buckets.items()), key=lambda t: t[0]):
        if cum >= q * count:
            return (prev, le)
        prev = le
    return (prev, float("inf"))


def fetch_metrics(url: str, timeout_s: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


class MetricsScraper:
    """Background thread writing one CSV row per interval; ``stop()`` flushes and joins."""

    def __init__(
        self,
        url: str,
        out_path: Path,
        *,
        interval_s: float = 5.0,
        fetch: Callable[[str], str] = fetch_metrics,
        clock: Callable[[], float] = time.time,
        mono_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self.out_path = out_path
        self.interval_s = interval_s
        self._fetch = fetch
        self._clock = clock
        # t_mono is the clock inference-perf stamps requests with; in WSL2 the wall clock drifted
        # ~7 s against it within one 2.5-minute stage (W3 dry run, 2026-09-11), so the trace
        # analysis aligns scraped samples with records on t_mono, never on t_unix.
        self._mono_clock = mono_clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rows = 0
        self.errors = 0

    def _run(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            header = ["t_unix", *[name.removeprefix("vllm:") for name in TRACKED], "t_mono"]
            writer.writerow(header)
            while not self._stop.is_set():
                stamp, mono = self._clock(), self._mono_clock()
                try:
                    values = parse_metrics(self._fetch(self.url))
                    writer.writerow(
                        [f"{stamp:.3f}", *[values.get(name, "") for name in TRACKED], f"{mono:.3f}"]
                    )
                    self.rows += 1
                except Exception:
                    self.errors += 1
                handle.flush()
                self._stop.wait(self.interval_s)

    def start(self) -> MetricsScraper:
        self._thread = threading.Thread(target=self._run, name="metrics-scraper", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 10)
