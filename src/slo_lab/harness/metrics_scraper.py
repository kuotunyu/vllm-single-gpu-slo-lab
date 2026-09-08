"""Poll a vLLM ``/metrics`` endpoint and keep the few series the analysis needs as CSV.

Tracked (names frozen in ``evidence/metrics-names.txt``): ``num_requests_running``,
``num_requests_waiting``, ``kv_cache_usage_perc``, ``prefix_cache_queries_total``,
``prefix_cache_hits_total``, ``prompt_tokens_total``, ``generation_tokens_total``,
``request_success_total``. Parsing is a tiny Prometheus text-format reader restricted to
gauge/counter sample lines; labels are ignored because the server hosts one model.
"""

from __future__ import annotations

import csv
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

TRACKED: tuple[str, ...] = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)


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
    ) -> None:
        self.url = url
        self.out_path = out_path
        self.interval_s = interval_s
        self._fetch = fetch
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rows = 0
        self.errors = 0

    def _run(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_unix", *[name.removeprefix("vllm:") for name in TRACKED]])
            while not self._stop.is_set():
                stamp = self._clock()
                try:
                    values = parse_metrics(self._fetch(self.url))
                    writer.writerow([f"{stamp:.3f}", *[values.get(name, "") for name in TRACKED]])
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
