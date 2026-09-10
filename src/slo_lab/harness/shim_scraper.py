"""Poll the admission shim's ``/_shim/stats`` into ``shim.csv`` during a trace stage (W3).

vLLM's ``/metrics`` only sees requests the shim let through, so under the bounded-queue policy the
queue lives in the shim. Time-to-recover needs the total (engine waiting + shim waiting), and the
rejection counters show when each policy started and stopped turning requests away.
"""

from __future__ import annotations

import csv
import json
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

COLUMNS: tuple[str, ...] = (
    "in_flight",
    "waiting",
    "admitted",
    "completed",
    "upstream_errors",
    "rejected_cap_full",
    "rejected_queue_full",
    "rejected_queue_timeout",
)


def fetch_stats(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stats_row(stats: dict[str, Any]) -> list[Any]:
    """One CSV row (without the timestamp) from a ``/_shim/stats`` payload."""
    rejected = stats.get("rejected") or {}
    return [
        stats.get("in_flight", ""),
        stats.get("waiting", ""),
        stats.get("admitted", ""),
        stats.get("completed", ""),
        stats.get("upstream_errors", ""),
        rejected.get("cap_full", 0),
        rejected.get("queue_full", 0),
        rejected.get("queue_timeout", 0),
    ]


class ShimScraper:
    """Background thread writing one ``shim.csv`` row per interval; keeps the last payload."""

    def __init__(
        self,
        url: str,
        out_path: Path,
        *,
        interval_s: float = 5.0,
        fetch: Callable[[str], dict[str, Any]] = fetch_stats,
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
        self.last: dict[str, Any] | None = None

    def _run(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["t_unix", *COLUMNS])
            while True:
                stamp = self._clock()
                try:
                    stats = self._fetch(self.url)
                    writer.writerow([f"{stamp:.3f}", *stats_row(stats)])
                    self.last = stats
                    self.rows += 1
                except Exception:
                    self.errors += 1
                handle.flush()
                if self._stop.wait(self.interval_s):
                    break

    def start(self) -> ShimScraper:
        self._thread = threading.Thread(target=self._run, name="shim-scraper", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 10)
