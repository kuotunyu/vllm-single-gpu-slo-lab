"""Time-bucketed views of a replayed trace, for figures and explainer animations.

Pure logic on the canonical per-request records (``slo_lab.slo.RequestRecord``): fixed-width
buckets of offered requests with attainment, TTFT p95 and rejection counts; a rolling merge of
consecutive buckets (so an animated readout does not jump bucket to bucket); a fixed-step
resampling of a queue series; and the trace phases of a stage manifest. No plotting or
animation dependency, so it is tested in CI like the rest of the package.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from slo_lab.slo import DEFAULT_SLO, Outcome, RequestRecord, Slo, meets_slo
from slo_lab.stats import percentile


@dataclass(frozen=True)
class Bucket:
    """Offered requests in ``[start_s, end_s)`` of the trace, by offer time."""

    start_s: float
    end_s: float
    offered: int
    met: int
    rejected: int
    timeouts: int
    ok_ttfts: tuple[float, ...]  # sorted TTFTs of successful requests, kept so buckets can merge

    @property
    def attainment(self) -> float | None:
        return self.met / self.offered if self.offered else None

    @property
    def ttft_p95_s(self) -> float | None:
        return percentile(self.ok_ttfts, 95) if self.ok_ttfts else None


@dataclass(frozen=True)
class Phase:
    name: str
    start_s: float
    end_s: float


def bucket_records(
    records: Iterable[RequestRecord],
    *,
    bucket_s: float = 10.0,
    end_s: float | None = None,
    slo: Slo = DEFAULT_SLO,
) -> list[Bucket]:
    """Fixed-width buckets from 0 to ``end_s`` (default: past the last offer).

    Offers at or beyond ``end_s`` are dropped.
    """
    if bucket_s <= 0:
        raise ValueError(f"bucket_s must be positive, got {bucket_s}")
    recs = list(records)
    if end_s is None:
        last = max((r.offered_at_s for r in recs), default=0.0)
        end_s = (math.floor(last / bucket_s) + 1) * bucket_s
    n = max(1, math.ceil(end_s / bucket_s - 1e-9))
    offered = [0] * n
    met = [0] * n
    rejected = [0] * n
    timeouts = [0] * n
    ttfts: list[list[float]] = [[] for _ in range(n)]
    for r in recs:
        i = int(r.offered_at_s // bucket_s)
        if i < 0 or i >= n:
            continue
        offered[i] += 1
        if meets_slo(r, slo):
            met[i] += 1
        if r.outcome is Outcome.REJECTED_429:
            rejected[i] += 1
        elif r.outcome is Outcome.TIMEOUT:
            timeouts[i] += 1
        if r.outcome is Outcome.OK and r.ttft_s is not None:
            ttfts[i].append(r.ttft_s)
    return [
        Bucket(
            i * bucket_s,
            (i + 1) * bucket_s,
            offered[i],
            met[i],
            rejected[i],
            timeouts[i],
            tuple(sorted(ttfts[i])),
        )
        for i in range(n)
    ]


def rolling(buckets: Sequence[Bucket], window: int) -> list[Bucket]:
    """Bucket ``i`` merged with the ``window - 1`` buckets before it (fewer at the start)."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    out: list[Bucket] = []
    for i, b in enumerate(buckets):
        chunk = buckets[max(0, i - window + 1) : i + 1]
        out.append(
            Bucket(
                chunk[0].start_s,
                b.end_s,
                sum(c.offered for c in chunk),
                sum(c.met for c in chunk),
                sum(c.rejected for c in chunk),
                sum(c.timeouts for c in chunk),
                tuple(sorted(t for c in chunk for t in c.ok_ttfts)),
            )
        )
    return out


def downsample(
    series: Sequence[tuple[float, float]], step_s: float, end_s: float, *, start_s: float = 0.0
) -> list[tuple[float, float]]:
    """``(t, value)`` on a fixed grid, the value being the latest sample at or before ``t``.

    0.0 before the first sample, so an animation can look values up by frame.
    """
    if step_s <= 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    pts = sorted(series)
    times = [t for t, _ in pts]
    out: list[tuple[float, float]] = []
    k = 0
    while True:
        t = start_s + k * step_s
        if t > end_s + 1e-9:
            break
        i = bisect_right(times, t) - 1
        out.append((round(t, 6), pts[i][1] if i >= 0 else 0.0))
        k += 1
    return out


def phases_of(manifest: dict[str, Any]) -> list[Phase]:
    """The trace phases a stage manifest records (``trace.phases``), in order."""
    trace = manifest.get("trace") or {}
    return [
        Phase(str(p["phase"]), float(p["start_s"]), float(p["end_s"]))
        for p in trace.get("phases") or []
    ]
