"""Seeded burst traces for inference-perf trace replay (spec §3.3, ADR 0012).

inference-perf 0.6.1 runs a multi-stage Poisson load one stage at a time and waits for every
request of a stage to finish before the next stage starts, so a 0.5x -> 1.5x -> 0.5x profile
would drain the burst backlog with no arrivals and start the recovery phase late. The admission
trace is therefore one continuous ``trace_replay`` stage fed by a file this module writes.

Arrivals are a piecewise-constant Poisson process drawn from ``random.Random(seed)``, with one
extra arrival pinned at t = 0 so the replay's own origin (its first row) is the trace origin and
phase boundaries stay at their nominal offsets. The three admission policies of a seed replay the
same file, which is what makes the comparison paired; inference-perf's built-in Poisson timer is
unseeded and could not give that. Its Azure reader keeps two fractional digits of a timestamp,
so arrivals are floored to a 10 ms grid here and the committed file is exactly what is replayed.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import yaml

HEADER = "TIMESTAMP,ContextTokens,GeneratedTokens"
ORIGIN = datetime(2026, 1, 1)
TICKS_PER_S = 100  # inference-perf keeps two fractional digits of each timestamp
_THREE_PHASES = ("pre", "burst", "recovery")

Profile = Sequence[tuple[float, int]]


def load_profile(path: Path) -> list[tuple[float, int]]:
    """``[(rate_multiplier, duration_s), ...]`` from a ``config/traffic`` multistage YAML."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [(float(s["rate_multiplier"]), int(s["duration_s"])) for s in cfg["stages"]]


def phase_bounds(profile: Profile) -> list[tuple[str, float, float]]:
    """Named ``(phase, start_s, end_s)`` rows; a three-stage profile is pre / burst / recovery."""
    names = _THREE_PHASES if len(profile) == 3 else tuple(f"phase-{i}" for i in range(len(profile)))
    out: list[tuple[str, float, float]] = []
    start = 0.0
    for name, (_, duration) in zip(names, profile, strict=True):
        out.append((name, start, start + float(duration)))
        start += float(duration)
    return out


def burst_arrivals(rate_ref: float, profile: Profile, seed: int) -> list[float]:
    """Arrival offsets (s) for ``profile`` at ``multiplier x rate_ref``, on the 10 ms grid."""
    if rate_ref <= 0:
        raise ValueError("rate_ref must be > 0")
    rng = random.Random(seed)
    ticks = [0]
    start = 0.0
    for multiplier, duration in profile:
        end = start + float(duration)
        rate = multiplier * rate_ref
        if rate > 0:
            t = start
            while True:
                t += rng.expovariate(rate)
                if t >= end:
                    break
                ticks.append(int(t * TICKS_PER_S))
        start = end
    ticks.sort()
    return [k / TICKS_PER_S for k in ticks]


def write_azure_trace(
    path: Path, arrivals: Sequence[float], *, input_tokens: int = 108, output_tokens: int = 132
) -> str:
    """Write ``TIMESTAMP,ContextTokens,GeneratedTokens`` rows; returns the file's sha256."""
    lines = [HEADER]
    for t in arrivals:
        centi = round(t * TICKS_PER_S)
        whole, frac = divmod(centi, TICKS_PER_S)
        stamp = ORIGIN + timedelta(seconds=whole)
        lines.append(f"{stamp:%Y-%m-%d %H:%M:%S}.{frac:02d},{input_tokens},{output_tokens}")
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
