"""Confidence intervals used throughout the lab (design spec §3.6).

- Latency percentiles: percentile bootstrap, B = 1000 by default, 95% two-sided.
- Proportions (SLO attainment, rejection rate, TMMLU+ accuracy): Wilson score interval.
- Paired comparisons between cells (same seeds): bootstrap over paired indices.

Every function is deterministic for a given `seed` so `make reproduce` yields a zero diff.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

Z_95 = 1.959963984540054
DEFAULT_N_BOOT = 1000

Statistic = Callable[[np.ndarray], float]


class CI(NamedTuple):
    point: float
    lo: float
    hi: float


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy default), q in [0, 100]."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    return float(np.percentile(arr, q))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Statistic,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI of `statistic` over `values` (resampling with replacement)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap of an empty sample is undefined")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    point = float(statistic(arr))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.fromiter((float(statistic(arr[row])) for row in idx), dtype=float, count=n_boot)
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return CI(point, float(lo), float(hi))


def percentile_ci(
    values: Sequence[float],
    q: float,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Bootstrap CI for the q-th percentile (e.g. q=95 for TTFT p95)."""
    return bootstrap_ci(
        values, lambda s: float(np.percentile(s, q)), n_boot=n_boot, alpha=alpha, seed=seed
    )


def paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    statistic: Statistic,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Bootstrap CI of statistic(a) - statistic(b) resampling *paired* indices jointly.

    Used for seed-paired cell comparisons (spec §3.6) and TMMLU+ same-question deltas (§5).
    """
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    if xa.size == 0 or xa.shape != xb.shape:
        raise ValueError("paired samples must be non-empty and of equal length")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    point = float(statistic(xa)) - float(statistic(xb))
    idx = rng.integers(0, xa.size, size=(n_boot, xa.size))
    boots = np.fromiter(
        (float(statistic(xa[row])) - float(statistic(xb[row])) for row in idx),
        dtype=float,
        count=n_boot,
    )
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return CI(point, float(lo), float(hi))


def wilson_ci(successes: int, n: int, *, z: float = Z_95) -> CI:
    """Wilson score interval for a binomial proportion.

    Returns (p_hat, lo, hi). Raises on n == 0 rather than guessing.
    """
    if n <= 0:
        raise ValueError("Wilson interval needs n >= 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must be within [0, n]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return CI(p, max(0.0, centre - half), min(1.0, centre + half))
