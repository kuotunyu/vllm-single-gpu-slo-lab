"""Paired TMMLU+ comparison between two cells over the same items (spec §5).

Two cells scored on the same 19,680-item test set are not two independent samples: every item is
answered by both. Comparing their Wilson intervals throws that pairing away and can leave a real
difference "not significant" because each interval is ~+-0.007 wide while the paired difference
is far better determined. This module keeps the pairing:

- ``mcnemar_exact`` gives the two-sided exact p-value from the discordant counts (the items where
  exactly one cell was right), which is the only information the null hypothesis is about;
- ``compare`` also reports the accuracy difference with a paired bootstrap interval
  (``slo_lab.stats.paired_diff_ci``, B = 1000 per the preregistration).

Items that errored on either side are excluded rather than scored as wrong, so a transient
failure in one run cannot masquerade as a quality difference; the count is reported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slo_lab.stats import DEFAULT_N_BOOT, paired_diff_ci

ItemKey = tuple[str, int]


def read_scored(path: Path) -> dict[ItemKey, bool]:
    """``(subject, index) -> correct`` from a ``tmmluplus_eval`` output file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(r["subject"], int(r["index"])): bool(r["correct"]) for r in data["records"]}


def _errored(path: Path) -> set[ItemKey]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(r["subject"], int(r["index"])) for r in data["records"] if r.get("error")}


def mcnemar_exact(discordant_a_only: int, discordant_b_only: int) -> float:
    """Two-sided exact McNemar p-value for the two discordant counts.

    Under the null the two cells are equally likely to be the one that got a discordant item
    right, so the smaller count is Binomial(n, 0.5) with n the total discordance. Exact rather
    than chi-squared because the discordant counts can be small for near-identical cells.

    Computed in log space: on the full TMMLU+ set two cells disagree on thousands of items, and
    ``2.0 ** n`` overflows a float above n = 1024. Underflows to 0.0 when the tail is below the
    smallest representable double, which for this test means "far beyond any usable threshold".
    """
    n = discordant_a_only + discordant_b_only
    if n == 0:
        return 1.0
    k = min(discordant_a_only, discordant_b_only)
    log_n_factorial = math.lgamma(n + 1)
    log_half_n = -n * math.log(2.0)
    terms = [
        log_n_factorial - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
        for i in range(k + 1)
    ]
    biggest = max(terms)
    log_tail = biggest + math.log(math.fsum(math.exp(t - biggest) for t in terms))
    if log_tail < -745.0:  # exp() underflows below this
        return 0.0
    return min(1.0, 2.0 * math.exp(log_tail))


@dataclass(frozen=True)
class PairedQuality:
    """Result of comparing cell ``b`` against cell ``a`` on their shared items."""

    a_name: str
    b_name: str
    n_shared: int
    excluded_errors: int
    a_correct: int
    b_correct: int
    a_accuracy: float
    b_accuracy: float
    delta: float
    delta_ci95: tuple[float, float]
    discordant_a_only: int
    discordant_b_only: int
    p_value: float

    def as_dict(self) -> dict[str, object]:
        return {
            "a": self.a_name,
            "b": self.b_name,
            "n_shared": self.n_shared,
            "excluded_errors": self.excluded_errors,
            "a_correct": self.a_correct,
            "b_correct": self.b_correct,
            "a_accuracy": self.a_accuracy,
            "b_accuracy": self.b_accuracy,
            "delta_b_minus_a": self.delta,
            "delta_ci95": list(self.delta_ci95),
            "discordant_a_only": self.discordant_a_only,
            "discordant_b_only": self.discordant_b_only,
            "mcnemar_exact_p": self.p_value,
        }


def find_full_sets(root: Path) -> dict[str, Path]:
    """``cell -> evidence/raw/w2/<cell>/tmmluplus/full.json`` for every cell that has one."""
    base = Path(root) / "evidence" / "raw" / "w2"
    if not base.is_dir():
        return {}
    return {
        d.name: d / "tmmluplus" / "full.json"
        for d in sorted(base.iterdir())
        if (d / "tmmluplus" / "full.json").exists()
    }


def rebuild_paired_tables(root: Path, out_dir: Path, *, baseline: str = "bf16") -> list[str]:
    """Write the paired quality table for every cell with a full set; returns the cells compared.

    The baseline is the unquantised cell when it exists, so each row reads "what quantisation
    costs"; otherwise the alphabetically first cell is used so the output stays deterministic.
    """
    full = find_full_sets(root)
    if len(full) < 2:
        return []
    base_cell = baseline if baseline in full else sorted(full)[0]
    results = [
        compare(full[base_cell], full[cell], a_name=base_cell, b_name=cell)
        for cell in sorted(full)
        if cell != base_cell
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": base_cell,
        "n_boot": DEFAULT_N_BOOT,
        "seed": 0,
        "rows": [r.as_dict() for r in results],
    }
    (out_dir / "paired.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    header = (
        f"# TMMLU+ paired comparison against `{base_cell}` (same items, exact McNemar)\n\n"
        "| cell | n shared | accuracy | baseline accuracy | delta | 95% paired CI | "
        "only cell right | only baseline right | McNemar p |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {r.b_name} | {r.n_shared} | {r.b_accuracy:.4f} | {r.a_accuracy:.4f} | "
        f"{r.delta:+.4f} | [{r.delta_ci95[0]:+.4f}, {r.delta_ci95[1]:+.4f}] | "
        f"{r.discordant_b_only} | {r.discordant_a_only} | {r.p_value:.3g} |\n"
        for r in results
    )
    (out_dir / "paired.md").write_text(header + rows, encoding="utf-8", newline="\n")
    return [r.b_name for r in results]


def compare(
    a_path: Path,
    b_path: Path,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    a_name: str | None = None,
    b_name: str | None = None,
) -> PairedQuality:
    """Compare two scored files item by item. ``delta`` is ``b`` minus ``a``."""
    a_path, b_path = Path(a_path), Path(b_path)
    a_scored, b_scored = read_scored(a_path), read_scored(b_path)
    bad = _errored(a_path) | _errored(b_path)
    keys = sorted((a_scored.keys() & b_scored.keys()) - bad)
    if not keys:
        raise ValueError(f"no shared items between {a_path.name} and {b_path.name}")
    a_vec = [1.0 if a_scored[k] else 0.0 for k in keys]
    b_vec = [1.0 if b_scored[k] else 0.0 for k in keys]
    a_correct = int(sum(a_vec))
    b_correct = int(sum(b_vec))
    n = len(keys)
    # paired_diff_ci returns statistic(first) - statistic(second); we want b - a
    ci = paired_diff_ci(b_vec, a_vec, lambda x: float(np.mean(x)), n_boot=n_boot, seed=seed)
    a_only = sum(1 for k in keys if a_scored[k] and not b_scored[k])
    b_only = sum(1 for k in keys if b_scored[k] and not a_scored[k])
    return PairedQuality(
        a_name=a_name or a_path.parent.parent.name,
        b_name=b_name or b_path.parent.parent.name,
        n_shared=n,
        excluded_errors=len(bad & (a_scored.keys() & b_scored.keys())),
        a_correct=a_correct,
        b_correct=b_correct,
        a_accuracy=a_correct / n,
        b_accuracy=b_correct / n,
        delta=b_correct / n - a_correct / n,
        delta_ci95=(ci.lo, ci.hi),
        discordant_a_only=a_only,
        discordant_b_only=b_only,
        p_value=mcnemar_exact(a_only, b_only),
    )
