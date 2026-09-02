"""SLO attainment, goodput, capacity (r_SLO) and the SLO-sensitivity grid (design spec §3.2, §3.6).

Definitions implemented here, verbatim from the spec where it decides and marked "proposal"
where the spec does:

- TPOT = (e2e - TTFT) / (output tokens - 1); NTPOT = e2e / output tokens.
- A request *meets the SLO* when both TTFT <= ttft_s and TPOT <= tpot_s.
- Attainment (headline) = met / offered. Offered includes HTTP 429 rejections, client timeouts
  and 5xx errors, all counted as misses (proposal). Admitted-only attainment is reported
  alongside for contrast.
- Goodput = met / window_s (requests per second that met the SLO).
- Rejection rate = (429 + timeout) / offered.
- r_SLO = the highest offered rate at which *every* seed reaches >= target attainment
  (target 0.95, proposal). The search walks up from the lowest rate and stops at the first rate
  where any seed fails, so a noisy recovery above a failing rate never counts (conservative).
  A rate that is missing a seed is treated as failing (paired design).
- SLO sensitivity: r_SLO recomputed from the same raw records for every (ttft, tpot) pair in
  TTFT in {0.5, 1, 2} s x TPOT in {30, 50, 100} ms (proposal).

Per-request records use this package's canonical schema (`RequestRecord`); an adapter from the
inference-perf raw JSON is W1 work and does not exist yet.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from slo_lab.stats import wilson_ci

ATTAINMENT_TARGET = 0.95
TTFT_GRID_S: tuple[float, ...] = (0.5, 1.0, 2.0)
TPOT_GRID_S: tuple[float, ...] = (0.03, 0.05, 0.10)


class Outcome(StrEnum):
    OK = "ok"
    REJECTED_429 = "rejected_429"
    TIMEOUT = "timeout"
    ERROR = "error"  # 5xx or transport failure


class RequestRecord(BaseModel):
    """One offered request. Latencies are seconds; `offered_at_s` is relative to stage start."""

    request_id: str
    offered_at_s: float = Field(ge=0.0)
    outcome: Outcome
    ttft_s: float | None = Field(default=None, ge=0.0)
    e2e_s: float | None = Field(default=None, ge=0.0)
    output_tokens: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ok_records_are_complete(self) -> RequestRecord:
        if self.outcome is Outcome.OK and (
            self.ttft_s is None or self.e2e_s is None or self.output_tokens is None
        ):
            raise ValueError("records with outcome=ok need ttft_s, e2e_s and output_tokens")
        if self.ttft_s is not None and self.e2e_s is not None and self.e2e_s < self.ttft_s:
            raise ValueError("e2e_s cannot be smaller than ttft_s")
        return self

    @property
    def tpot_s(self) -> float | None:
        """Time per output token; undefined (None) unless the request succeeded with >= 2 tokens."""
        if self.outcome is not Outcome.OK or self.output_tokens is None or self.output_tokens < 2:
            return None
        assert self.e2e_s is not None and self.ttft_s is not None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)

    @property
    def ntpot_s(self) -> float | None:
        """Normalised TPOT = e2e / output tokens (memo §3(b)); reported alongside TPOT."""
        if self.outcome is not Outcome.OK or not self.output_tokens:
            return None
        assert self.e2e_s is not None
        return self.e2e_s / self.output_tokens


class Slo(BaseModel):
    """Per-request thresholds. Defaults are the re-plan §4 SLO: TTFT <= 1 s, TPOT <= 50 ms."""

    ttft_s: float = Field(default=1.0, gt=0.0)
    tpot_s: float = Field(default=0.05, gt=0.0)


DEFAULT_SLO = Slo()


def meets_slo(record: RequestRecord, slo: Slo = DEFAULT_SLO) -> bool:
    """True only for a successful request within both thresholds.

    A successful request whose TPOT is undefined (fewer than 2 output tokens) is counted as a
    miss: the workload fixes output length, so this only happens when something went wrong.
    """
    if record.outcome is not Outcome.OK:
        return False
    tpot = record.tpot_s
    if tpot is None or record.ttft_s is None:
        return False
    return record.ttft_s <= slo.ttft_s and tpot <= slo.tpot_s


def filter_window(
    records: Iterable[RequestRecord], start_s: float = 0.0, end_s: float | None = None
) -> list[RequestRecord]:
    """Keep records offered inside [start_s, end_s). Use start_s=60 to drop the warm-up minute."""
    return [
        r
        for r in records
        if r.offered_at_s >= start_s and (end_s is None or r.offered_at_s < end_s)
    ]


class AttainmentSummary(BaseModel):
    offered: int
    admitted: int
    met: int
    rejected_429: int
    timeouts: int
    errors: int
    attainment_offered: float
    attainment_offered_ci95: tuple[float, float]
    attainment_admitted: float | None
    rejection_rate: float
    goodput_rps: float | None
    window_s: float | None
    slo: Slo


def summarise(
    records: Sequence[RequestRecord], slo: Slo = DEFAULT_SLO, *, window_s: float | None = None
) -> AttainmentSummary:
    """Headline attainment (offered denominator), admitted-only attainment, rejection rate, goodput."""
    offered = len(records)
    if offered == 0:
        raise ValueError("no offered requests in the measurement window")
    rejected = sum(r.outcome is Outcome.REJECTED_429 for r in records)
    timeouts = sum(r.outcome is Outcome.TIMEOUT for r in records)
    errors = sum(r.outcome is Outcome.ERROR for r in records)
    admitted = offered - rejected
    met = sum(meets_slo(r, slo) for r in records)
    ci = wilson_ci(met, offered)
    return AttainmentSummary(
        offered=offered,
        admitted=admitted,
        met=met,
        rejected_429=rejected,
        timeouts=timeouts,
        errors=errors,
        attainment_offered=met / offered,
        attainment_offered_ci95=(ci.lo, ci.hi),
        attainment_admitted=(met / admitted) if admitted else None,
        rejection_rate=(rejected + timeouts) / offered,
        goodput_rps=(met / window_s) if window_s else None,
        window_s=window_s,
        slo=slo,
    )


class SweepCell(BaseModel):
    """Attainment of one (offered rate, seed) run, already reduced to a proportion."""

    offered_rate: float = Field(gt=0.0)
    seed: int
    attainment: float = Field(ge=0.0, le=1.0)


class CapacityResult(BaseModel):
    r_slo: float | None
    target: float
    rates: list[float]
    seeds: list[int]
    min_attainment_by_rate: dict[float, float | None]
    limiting_rate: float | None  # first rate at which some seed missed the target (or had no data)


def r_slo(cells: Iterable[SweepCell], *, target: float = ATTAINMENT_TARGET) -> CapacityResult:
    """Capacity r_SLO: highest offered rate where all seeds reach >= target, contiguous from the bottom."""
    by_rate: dict[float, dict[int, float]] = {}
    seeds: set[int] = set()
    for cell in cells:
        by_rate.setdefault(cell.offered_rate, {})[cell.seed] = cell.attainment
        seeds.add(cell.seed)
    if not by_rate:
        raise ValueError("no sweep cells")
    rates = sorted(by_rate)
    seed_list = sorted(seeds)
    best: float | None = None
    limiting: float | None = None
    min_by_rate: dict[float, float | None] = {}
    for rate in rates:
        atts = by_rate[rate]
        complete = all(s in atts for s in seed_list)
        min_att = min(atts.values()) if complete else None
        min_by_rate[rate] = min_att
        if limiting is not None:
            continue  # already stopped; keep filling the table for the reader
        if min_att is not None and min_att >= target:
            best = rate
        else:
            limiting = rate
    return CapacityResult(
        r_slo=best,
        target=target,
        rates=rates,
        seeds=seed_list,
        min_attainment_by_rate=min_by_rate,
        limiting_rate=limiting,
    )


class SweepRun(BaseModel):
    """Raw per-request records of one (offered rate, seed) run, for zero-cost recomputation."""

    offered_rate: float = Field(gt=0.0)
    seed: int
    records: list[RequestRecord]


def sensitivity_grid(
    runs: Sequence[SweepRun],
    *,
    ttft_grid: Sequence[float] = TTFT_GRID_S,
    tpot_grid: Sequence[float] = TPOT_GRID_S,
    target: float = ATTAINMENT_TARGET,
    window_start_s: float = 0.0,
) -> dict[tuple[float, float], float | None]:
    """r_SLO for every (ttft_s, tpot_s) threshold pair, recomputed from raw records."""
    grid: dict[tuple[float, float], float | None] = {}
    windowed = [(run, filter_window(run.records, window_start_s)) for run in runs]
    for ttft in ttft_grid:
        for tpot in tpot_grid:
            slo = Slo(ttft_s=ttft, tpot_s=tpot)
            cells = [
                SweepCell(
                    offered_rate=run.offered_rate,
                    seed=run.seed,
                    attainment=summarise(recs, slo).attainment_offered,
                )
                for run, recs in windowed
                if recs
            ]
            grid[(ttft, tpot)] = r_slo(cells, target=target).r_slo if cells else None
    return grid


def grid_markdown(grid: dict[tuple[float, float], float | None], unit: str = "req/s") -> str:
    """Render the sensitivity grid as a Markdown table (rows: TTFT, columns: TPOT)."""
    ttfts = sorted({k[0] for k in grid})
    tpots = sorted({k[1] for k in grid})
    header = "| TTFT \\ TPOT | " + " | ".join(f"{t * 1000:.0f} ms" for t in tpots) + " |"
    sep = "|---|" + "---|" * len(tpots)
    rows = []
    for ttft in ttfts:
        cells = []
        for tpot in tpots:
            v = grid.get((ttft, tpot))
            cells.append("n/a" if v is None else f"{v:g} {unit}")
        rows.append(f"| {ttft:g} s | " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def read_records_jsonl(path: Path) -> list[RequestRecord]:
    """Load canonical per-request records (one JSON object per line; blank lines ignored)."""
    out: list[RequestRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(RequestRecord.model_validate(json.loads(line)))
    return out


def write_records_jsonl(path: Path, records: Iterable[RequestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
