"""Cost accounting (design spec §6).

Formula (re-plan §4):

    gpu_usd_per_h   = purchase_usd / (amortisation_years * 365 * duty_hours_per_day)
    power_usd_per_h = P_avg_kW * electricity_usd_per_kwh
    $/M output tok  = (gpu_usd_per_h + power_usd_per_h) / 3600 / output_tok_per_s * 1e6

`output_tok_per_s` and `P_avg` come from the same measurement window. Two values are always
reported: the measured one at the r_SLO operating point, and the utilisation-naive one at the
closed-loop peak with U = tok/s(r_SLO) / tok/s(peak) and the caveat "actual ≈ naive / U".
Wh per million output tokens is reported too because it is price-free and comparable across cards.

Power comes from 1 s NVML samples (power/sampler.py). `integrate` uses the trapezoidal rule over
sample timestamps; the idle baseline is the arithmetic mean of the idle-phase samples. The
headline cost uses *gross* board power (the card is dedicated to the service); the idle-subtracted
(net) figure is the marginal energy per token and is reported as a sensitivity, never as the
headline. Both are lower bounds for whole-machine electricity (claim ceiling 6).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Literal, NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

HOURS_PER_DAY_SENSITIVITY = 8

CostStatus = Literal["owner_input_pending", "owner_provided"]


class CostConfig(BaseModel):
    """`config/cost.yaml`. Field names are the spec's; extra keys (source notes) are kept."""

    model_config = ConfigDict(extra="allow")

    status: CostStatus = "owner_input_pending"
    gpu_purchase_price_twd: float = Field(gt=0.0)
    purchase_date: str
    amortisation_years: float = Field(gt=0.0)
    duty_hours_per_day: float = Field(gt=0.0, le=24.0)
    electricity_twd_per_kwh: float = Field(gt=0.0)
    twd_per_usd: float = Field(gt=0.0)
    input_output_price_ratio: float | None = Field(default=None, gt=0.0)

    @property
    def is_placeholder(self) -> bool:
        return self.status == "owner_input_pending"

    @property
    def purchase_usd(self) -> float:
        return self.gpu_purchase_price_twd / self.twd_per_usd

    @property
    def electricity_usd_per_kwh(self) -> float:
        return self.electricity_twd_per_kwh / self.twd_per_usd


def load_cost_config(path: Path) -> CostConfig:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return CostConfig.model_validate(data)


# --- formula ---------------------------------------------------------------------------------


def gpu_usd_per_hour(cfg: CostConfig, *, duty_hours_per_day: float | None = None) -> float:
    duty = cfg.duty_hours_per_day if duty_hours_per_day is None else duty_hours_per_day
    return cfg.purchase_usd / (cfg.amortisation_years * 365.0 * duty)


def power_usd_per_hour(p_avg_w: float, cfg: CostConfig) -> float:
    return (p_avg_w / 1000.0) * cfg.electricity_usd_per_kwh


def usd_per_million_output_tokens(
    gpu_usd_per_h: float, power_usd_per_h: float, output_tok_per_s: float
) -> float:
    if output_tok_per_s <= 0:
        raise ValueError("output_tok_per_s must be > 0")
    return (gpu_usd_per_h + power_usd_per_h) / 3600.0 / output_tok_per_s * 1e6


def wh_per_million_output_tokens(p_avg_w: float, output_tok_per_s: float) -> float:
    if output_tok_per_s <= 0:
        raise ValueError("output_tok_per_s must be > 0")
    return p_avg_w / (output_tok_per_s * 3600.0) * 1e6


def blended_usd_per_million_output_tokens(
    gpu_usd_per_h: float,
    power_usd_per_h: float,
    input_tok_per_s: float,
    output_tok_per_s: float,
    ratio: float,
) -> float:
    """Google Inference Quickstart blend (memo §3(d)):
    $/output token = (GPU $/s) / ((1/ratio) * input tok/s + output tok/s)."""
    usd_per_s = (gpu_usd_per_h + power_usd_per_h) / 3600.0
    denom = input_tok_per_s / ratio + output_tok_per_s
    if denom <= 0:
        raise ValueError("token rates must be positive")
    return usd_per_s / denom * 1e6


# --- power samples ---------------------------------------------------------------------------

POWER_CSV_COLUMNS = (
    "timestamp_iso",
    "t_s",
    "power_w",
    "util_gpu_pct",
    "mem_used_mib",
    "clocks_sm_mhz",
    "temp_c",
    "phase",
)


class PowerSample(NamedTuple):
    t_s: float
    power_w: float
    util_gpu_pct: float | None = None
    mem_used_mib: float | None = None
    clocks_sm_mhz: float | None = None
    temp_c: float | None = None
    phase: str = ""


def _opt_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_power_csv(path: Path) -> list[PowerSample]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [
            PowerSample(
                t_s=float(row["t_s"]),
                power_w=float(row["power_w"]),
                util_gpu_pct=_opt_float(row.get("util_gpu_pct")),
                mem_used_mib=_opt_float(row.get("mem_used_mib")),
                clocks_sm_mhz=_opt_float(row.get("clocks_sm_mhz")),
                temp_c=_opt_float(row.get("temp_c")),
                phase=row.get("phase") or "",
            )
            for row in reader
        ]
    return sorted(rows, key=lambda s: s.t_s)


def select_window(
    samples: Iterable[PowerSample],
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    phase: str | None = None,
) -> list[PowerSample]:
    """Filter samples by phase label and/or [start_s, end_s] time window."""
    return [
        s
        for s in samples
        if (phase is None or s.phase == phase)
        and (start_s is None or s.t_s >= start_s)
        and (end_s is None or s.t_s <= end_s)
    ]


class PowerSummary(BaseModel):
    n_samples: int
    duration_s: float
    energy_wh: float
    p_avg_w: float
    p_max_w: float
    temp_max_c: float | None
    clocks_sm_min_mhz: float | None


def integrate(samples: Sequence[PowerSample]) -> PowerSummary:
    """Trapezoidal energy over sample timestamps; P_avg = energy / duration."""
    if len(samples) < 2:
        raise ValueError("need at least 2 power samples to integrate")
    ordered = sorted(samples, key=lambda s: s.t_s)
    energy_j = 0.0
    for a, b in pairwise(ordered):
        dt = b.t_s - a.t_s
        if dt < 0:
            raise ValueError("timestamps must be non-decreasing")
        energy_j += 0.5 * (a.power_w + b.power_w) * dt
    duration = ordered[-1].t_s - ordered[0].t_s
    if duration <= 0:
        raise ValueError("window duration must be > 0")
    temps = [s.temp_c for s in ordered if s.temp_c is not None]
    clocks = [s.clocks_sm_mhz for s in ordered if s.clocks_sm_mhz is not None]
    return PowerSummary(
        n_samples=len(ordered),
        duration_s=duration,
        energy_wh=energy_j / 3600.0,
        p_avg_w=energy_j / duration,
        p_max_w=max(s.power_w for s in ordered),
        temp_max_c=max(temps) if temps else None,
        clocks_sm_min_mhz=min(clocks) if clocks else None,
    )


def idle_baseline_w(idle_samples: Sequence[PowerSample]) -> float:
    if not idle_samples:
        raise ValueError("no idle samples")
    return fmean(s.power_w for s in idle_samples)


# --- report ----------------------------------------------------------------------------------


class OperatingPoint(BaseModel):
    """Measured throughput and mean board power over one window."""

    output_tok_per_s: float = Field(gt=0.0)
    p_avg_w: float = Field(ge=0.0)
    input_tok_per_s: float | None = Field(default=None, ge=0.0)
    label: str = ""


class CostReport(BaseModel):
    config_status: CostStatus
    gpu_usd_per_h: float
    gpu_usd_per_h_duty8: float
    power_usd_per_h_at_r_slo: float
    usd_per_m_output_tok_at_r_slo: float
    usd_per_m_output_tok_at_r_slo_duty8: float
    wh_per_m_output_tok_at_r_slo: float
    usd_per_m_output_tok_naive: float | None
    wh_per_m_output_tok_naive: float | None
    utilisation_u: float | None
    blended_usd_per_m_output_tok_at_r_slo: float | None
    caveats: list[str] = Field(default_factory=list)


def cost_report(
    cfg: CostConfig, at_r_slo: OperatingPoint, peak: OperatingPoint | None = None
) -> CostReport:
    gpu_h = gpu_usd_per_hour(cfg)
    gpu_h8 = gpu_usd_per_hour(cfg, duty_hours_per_day=HOURS_PER_DAY_SENSITIVITY)
    pow_h = power_usd_per_hour(at_r_slo.p_avg_w, cfg)
    caveats: list[str] = [
        "Electricity covers GPU board power only (NVML); host power is unmeasured, so the "
        "power term is a lower bound (claim ceiling 6).",
    ]
    if cfg.is_placeholder:
        caveats.insert(0, "config/cost.yaml still holds owner-input placeholders; do not publish.")
    naive = naive_wh = u = None
    if peak is not None:
        naive = usd_per_million_output_tokens(
            gpu_h, power_usd_per_hour(peak.p_avg_w, cfg), peak.output_tok_per_s
        )
        naive_wh = wh_per_million_output_tokens(peak.p_avg_w, peak.output_tok_per_s)
        u = at_r_slo.output_tok_per_s / peak.output_tok_per_s
        caveats.append(
            f"Utilisation-naive $/M assumes peak throughput; at the driven utilisation "
            f"U = {u:.3f} the actual cost is approximately naive / U."
        )
    blended = None
    if cfg.input_output_price_ratio is not None and at_r_slo.input_tok_per_s is not None:
        blended = blended_usd_per_million_output_tokens(
            gpu_h,
            pow_h,
            at_r_slo.input_tok_per_s,
            at_r_slo.output_tok_per_s,
            cfg.input_output_price_ratio,
        )
    return CostReport(
        config_status=cfg.status,
        gpu_usd_per_h=gpu_h,
        gpu_usd_per_h_duty8=gpu_h8,
        power_usd_per_h_at_r_slo=pow_h,
        usd_per_m_output_tok_at_r_slo=usd_per_million_output_tokens(
            gpu_h, pow_h, at_r_slo.output_tok_per_s
        ),
        usd_per_m_output_tok_at_r_slo_duty8=usd_per_million_output_tokens(
            gpu_h8, pow_h, at_r_slo.output_tok_per_s
        ),
        wh_per_m_output_tok_at_r_slo=wh_per_million_output_tokens(
            at_r_slo.p_avg_w, at_r_slo.output_tok_per_s
        ),
        usd_per_m_output_tok_naive=naive,
        wh_per_m_output_tok_naive=naive_wh,
        utilisation_u=u,
        blended_usd_per_m_output_tok_at_r_slo=blended,
        caveats=caveats,
    )
