"""1 s NVML power sampler writing `power.csv` (design spec §6).

The spec proposed `nvidia-smi --query-gpu=...`; on the WSL2 host the driver rejects
`--query-gpu`, so the sampler reads NVML directly (same counters: power.draw, utilization.gpu,
memory.used, clocks.sm, temperature.gpu). Power limit and clocks are never touched; temperature
and SM clock are recorded so thermal throttling is visible in the evidence.

`run_sampler` takes an injected reader/clock/sleep so it is testable without a GPU. Phases
(idle / warmup / measure) are labelled per invocation and appended to one CSV; `cost.py` selects
windows by phase or time.
"""

from __future__ import annotations

import contextlib
import csv
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, TextIO

from slo_lab.cost import POWER_CSV_COLUMNS
from slo_lab.nvml import NvmlUnavailableError, init_nvml, load_pynvml

Phase = str


class Reading:
    __slots__ = ("clocks_sm_mhz", "mem_used_mib", "power_w", "temp_c", "util_gpu_pct")

    def __init__(
        self,
        power_w: float,
        util_gpu_pct: float | None = None,
        mem_used_mib: float | None = None,
        clocks_sm_mhz: float | None = None,
        temp_c: float | None = None,
    ) -> None:
        self.power_w = power_w
        self.util_gpu_pct = util_gpu_pct
        self.mem_used_mib = mem_used_mib
        self.clocks_sm_mhz = clocks_sm_mhz
        self.temp_c = temp_c


class GpuReader(Protocol):
    def read(self) -> Reading: ...


class NvmlReader:
    """Reads one GPU through pynvml. Construction fails fast with NvmlUnavailableError."""

    def __init__(self, index: int = 0) -> None:
        self._nvml = load_pynvml()
        init_nvml(self._nvml)
        try:
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            raise NvmlUnavailableError(f"cannot open GPU index {index}: {exc}") from exc

    def read(self) -> Reading:
        nv = self._nvml
        h = self._handle
        power_w = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
        util = nv.nvmlDeviceGetUtilizationRates(h).gpu
        mem = nv.nvmlDeviceGetMemoryInfo(h).used / (1024.0 * 1024.0)
        clocks = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
        temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        return Reading(power_w, float(util), mem, float(clocks), float(temp))

    def close(self) -> None:
        with contextlib.suppress(Exception):  # best effort
            self._nvml.nvmlShutdown()


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def run_sampler(
    reader: GpuReader,
    out: TextIO,
    *,
    interval_s: float = 1.0,
    max_samples: int | None = None,
    phase: Phase = "",
    stop: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wall: Callable[[], datetime] = lambda: datetime.now(UTC),
    write_header: bool = True,
    t0: float | None = None,
) -> int:
    """Sample until `max_samples` or `stop()`; returns the number of rows written.

    Rows are scheduled drift-free at t0 + k * interval_s. `t_s` is `clock() - t0`; by default
    t0 is the first sample of this invocation, so either pass a shared `t0` (monotonic seconds,
    system-wide on Linux and Windows) across phases or cut windows by the `phase` column.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")
    writer = csv.writer(out, lineterminator="\n")
    if write_header:
        writer.writerow(POWER_CSV_COLUMNS)
    if t0 is None:
        t0 = clock()
    written = 0
    while (max_samples is None or written < max_samples) and not (stop and stop()):
        now = clock()
        r = reader.read()
        writer.writerow(
            [
                wall().isoformat(timespec="milliseconds"),
                f"{now - t0:.3f}",
                _fmt(r.power_w),
                _fmt(r.util_gpu_pct),
                _fmt(r.mem_used_mib),
                _fmt(r.clocks_sm_mhz),
                _fmt(r.temp_c),
                phase,
            ]
        )
        out.flush()
        written += 1
        delay = interval_s - ((clock() - t0) % interval_s)
        if 0 < delay <= interval_s:
            sleep(delay)
    return written
