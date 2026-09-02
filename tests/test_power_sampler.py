import csv
import io
from datetime import UTC, datetime

import pytest

from slo_lab.cost import POWER_CSV_COLUMNS, integrate, read_power_csv, select_window
from slo_lab.nvml import NvmlUnavailableError
from slo_lab.power import sampler
from slo_lab.power.sampler import NvmlReader, Reading, run_sampler


class FakeReader:
    def __init__(self, readings):
        self._it = iter(readings)

    def read(self) -> Reading:
        return next(self._it)


class FakeClock:
    """Monotonic clock that only advances when the sampler sleeps."""

    def __init__(self) -> None:
        self.t = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def test_run_sampler_writes_header_rows_and_schedules_drift_free():
    readings = [
        Reading(100.0, 50, 1024, 2500, 60),
        Reading(110.0, 55, 1030, 2490, 61),
        Reading(105.0),
    ]
    clock = FakeClock()
    buf = io.StringIO()
    n = run_sampler(
        FakeReader(readings),
        buf,
        interval_s=1.0,
        max_samples=3,
        phase="idle",
        clock=clock,
        sleep=clock.sleep,
        wall=lambda: datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert n == 3
    rows = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert list(rows[0].keys()) == list(POWER_CSV_COLUMNS)
    assert [r["t_s"] for r in rows] == ["0.000", "1.000", "2.000"]
    assert [r["power_w"] for r in rows] == ["100", "110", "105"]
    assert rows[0]["util_gpu_pct"] == "50" and rows[2]["util_gpu_pct"] == ""
    assert {r["phase"] for r in rows} == {"idle"}
    assert rows[0]["timestamp_iso"].startswith("2026-09-03T00:00:00")
    assert clock.sleeps == [1.0, 1.0, 1.0]


def test_run_sampler_stop_callback_and_no_header_on_append():
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    buf = io.StringIO()
    n = run_sampler(
        FakeReader([Reading(1.0)] * 10),
        buf,
        interval_s=0.5,
        stop=stop,
        clock=FakeClock(),
        sleep=lambda s: None,
        write_header=False,
    )
    assert n == 2
    assert not buf.getvalue().startswith("timestamp_iso")


def test_sampler_output_round_trips_into_cost_integration(tmp_path):
    path = tmp_path / "power.csv"
    clock = FakeClock()
    with path.open("w", encoding="utf-8", newline="") as fh:
        run_sampler(
            FakeReader([Reading(20.0)] * 3),
            fh,
            max_samples=3,
            phase="idle",
            clock=clock,
            sleep=clock.sleep,
        )
        run_sampler(
            FakeReader([Reading(300.0)] * 5),
            fh,
            max_samples=5,
            phase="measure",
            clock=clock,
            sleep=clock.sleep,
            write_header=False,
        )
    rows = read_power_csv(path)
    assert len(rows) == 8
    measure = select_window(rows, phase="measure")
    assert integrate(measure).p_avg_w == pytest.approx(300.0)
    assert integrate(measure).duration_s == pytest.approx(4.0)


def test_run_sampler_rejects_bad_interval():
    with pytest.raises(ValueError):
        run_sampler(FakeReader([]), io.StringIO(), interval_s=0)


def test_nvml_reader_degrades_clearly_when_pynvml_missing(monkeypatch):
    def missing():
        raise NvmlUnavailableError("NVML is unavailable. Install the GPU extra")

    monkeypatch.setattr(sampler, "load_pynvml", missing)
    with pytest.raises(NvmlUnavailableError, match="Install the GPU extra"):
        NvmlReader()


def test_nvml_reader_degrades_clearly_when_nvml_init_fails(monkeypatch):
    class FakePynvml:
        @staticmethod
        def nvmlInit():
            raise RuntimeError("NVML Shared Library Not Found")

    monkeypatch.setattr(sampler, "load_pynvml", lambda: FakePynvml)
    with pytest.raises(NvmlUnavailableError, match="nvmlInit failed"):
        NvmlReader()


def test_nvml_reader_reads_through_fake_pynvml(monkeypatch):
    class Util:
        gpu = 42

    class Mem:
        used = 2 * 1024 * 1024

    class FakePynvml:
        NVML_CLOCK_SM = 1
        NVML_TEMPERATURE_GPU = 0

        @staticmethod
        def nvmlInit():
            pass

        @staticmethod
        def nvmlShutdown():
            pass

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):
            return f"h{i}"

        @staticmethod
        def nvmlDeviceGetPowerUsage(h):
            return 123456

        @staticmethod
        def nvmlDeviceGetUtilizationRates(h):
            return Util()

        @staticmethod
        def nvmlDeviceGetMemoryInfo(h):
            return Mem()

        @staticmethod
        def nvmlDeviceGetClockInfo(h, kind):
            return 2520

        @staticmethod
        def nvmlDeviceGetTemperature(h, kind):
            return 57

    monkeypatch.setattr(sampler, "load_pynvml", lambda: FakePynvml)
    reader = NvmlReader(0)
    r = reader.read()
    assert (r.power_w, r.util_gpu_pct, r.mem_used_mib, r.clocks_sm_mhz, r.temp_c) == (
        123.456,
        42.0,
        2.0,
        2520.0,
        57.0,
    )
    reader.close()
