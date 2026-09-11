"""Run one measured stage against an already-running vLLM server (spec §3.3, §3.4, §8 W2).

A stage is (cell, load point, seed). The server lifecycle and the quiet-GPU gate belong to the
batch driver, because the gate must run before the server takes the card. This module does the
part that has to be reproducible per stage:

1. warm-up: ``warmup_requests`` sequential streaming completions, TTFT of each kept;
2. power sampling (NVML, 1 s) and ``/metrics`` scraping (5 s) in background threads;
3. inference-perf with a generated config (open-loop Poisson or closed-loop concurrency);
4. adaptation to ``records.jsonl``, a summary over the measurement window, and a
   ``manifest.json`` binding versions, flags, environment switches and raw-file digests.

The bulky inference-perf outputs stay under the run directory (ignored by git); the adapted
records, summary, power CSV, metrics CSV and manifest are what get promoted to evidence.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import threading
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slo_lab.admission_analysis import stage_metrics, waiting_series
from slo_lab.harness.ipf_adapter import adapt_file_with_origin, write_records_jsonl
from slo_lab.harness.ipf_config import (
    closed_loop_config,
    open_loop_config,
    trace_replay_config,
    write_config,
)
from slo_lab.harness.metrics_scraper import (
    HISTOGRAMS,
    SPEC_DECODE_PER_POS,
    MetricsScraper,
    fetch_metrics,
    histogram_delta,
    histogram_quantile_bounds,
    parse_histograms,
    parse_labelled,
    parse_metrics,
)
from slo_lab.harness.shim_scraper import ShimScraper
from slo_lab.harness.trace import load_profile, phase_bounds
from slo_lab.slo import DEFAULT_SLO, RequestRecord, filter_window, summarise
from slo_lab.stats import percentile


@dataclass
class WarmupResult:
    n: int
    ttft_s: list[float]
    median_1_100: float | None
    median_101_200: float | None
    median_201_300: float | None
    wall_s: float
    tpot_s: list[float] | None = None
    tpot_median_s: float | None = None


def _stream_ttft(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: float = 120.0
) -> float:
    """One streaming completion; returns TTFT in seconds (first SSE data line)."""
    return _stream_request(base_url, model, prompt, max_tokens, timeout_s)[0]


def _stream_request(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: float = 120.0
) -> tuple[float, float, int]:
    """One streaming completion; returns (TTFT, e2e, data chunks) in seconds / count."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": True,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    ttft: float | None = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("data:") and "[DONE]" not in line:
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
    e2e = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError("stream ended without a data chunk")
    return ttft, e2e, chunks


def warm_up(base_url: str, model: str, *, n: int = 100, output_tokens: int = 132) -> WarmupResult:
    """``n`` sequential requests with distinct nonce prompts; TTFT and TPOT per request are kept.

    The TPOT of these single-stream requests doubles as a tenancy probe: on an otherwise idle
    card it is a stable per-host constant (18-19 ms for Qwen3-8B-FP8 on the 4090), so a re-warm
    probe that drifts by more than ~15% says another GPU tenant appeared (W2, 2026-09-09).
    """
    started = time.perf_counter()
    ttfts: list[float] = []
    tpots: list[float] = []
    for i in range(n):
        prompt = (
            f"warmup nonce {i:05d} {hashlib.sha256(str(i).encode()).hexdigest()[:16]} "
            + "the quick brown fox " * 20
        )
        ttft, e2e, chunks = _stream_request(base_url, model, prompt, output_tokens)
        ttfts.append(ttft)
        if chunks > 1:
            tpots.append((e2e - ttft) / (chunks - 1))

    def med(lo: int, hi: int) -> float | None:
        block = ttfts[lo:hi]
        return float(percentile(block, 50)) if len(block) >= 10 else None

    return WarmupResult(
        n=n,
        ttft_s=ttfts,
        median_1_100=med(0, 100),
        median_101_200=med(100, 200),
        median_201_300=med(200, 300),
        wall_s=time.perf_counter() - started,
        tpot_s=tpots,
        tpot_median_s=float(percentile(tpots, 50)) if tpots else None,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _io_pressure() -> str | None:
    try:
        return Path("/proc/pressure/io").read_text(encoding="utf-8").splitlines()[0]
    except OSError:
        return None


WINDOWS_POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
_WINDOWS_GPU_MEMORY_COMMAND = (
    "$c=(Get-Counter -Counter '\\GPU Adapter Memory(*)\\Dedicated Usage',"
    "'\\GPU Adapter Memory(*)\\Shared Usage','\\GPU Adapter Memory(*)\\Total Committed' "
    "-ErrorAction Stop).CounterSamples;"
    "$d=($c|?{$_.Path -like '*dedicated usage'}|measure CookedValue -Sum).Sum;"
    "$s=($c|?{$_.Path -like '*shared usage'}|measure CookedValue -Sum).Sum;"
    "$t=($c|?{$_.Path -like '*total committed'}|measure CookedValue -Sum).Sum;"
    "'dedicated_mb={0:F0} shared_mb={1:F0} committed_mb={2:F0}' -f ($d/1MB),($s/1MB),($t/1MB)"
)


def parse_windows_gpu_memory(text: str) -> dict[str, float] | None:
    """``dedicated_mb=... shared_mb=... committed_mb=...`` -> dict; None when absent."""
    found = dict(re.findall(r"(dedicated_mb|shared_mb|committed_mb)=(-?\d+(?:\.\d+)?)", text))
    if len(found) != 3:
        return None
    return {key: float(value) for key, value in found.items()}


def windows_gpu_memory(
    powershell: Path = WINDOWS_POWERSHELL, timeout_s: float = 20.0
) -> dict[str, float] | None:
    """Host-side VRAM accounting through WSL interop (best effort, None off WSL2).

    NVML inside WSL2 sees only the guest; the WDDM clients on the desktop (compositor,
    browsers) are invisible to it. When their allocations plus vLLM's budget push
    ``committed_mb`` past the physical card, VidMm pages VRAM over PCIe and vLLM's step time
    doubles while utilization reads ~97% (ADR 0007, 2026-09-09).
    """
    if not powershell.exists():
        return None
    try:
        proc = subprocess.run(
            [
                str(powershell),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _WINDOWS_GPU_MEMORY_COMMAND,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_windows_gpu_memory(proc.stdout)


def _host_state() -> dict[str, Any]:
    """Load average, CPU pressure and host-side VRAM accounting (tenancy evidence)."""
    state: dict[str, Any] = {"loadavg": None, "cpu_pressure": None, "windows_gpu_memory": None}
    with contextlib.suppress(OSError, ValueError):
        fields = Path("/proc/loadavg").read_text(encoding="utf-8").split()[:3]
        state["loadavg"] = [float(x) for x in fields]
    with contextlib.suppress(OSError):
        text = Path("/proc/pressure/cpu").read_text(encoding="utf-8")
        state["cpu_pressure"] = text.splitlines()[0]
    state["windows_gpu_memory"] = windows_gpu_memory()
    return state


def proc_cpu_s(pid: int | None) -> float | None:
    """utime + stime of a process in seconds from ``/proc/<pid>/stat``; None when unavailable.

    The shim is one asyncio process; if it saturated a core during a burst its own latency
    would be part of every policy's numbers, so the trace manifest records its CPU time.
    """
    if pid is None:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def clock_offset_s() -> float:
    """Wall clock minus the monotonic clock inference-perf stamps requests with."""
    return time.time() - time.monotonic()


SPEC_DECODE_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def counter_delta(
    before: dict[str, float] | None, after: dict[str, float] | None, name: str
) -> float | None:
    """``after - before`` of one cumulative counter; None when either snapshot lacks it."""
    if not before or not after or name not in before or name not in after:
        return None
    return float(after[name]) - float(before[name])


def spec_decode_summary(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
    per_pos_before: dict[str, float] | None = None,
    per_pos_after: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Draft acceptance over a stage from the server's spec-decode counters (W4, ADR 0015).

    vLLM 0.28 counts drafts, draft tokens and accepted tokens (plus accepted tokens per draft
    position); the acceptance rate is accepted / draft tokens and the mean acceptance length is
    1 + accepted / drafts, both as vLLM's own logger computes them. None when the server exposes
    no such counters (no speculative decoding); rates None when the stage drafted nothing.
    """
    deltas = {name: counter_delta(before, after, name) for name in SPEC_DECODE_COUNTERS}
    if any(v is None for v in deltas.values()):
        return None
    drafts, draft_tokens, accepted = (deltas[name] for name in SPEC_DECODE_COUNTERS)
    positions = sorted(
        {*(per_pos_before or {}), *(per_pos_after or {})},
        key=lambda p: (not p.isdigit(), int(p) if p.isdigit() else 0, p),
    )
    accepted_per_pos = [
        float((per_pos_after or {}).get(p, 0.0)) - float((per_pos_before or {}).get(p, 0.0))
        for p in positions
    ]
    return {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "acceptance_rate": round(accepted / draft_tokens, 6) if draft_tokens > 0 else None,
        "mean_acceptance_length": round(1 + accepted / drafts, 6) if drafts > 0 else None,
        "accepted_per_pos": accepted_per_pos,
        "acceptance_per_pos": [
            round(a / drafts, 6) if drafts > 0 else None for a in accepted_per_pos
        ],
    }


def trace_stage_summary(
    run_dir: Path,
    records: Sequence[RequestRecord],
    *,
    origin_monotonic_s: float,
    clock_offsets: Sequence[float],
    phases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Per-phase summaries and recovery times, with scraped queues put on the records' axis."""
    origin_unix = origin_monotonic_s + sum(clock_offsets) / len(clock_offsets)
    waiting = waiting_series(run_dir, origin_mono=origin_monotonic_s, origin_unix=origin_unix)
    sm = stage_metrics(records, phases, waiting or None)
    return {
        "phase_summaries": {k: v for k, v in sm.items() if isinstance(v, dict)},
        "time_to_recover_s": sm["time_to_recover_s"],
        "time_to_recover_attainment_s": sm["time_to_recover_attainment_s"],
        "waiting_samples": len(waiting),
    }


def stage_window(
    records: Sequence[RequestRecord], *, kind: str, discard_first_s: float, duration_s: int
) -> tuple[list[RequestRecord], float | None]:
    """Measurement window of a stage: records offered after the discard period, and its length.

    Open-loop stages have a fixed duration, so the window is ``duration_s - discard_first_s``.
    Closed-loop stages run until ``num_requests`` complete, so the window runs from the discard
    point to the last completion (``offered_at + e2e``). A closed-loop stage shorter than the
    discard period yields an empty window instead of silently reporting its start-up transient
    (the exploratory FP8 sweep of 2026-09-09 had no discard and 60-100 s stages; ADR 0006).
    """
    if kind == "trace":
        # an admission trace is measured whole: its phases are summarised separately
        return list(records), float(duration_s)
    window = filter_window(records, start_s=discard_first_s)
    if kind == "open_loop":
        return window, float(duration_s - discard_first_s)
    if not window:
        return window, None
    t_end = max(r.offered_at_s + (r.e2e_s or 0.0) for r in records)
    window_s = t_end - discard_first_s
    return window, (window_s if window_s > 0 else None)


def _server_histograms(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Server-side latency histograms accrued during the stage (loadgen-independent view)."""
    if before is None or after is None:
        return None

    def bounds(delta: dict[str, Any], q: float) -> list[float | None]:
        lo, hi = histogram_quantile_bounds(delta, q)
        return [lo, None if math.isinf(hi) else hi]

    out: dict[str, Any] = {}
    for name in HISTOGRAMS:
        delta = histogram_delta(before[name], after[name])
        out[name.removeprefix("vllm:")] = {
            "count": delta["count"],
            "sum": delta["sum"],
            "p50_bounds_s": bounds(delta, 0.50),
            "p95_bounds_s": bounds(delta, 0.95),
            "p99_bounds_s": bounds(delta, 0.99),
            "buckets": delta["buckets"],
        }
    return out


def _power_window(path: Path, *, start_s: float, output_tokens: int) -> dict[str, Any] | None:
    """Mean draw and energy over ``power.csv`` rows with ``t_s >= start_s``.

    The sampler starts a few seconds before inference-perf, so the cut is aligned with the
    record window to within the loadgen's own start-up time.
    """
    if not path.exists():
        return None
    rows: list[tuple[float, float, dict[str, str]]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                t, p = float(row["t_s"]), float(row["power_w"])
            except (KeyError, ValueError):
                continue
            if t >= start_s:
                rows.append((t, p, row))
    if len(rows) < 2:
        return None
    span = rows[-1][0] - rows[0][0]
    mean_w = sum(p for _, p, _ in rows) / len(rows)
    wh = mean_w * span / 3600.0
    temps = [float(r["temp_c"]) for _, _, r in rows if r.get("temp_c")]
    clocks = [float(r["clocks_sm_mhz"]) for _, _, r in rows if r.get("clocks_sm_mhz")]
    utils = [float(r["util_gpu_pct"]) for _, _, r in rows if r.get("util_gpu_pct")]
    mean_util = sum(utils) / len(utils) if utils else None
    return {
        "samples": len(rows),
        "span_s": round(span, 1),
        "mean_w": round(mean_w, 1),
        "max_w": round(max(p for _, p, _ in rows), 1),
        "wh": round(wh, 4),
        "output_tok_per_wh": round(output_tokens / wh, 1) if wh > 0 else None,
        "mean_temp_c": round(sum(temps) / len(temps), 1) if temps else None,
        "max_temp_c": max(temps) if temps else None,
        "mean_sm_mhz": round(sum(clocks) / len(clocks)) if clocks else None,
        "mean_util_pct": round(mean_util, 1) if mean_util is not None else None,
        # Tenancy signature: a second GPU context time-slicing the card keeps NVML utilization
        # near 100% while vLLM's own work (and hence power) halves. Clean stages on the 4090
        # sit at >= 2.3 W per utilization point and rise with concurrency; contaminated stages
        # of 2026-09-09 sat at 1.8 (ADR 0006).
        "w_per_util_point": round(mean_w / mean_util, 2) if mean_util else None,
    }


class _PowerThread:
    """Wrap ``run_sampler`` with the NVML reader in a thread; degrades to a note without NVML."""

    def __init__(self, out_path: Path, phase: str) -> None:
        self.out_path = out_path
        self.phase = phase
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.note: str | None = None
        self.rows = 0

    def _run(self) -> None:
        try:
            from slo_lab.power.sampler import NvmlReader, run_sampler
        except Exception as exc:
            self.note = f"power sampler unavailable: {exc}"
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            reader = NvmlReader()
        except Exception as exc:
            self.note = f"NVML unavailable: {exc}"
            return
        with self.out_path.open("w", encoding="utf-8", newline="") as handle:
            self.rows = run_sampler(
                reader, handle, interval_s=1.0, phase=self.phase, stop=self._stop.is_set
            )

    def start(self) -> _PowerThread:
        self._thread = threading.Thread(target=self._run, name="power-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)


def run_stage(
    *,
    run_dir: Path,
    cell: str,
    model: str,
    base_url: str,
    metrics_url: str,
    seed: int,
    kind: str,
    rate_rps: float | None = None,
    duration_s: int = 300,
    concurrency: int | None = None,
    num_requests: int | None = None,
    warmup_requests: int = 100,
    discard_first_s: float = 60.0,
    inference_perf_bin: str = "inference-perf",
    engine_flags: dict[str, Any] | None = None,
    workers: int = 4,
    trace_file: Path | None = None,
    profile_path: Path | None = None,
    policy: str | None = None,
    shim_stats_url: str | None = None,
    shim_pid: int | None = None,
    specdec: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    io_before = _io_pressure()

    # admission trace (W3, ADR 0012): one continuous replay through the shim, measured whole
    trace_info: dict[str, Any] | None = None
    if kind == "trace":
        if trace_file is None or profile_path is None:
            raise ValueError("a trace stage needs trace_file and profile_path")
        profile = load_profile(profile_path)
        phases = [
            {"phase": name, "start_s": lo, "end_s": hi} for name, lo, hi in phase_bounds(profile)
        ]
        duration_s = int(sum(d for _, d in profile))
        discard_first_s = 0.0
        arrivals = sum(
            1 for line in trace_file.read_text(encoding="utf-8").splitlines()[1:] if line
        )
        trace_info = {
            "file": trace_file.name,
            "sha256": _sha256(trace_file),
            "arrivals": arrivals,
            "profile": profile_path.name,
            "phases": phases,
        }

    # 1. warm-up (its own power phase so the measurement window is clean)
    power_warm = _PowerThread(run_dir / "power-warmup.csv", "warmup").start()
    warm = warm_up(base_url, model, n=warmup_requests) if warmup_requests > 0 else None
    power_warm.stop()

    # 2. background sampling for the measured window
    power = _PowerThread(run_dir / "power.csv", "measure").start()
    scraper = MetricsScraper(
        metrics_url, run_dir / "metrics.csv", interval_s=5.0, fetch=fetch_metrics
    ).start()
    metrics_before = None
    hist_before = None
    pos_before: dict[str, float] | None = None
    try:
        text = fetch_metrics(metrics_url)
        metrics_before = parse_metrics(text)
        hist_before = parse_histograms(text)
        pos_before = parse_labelled(text, SPEC_DECODE_PER_POS, "position")
    except Exception:
        pass
    host_before = _host_state()
    shim_scraper = (
        ShimScraper(shim_stats_url, run_dir / "shim.csv", interval_s=5.0).start()
        if shim_stats_url
        else None
    )
    shim_cpu_before = proc_cpu_s(shim_pid)
    offset_before = clock_offset_s()

    # 3. inference-perf
    report_dir = run_dir / "ipf"
    if kind == "trace":
        assert trace_file is not None and trace_info is not None
        cfg = trace_replay_config(
            model=model,
            base_url=base_url,
            report_dir=str(report_dir),
            trace_file=str(trace_file),
            duration_s=duration_s,
            mean_rate_rps=trace_info["arrivals"] / duration_s,
            seed=seed,
            workers=workers,
        )
    elif kind == "open_loop":
        assert rate_rps is not None
        cfg = open_loop_config(
            model=model,
            base_url=base_url,
            report_dir=str(report_dir),
            rate_rps=rate_rps,
            duration_s=duration_s,
            seed=seed,
            workers=workers,
        )
    elif kind == "closed_loop":
        assert concurrency is not None and num_requests is not None
        cfg = closed_loop_config(
            model=model,
            base_url=base_url,
            report_dir=str(report_dir),
            concurrency=concurrency,
            num_requests=num_requests,
            seed=seed,
            workers=workers,
        )
    else:
        raise ValueError(f"unknown stage kind {kind}")
    cfg_path = write_config(cfg, run_dir / "inference-perf.yaml")
    t_load_start = time.time()
    t_load_start_mono = time.monotonic()
    proc = subprocess.run(
        [inference_perf_bin, "-c", str(cfg_path), "--log-level", "INFO"],
        capture_output=True,
        text=True,
        timeout=duration_s + 1800,
        check=False,
    )
    t_load_end = time.time()
    (run_dir / "inference-perf.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8"
    )

    offset_after = clock_offset_s()
    shim_cpu_after = proc_cpu_s(shim_pid)
    if shim_scraper is not None:
        shim_scraper.stop()
    scraper.stop()
    power.stop()
    metrics_after = None
    hist_after = None
    pos_after: dict[str, float] | None = None
    try:
        text = fetch_metrics(metrics_url)
        metrics_after = parse_metrics(text)
        hist_after = parse_histograms(text)
        pos_after = parse_labelled(text, SPEC_DECODE_PER_POS, "position")
    except Exception:
        pass
    host_after = _host_state()

    # 4. adapt + summarise
    per_request = report_dir / "per_request_lifecycle_metrics.json"
    result: dict[str, Any] = {
        "cell": cell,
        "model": model,
        "seed": seed,
        "kind": kind,
        "rate_rps": rate_rps,
        "duration_s": duration_s,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "started_at": started_at,
        "inference_perf_returncode": proc.returncode,
        "load_wall_s": round(t_load_end - t_load_start, 2),
        "warmup": asdict(warm) | {"ttft_s": None, "tpot_s": None} if warm else None,
        "warmup_ttft_median_s": (float(percentile(warm.ttft_s, 50)) if warm else None),
        "probe_tpot_median_s": warm.tpot_median_s if warm else None,
        "io_pressure_before": io_before,
        "io_pressure_after": _io_pressure(),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "server_histograms": _server_histograms(hist_before, hist_after),
        "host_before": host_before,
        "host_after": host_after,
        "discard_first_s": discard_first_s,
        "power_rows": power.rows,
        "power_note": power.note,
        "metrics_rows": scraper.rows,
        "engine_flags": engine_flags or {},
        "env": {
            k: os.environ.get(k)
            for k in (
                "VLLM_WSL2_ENABLE_PIN_MEMORY",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "HF_HUB_OFFLINE",
            )
        },
        "platform": platform.platform(),
        # W4 (ADR 0015 item 6): draft acceptance over the stage from the server's spec-decode
        # counters (None on a server without speculative decoding) and the engine's preemptions
        "spec_decode": spec_decode_summary(metrics_before, metrics_after, pos_before, pos_after),
        "preemptions": counter_delta(metrics_before, metrics_after, "vllm:num_preemptions_total"),
    }
    if specdec is not None:
        result["specdec"] = specdec
        result["family"] = family
    if trace_info is not None or shim_scraper is not None or shim_pid is not None:
        # the shim's own counters and CPU time, for every stage kind that ran through it
        cpu = (
            round(shim_cpu_after - shim_cpu_before, 2)
            if shim_cpu_before is not None and shim_cpu_after is not None
            else None
        )
        result.update(
            {
                "shim_cpu_s": cpu,
                "shim_rows": shim_scraper.rows if shim_scraper else 0,
                "shim_final": shim_scraper.last if shim_scraper else None,
            }
        )
    if trace_info is not None:
        result.update(
            {
                "policy": policy,
                "trace": trace_info,
                "clock_offset_s": {"before": offset_before, "after": offset_after},
            }
        )
    if warm:
        (run_dir / "warmup-ttft.json").write_text(
            json.dumps({"ttft_s": warm.ttft_s, "tpot_s": warm.tpot_s}, indent=0) + "\n",
            encoding="utf-8",
        )
    if per_request.exists():
        records, origin_monotonic = adapt_file_with_origin(per_request)
        write_records_jsonl(records, run_dir / "records.jsonl")
        if trace_info is not None:
            offsets = [offset_before, offset_after]
            result["records_origin_monotonic_s"] = origin_monotonic
            # sanity of the clock alignment, on the monotonic clock only: the first request
            # leaves a few seconds after launch (tokenizer load, trace read, worker start-up)
            result["first_request_after_launch_s"] = round(origin_monotonic - t_load_start_mono, 3)
            result["wall_clock_drift_s"] = round(offset_after - offset_before, 3)
            result.update(
                trace_stage_summary(
                    run_dir,
                    records,
                    origin_monotonic_s=origin_monotonic,
                    clock_offsets=offsets,
                    phases=trace_info["phases"],
                )
            )
        window, window_s = stage_window(
            records, kind=kind, discard_first_s=discard_first_s, duration_s=duration_s
        )
        summary = summarise(window, DEFAULT_SLO, window_s=window_s) if window else None
        ok = [r for r in window if r.ttft_s is not None and r.e2e_s is not None]
        ttfts = [r.ttft_s for r in ok]  # type: ignore[misc]
        tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
        result.update(
            {
                "records": len(records),
                "window_records": len(window),
                "window_s": window_s,
                "summary": summary.model_dump() if summary else None,
                "power_window": _power_window(
                    run_dir / "power.csv",
                    start_s=discard_first_s,
                    output_tokens=sum(r.output_tokens or 0 for r in ok),
                ),
                "ttft_p50_s": float(percentile(ttfts, 50)) if ttfts else None,
                "ttft_p95_s": float(percentile(ttfts, 95)) if ttfts else None,
                "tpot_p50_s": float(percentile(tpots, 50)) if tpots else None,
                "tpot_p95_s": float(percentile(tpots, 95)) if tpots else None,
                "achieved_rps": (len(window) / window_s) if window_s else None,
                "output_tok_per_s": (sum(r.output_tokens or 0 for r in ok) / window_s)
                if window_s
                else None,
                "raw_sha256": {"per_request_lifecycle_metrics.json": _sha256(per_request)},
            }
        )
    else:
        result.update(
            {
                "records": 0,
                "error": "inference-perf produced no per-request file",
                "stderr_tail": proc.stderr[-800:],
            }
        )
    (run_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return result
