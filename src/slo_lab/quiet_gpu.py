"""Quiet-GPU gate (design spec §9, re-plan §3): refuse to start a run unless the GPU is idle.

The 4090 is shared with a cron job that runs 00:00-08:00; this lab only measures afterwards.
Before vLLM starts, the harness calls `snapshot()` + `decide()`: any other compute process,
pre-existing memory use above a threshold, or GPU utilization above a threshold blocks the run.
The snapshot (NVML device state, compute processes, utilization, best-effort `nvidia-smi` text)
is written as JSON next to the run's evidence so a reader can verify the card was quiet.

W1 finding (2026-09-08, WSL2 kernel 6.6.114, driver 591.86): NVML's compute-process list is
always empty under WSL2 even while vLLM holds 23 GiB, so the process criterion cannot detect a
foreign job there. The memory and utilization criteria are the effective gate on that host; the
snapshot records `process_list_trustworthy` so the evidence says which criteria actually applied.
The idle baseline on that host is about 2,600 MiB because Windows itself holds VRAM, hence the
default threshold below.

`decide` is pure so it is unit-tested with injected fake values; NVML is imported lazily.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import time
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from slo_lab.nvml import init_nvml, load_pynvml

DEFAULT_MEMORY_THRESHOLD_MIB = 3072.0  # W1: idle 2,609 MiB measured on the WSL2 host, 2026-09-08
DEFAULT_UTILIZATION_THRESHOLD_PERCENT = (
    10.0  # W2: idle host samples 0-9% (Windows desktop shares the card); mean of 5 x 1 s
)
UTILIZATION_SAMPLES = 5
UTILIZATION_INTERVAL_S = 1.0


class GpuProcess(BaseModel):
    pid: int
    name: str | None = None
    used_memory_mib: float | None = None


class Decision(BaseModel):
    ok: bool
    reasons: list[str] = Field(default_factory=list)


def decide(
    processes: Sequence[GpuProcess],
    memory_used_mib: float,
    *,
    memory_threshold_mib: float = DEFAULT_MEMORY_THRESHOLD_MIB,
    allow_pids: Collection[int] = (),
    utilization_percent: float | None = None,
    utilization_threshold_percent: float = DEFAULT_UTILIZATION_THRESHOLD_PERCENT,
) -> Decision:
    """Allow only when no foreign compute process exists, memory use and utilization are low.

    ``utilization_percent`` is optional so callers without NVML utilization data keep the older
    two-criterion behaviour; when given, it is the criterion that still works on WSL2.
    """
    reasons: list[str] = []
    for proc in processes:
        if proc.pid in allow_pids:
            continue
        mem = "?" if proc.used_memory_mib is None else f"{proc.used_memory_mib:.0f} MiB"
        reasons.append(f"compute process present: pid={proc.pid} name={proc.name or '?'} mem={mem}")
    if memory_used_mib > memory_threshold_mib:
        reasons.append(
            f"GPU memory already in use: {memory_used_mib:.0f} MiB > threshold "
            f"{memory_threshold_mib:.0f} MiB"
        )
    if utilization_percent is not None and utilization_percent > utilization_threshold_percent:
        reasons.append(
            f"GPU busy: utilization {utilization_percent:.0f}% > threshold "
            f"{utilization_threshold_percent:.0f}%"
        )
    return Decision(ok=not reasons, reasons=reasons)


def process_list_trustworthy() -> bool:
    """False on WSL2, where NVML never lists compute processes (W1 finding, 2026-09-08)."""
    try:
        with open("/proc/version", encoding="utf-8") as handle:
            return "microsoft" not in handle.read().lower()
    except OSError:
        return True


def _run_nvidia_smi(args: list[str]) -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"args": args, "available": False}
    try:
        proc = subprocess.run([exe, *args], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"args": args, "available": True, "error": str(exc)}
    return {
        "args": args,
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def snapshot(*, index: int = 0, include_nvidia_smi: bool = True) -> dict[str, Any]:
    """Collect NVML state for one GPU. Raises NvmlUnavailableError when NVML is absent."""
    nv = load_pynvml()
    init_nvml(nv)
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        mem = nv.nvmlDeviceGetMemoryInfo(handle)
        procs: list[GpuProcess] = []
        for p in nv.nvmlDeviceGetComputeRunningProcesses(handle):
            name: str | None
            try:
                raw = nv.nvmlSystemGetProcessName(p.pid)
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
            except Exception:
                name = None
            used = getattr(p, "usedGpuMemory", None)
            procs.append(
                GpuProcess(
                    pid=int(p.pid),
                    name=name,
                    used_memory_mib=None if used is None else used / (1024.0 * 1024.0),
                )
            )
        raw_name = nv.nvmlDeviceGetName(handle)
        raw_driver = nv.nvmlSystemGetDriverVersion()
        # A single utilization read is too noisy on a card shared with the Windows desktop
        # (W2 smoke: one 9% blip refused an idle GPU); average a short burst of samples.
        utilization_samples: list[float] = []
        for i in range(UTILIZATION_SAMPLES):
            try:
                utilization_samples.append(float(nv.nvmlDeviceGetUtilizationRates(handle).gpu))
            except Exception:
                break
            if i < UTILIZATION_SAMPLES - 1:
                time.sleep(UTILIZATION_INTERVAL_S)
        utilization = (
            sum(utilization_samples) / len(utilization_samples) if utilization_samples else None
        )
        snap: dict[str, Any] = {
            "taken_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "gpu_index": index,
            "gpu_name": raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name),
            "driver_version": raw_driver.decode()
            if isinstance(raw_driver, bytes)
            else str(raw_driver),
            "memory_total_mib": mem.total / (1024.0 * 1024.0),
            "memory_used_mib": mem.used / (1024.0 * 1024.0),
            "utilization_percent": utilization,
            "utilization_samples_percent": utilization_samples,
            "compute_processes": [p.model_dump() for p in procs],
            "process_list_trustworthy": process_list_trustworthy(),
        }
    finally:
        with contextlib.suppress(Exception):  # best effort
            nv.nvmlShutdown()
    if include_nvidia_smi:
        snap["nvidia_smi"] = _run_nvidia_smi([])
        snap["nvidia_smi_compute_apps"] = _run_nvidia_smi(
            ["--query-compute-apps=pid,process_name,used_memory", "--format=csv"]
        )
    return snap


def decide_from_snapshot(snap: dict[str, Any], **kwargs: Any) -> Decision:
    procs = [GpuProcess.model_validate(p) for p in snap.get("compute_processes", [])]
    utilization = snap.get("utilization_percent")
    return decide(
        procs,
        float(snap.get("memory_used_mib", 0.0)),
        utilization_percent=None if utilization is None else float(utilization),
        **kwargs,
    )


def write_snapshot(path: Path, snap: dict[str, Any], decision: Decision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**snap, "decision": decision.model_dump()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
