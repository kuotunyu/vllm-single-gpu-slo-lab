"""Quiet-GPU gate (design spec §9, re-plan §3): refuse to start a run unless the GPU is idle.

The 4090 is shared with a cron job that runs 00:00-08:00; this lab only measures afterwards.
Before vLLM starts, the harness calls `snapshot()` + `decide()`: any other compute process, or
pre-existing memory use above a threshold (proposal: 1024 MiB), blocks the run. The snapshot
(NVML device state, compute processes, best-effort `nvidia-smi` text) is written as JSON next to
the run's evidence so a reader can verify the card was quiet.

`decide` is pure so it is unit-tested with injected fake process lists; NVML is imported lazily.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from slo_lab.nvml import init_nvml, load_pynvml

DEFAULT_MEMORY_THRESHOLD_MIB = 1024.0  # proposal; frozen in preregistration at W1


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
) -> Decision:
    """Allow only when no foreign compute process exists and memory use is under the threshold."""
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
    return Decision(ok=not reasons, reasons=reasons)


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
        snap: dict[str, Any] = {
            "taken_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "gpu_index": index,
            "gpu_name": raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name),
            "driver_version": raw_driver.decode()
            if isinstance(raw_driver, bytes)
            else str(raw_driver),
            "memory_total_mib": mem.total / (1024.0 * 1024.0),
            "memory_used_mib": mem.used / (1024.0 * 1024.0),
            "compute_processes": [p.model_dump() for p in procs],
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
    return decide(procs, float(snap.get("memory_used_mib", 0.0)), **kwargs)


def write_snapshot(path: Path, snap: dict[str, Any], decision: Decision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**snap, "decision": decision.model_dump()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
