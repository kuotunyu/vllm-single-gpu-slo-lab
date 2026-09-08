import json

from slo_lab.quiet_gpu import (
    DEFAULT_MEMORY_THRESHOLD_MIB,
    Decision,
    GpuProcess,
    decide,
    decide_from_snapshot,
    process_list_trustworthy,
    write_snapshot,
)


def test_quiet_gpu_allows_when_idle():
    assert decide([], memory_used_mib=300.0) == Decision(ok=True, reasons=[])


def test_quiet_gpu_refuses_on_any_foreign_compute_process():
    procs = [GpuProcess(pid=4242, name="python", used_memory_mib=512.0)]
    d = decide(procs, memory_used_mib=600.0)
    assert not d.ok
    assert d.reasons == ["compute process present: pid=4242 name=python mem=512 MiB"]


def test_quiet_gpu_refuses_on_memory_above_threshold_even_without_processes():
    d = decide([], memory_used_mib=DEFAULT_MEMORY_THRESHOLD_MIB + 1)
    assert not d.ok
    assert d.reasons[0].startswith(
        f"GPU memory already in use: {DEFAULT_MEMORY_THRESHOLD_MIB + 1:.0f} MiB > threshold "
        f"{DEFAULT_MEMORY_THRESHOLD_MIB:.0f} MiB"
    )
    assert decide([], memory_used_mib=DEFAULT_MEMORY_THRESHOLD_MIB).ok


def test_quiet_gpu_refuses_on_utilization_even_when_process_list_is_empty():
    # WSL2 never lists compute processes, so a busy card must still be caught by utilization.
    d = decide([], memory_used_mib=300.0, utilization_percent=42.0)
    assert not d.ok
    assert d.reasons == ["GPU busy: utilization 42% > threshold 10%"]
    assert decide([], memory_used_mib=300.0, utilization_percent=9.0).ok
    assert decide([], memory_used_mib=300.0, utilization_percent=None).ok


def test_decide_from_snapshot_uses_utilization_and_records_trust_flag():
    snap = {"memory_used_mib": 100.0, "compute_processes": [], "utilization_percent": 30}
    assert not decide_from_snapshot(snap).ok
    snap["utilization_percent"] = 0
    assert decide_from_snapshot(snap).ok
    assert isinstance(process_list_trustworthy(), bool)


def test_quiet_gpu_allow_list_and_custom_threshold():
    procs = [GpuProcess(pid=1, name="vllm"), GpuProcess(pid=2, name=None, used_memory_mib=None)]
    d = decide(procs, memory_used_mib=5000.0, memory_threshold_mib=8000.0, allow_pids={1})
    assert not d.ok
    assert d.reasons == ["compute process present: pid=2 name=? mem=?"]
    assert decide(procs, memory_used_mib=5000.0, memory_threshold_mib=8000.0, allow_pids={1, 2}).ok


def test_decide_from_snapshot_and_write_snapshot(tmp_path):
    snap = {
        "gpu_name": "fake",
        "memory_used_mib": 100.0,
        "compute_processes": [{"pid": 7, "name": "x", "used_memory_mib": 50.0}],
    }
    d = decide_from_snapshot(snap)
    assert not d.ok and "pid=7" in d.reasons[0]
    out = tmp_path / "evidence" / "quiet_gpu.json"
    write_snapshot(out, snap, d)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["decision"] == {"ok": False, "reasons": d.reasons}
    assert payload["gpu_name"] == "fake"
