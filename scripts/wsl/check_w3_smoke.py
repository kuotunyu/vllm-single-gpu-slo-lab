"""Gate the W3 night on its GPU smoke (plan 2026-09-11 Task B1); exit 1 on a functional failure.

Functional checks fail the gate: inference-perf ran, every trace row became a record, the phase
summaries and both scrapers are present, the clock alignment is plausible, native queueing
rejected nothing, the bounded queue rejected and those rejections reached the records as 429s,
and the shim did not saturate a core. Latency against W2's direct measurement is printed as a
warning only, because the smoke's pre-burst phase is 60 s long and starts from an empty server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# FP8 open-loop at 21.84 rps without the shim, worst of three seeds (analysis/tables/w2-fp8-open-loop)
W2_TPOT_P95_S = 0.0246
W2_TTFT_P95_S = 0.0896
PHASES = {"all", "pre", "burst", "recovery"}


def check(seed_dir: Path) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    notes: list[str] = []
    manifests: dict[str, dict] = {}
    for policy in ("direct", "passthrough", "bounded_queue"):
        path = seed_dir / f"trace-{policy}" / "manifest.json"
        if not path.exists():
            fails.append(f"{policy}: no manifest")
            continue
        manifests[policy] = json.loads(path.read_text(encoding="utf-8"))
    for policy, m in manifests.items():
        rc = m.get("inference_perf_returncode")
        records = m.get("records") or 0
        arrivals = (m.get("trace") or {}).get("arrivals") or 0
        phases = m.get("phase_summaries") or {}
        if rc != 0:
            fails.append(f"{policy}: inference-perf exit code {rc}")
        if not arrivals or abs(records - arrivals) > max(2, arrivals // 1000):
            fails.append(f"{policy}: {records} records for {arrivals} trace rows")
        if not set(phases) >= PHASES:
            fails.append(f"{policy}: phase summaries {sorted(phases)}")
        lag = m.get("first_request_after_launch_s")
        if lag is None or not 0 < lag < 180:
            fails.append(f"{policy}: first request {lag} s after launch (clock alignment)")
        if policy != "direct" and not m.get("shim_rows"):
            fails.append(f"{policy}: shim.csv has no rows")
        errors = (m.get("summary") or {}).get("errors") or 0
        upstream_errors = (m.get("shim_final") or {}).get("upstream_errors") or 0
        if policy != "direct" and upstream_errors:
            fails.append(f"{policy}: {upstream_errors} shim-to-vLLM failures (502)")
        if policy != "direct" and errors > max(2, records // 200):
            fails.append(f"{policy}: {errors} errors, more than 0.5 % of {records} records")
        elif policy != "direct" and errors:
            # a completed stream whose every chunk carries empty text (the model emitted EOS
            # first, then special tokens under ignore_eos) has no content timestamp, so the
            # adapter cannot give it a TTFT and counts it as a miss (W3 smoke 2: 8 of 7,820)
            notes.append(
                f"{policy}: {errors} streams without a content timestamp (counted as misses)"
            )
        elif errors:
            # direct: inference-perf reuses keep-alive connections for up to 15 s while uvicorn
            # closes idle ones after 5 s, so a few requests race the close (seen in W3 smoke 2).
            # Through the shim that race is gone (aiohttp server idles 75 s; shim->vLLM is one
            # connection per request), so only the shim arms are held to zero errors.
            notes.append(f"{policy}: {errors} client-side connection errors (keep-alive race)")
        if not m.get("metrics_rows"):
            fails.append(f"{policy}: metrics.csv has no rows")
        cpu, wall = m.get("shim_cpu_s"), m.get("load_wall_s")
        if cpu is not None and wall and cpu / wall > 0.8:
            fails.append(f"{policy}: shim used {cpu / wall:.2f} of a core")
        whole = phases.get("all") or {}
        notes.append(
            f"{policy}: attainment {whole.get('attainment')}, burst "
            f"{(phases.get('burst') or {}).get('attainment')}, rejection {whole.get('rejection_rate')}, "
            f"time-to-recover {m.get('time_to_recover_s')} s, shim CPU {cpu} s of {wall} s, "
            f"first request {lag} s after launch"
        )
    native = manifests.get("passthrough")
    direct = manifests.get("direct")
    if native and direct:
        # the shim-overhead check of spec §3.5: same session, same trace, with and without shim
        p_pre = (native.get("phase_summaries") or {}).get("pre") or {}
        d_pre = (direct.get("phase_summaries") or {}).get("pre") or {}
        p_tpot, d_tpot = p_pre.get("tpot_p95_s"), d_pre.get("tpot_p95_s")
        p_ttft, d_ttft = p_pre.get("ttft_p95_s"), d_pre.get("ttft_p95_s")
        notes.append(
            f"pre-burst p95 through shim vs direct: TPOT {p_tpot} vs {d_tpot} s, "
            f"TTFT {p_ttft} vs {d_ttft} s"
        )
        if None in (p_tpot, d_tpot, p_ttft, d_ttft):
            fails.append("shim-overhead check: missing pre-phase latency")
        else:
            if p_tpot > d_tpot * 1.10 + 0.001:
                fails.append(f"shim adds TPOT: p95 {p_tpot:.4f} s vs direct {d_tpot:.4f} s")
            if p_ttft > d_ttft + 0.05:
                fails.append(f"shim adds TTFT: p95 {p_ttft:.3f} s vs direct {d_ttft:.3f} s")
    for arm in (native, direct):
        if arm and (arm.get("summary") or {}).get("rejected_429"):
            fails.append(f"{arm.get('policy')}: rejections in an arm that must reject none")
    if native:
        pre = (native.get("phase_summaries") or {}).get("pre") or {}
        tpot, ttft = pre.get("tpot_p95_s"), pre.get("ttft_p95_s")
        notes.append(
            f"passthrough pre-burst TPOT p95 {tpot} s (W2 direct {W2_TPOT_P95_S}), "
            f"TTFT p95 {ttft} s (W2 direct {W2_TTFT_P95_S})"
        )
        if tpot is not None and tpot > W2_TPOT_P95_S * 1.15:
            notes.append("WARNING: TPOT through the shim is more than 15 % above W2 direct")
        if ttft is not None and ttft > max(0.2, W2_TTFT_P95_S * 1.5):
            notes.append("WARNING: TTFT through the shim is well above W2 direct")
    bounded = manifests.get("bounded_queue")
    if bounded:
        rejected = ((bounded.get("shim_final") or {}).get("rejected")) or {}
        if not (rejected.get("queue_full") or rejected.get("queue_timeout")):
            fails.append(
                f"bounded_queue: the shim rejected nothing during a 1.5x burst ({rejected})"
            )
        if not (bounded.get("summary") or {}).get("rejected_429"):
            fails.append("bounded_queue: shim rejections did not reach the records as rejected_429")
    return fails, notes


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_w3_smoke.py <smoke seed dir>", file=sys.stderr)
        return 2
    fails, notes = check(Path(argv[0]))
    for n in notes:
        print(f"note: {n}")
    for f in fails:
        print(f"FAIL: {f}")
    print("SMOKE_OK" if not fails else "SMOKE_FAILED")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
