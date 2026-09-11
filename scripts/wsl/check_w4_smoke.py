"""Gate the W4 night on its GPU smoke (ADR 0015 item 10); exit 1 on a functional failure.

usage: check_w4_smoke.py <run root> <cell>:<family>:<specdec>[:<stage,stage,...>] ...

For every stage manifest of the named cells (found anywhere below the root as
``<cell>/seed-*/<stage>/manifest.json``): inference-perf ran, requests became records, errors stay
under 0.5 %, the manifest carries the expected labels, the engine reported no preemption, the shim
had no upstream failure, /metrics was scraped, an open-loop stage has a measurement window, and,
for an accelerated cell, the server exposed the spec-decode counters, drafted, and accepted between
0 and 100 % of the draft tokens. A baseline (``none``) cell must expose no spec-decode counters.
Acceptance and probe TPOT are printed as notes; a closed-loop stage that ended inside the 60 s
discard period is noted (the cell chain re-runs such stages with three times the requests).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ERROR_FRACTION = 0.005


def check_manifest(
    m: dict, family: str, specdec: str, stage: str, label: str
) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    notes: list[str] = []
    rc = m.get("inference_perf_returncode")
    records = m.get("records") or 0
    summary = m.get("summary") or {}
    if rc != 0:
        fails.append(f"{label}: inference-perf exit code {rc}")
    if not records:
        fails.append(f"{label}: no records")
    expected = m.get("num_requests")
    if expected and abs(records - expected) > max(2, expected // 1000):
        fails.append(f"{label}: {records} records for {expected} requests")
    errors = summary.get("errors") or 0
    if records and errors > max(2, int(records * ERROR_FRACTION)):
        fails.append(f"{label}: {errors} errors, more than 0.5 % of {records} records")
    if m.get("specdec") != specdec or m.get("family") != family:
        fails.append(
            f"{label}: labels {m.get('specdec')}/{m.get('family')}, expected {specdec}/{family}"
        )
    window = m.get("window_records") or 0
    if stage.startswith("ol-") and not window:
        fails.append(f"{label}: open-loop stage has no measurement window")
    elif not window:
        notes.append(
            f"{label}: closed-loop stage ended inside the 60 s discard (no window; the chain "
            "re-runs such stages with 3x requests, ADR 0015 item 3)"
        )
    preemptions = m.get("preemptions")
    if preemptions is None:
        fails.append(f"{label}: preemption counter not scraped")
    elif preemptions > 0:
        fails.append(
            f"{label}: {preemptions:.0f} preemptions (lower --max-num-seqs, ADR 0015 item 5)"
        )
    shim_final = m.get("shim_final") or {}
    if not m.get("shim_rows"):
        fails.append(f"{label}: shim.csv has no rows (stage did not run through the shim?)")
    elif shim_final.get("upstream_errors"):
        fails.append(f"{label}: {shim_final['upstream_errors']} shim-to-vLLM failures (502)")
    if not m.get("metrics_rows"):
        fails.append(f"{label}: metrics.csv has no rows")
    if m.get("probe_tpot_median_s") is None:
        fails.append(f"{label}: no single-stream probe TPOT")
    sd = m.get("spec_decode")
    if specdec == "none":
        if sd is not None:
            fails.append(f"{label}: a baseline cell exposed spec-decode counters: {sd}")
    elif sd is None:
        fails.append(f"{label}: no spec-decode counters in /metrics (names differ from TRACKED?)")
    else:
        drafts = sd.get("drafts") or 0
        rate = sd.get("acceptance_rate")
        if drafts <= 0:
            fails.append(f"{label}: the server drafted nothing (speculative config inert)")
        elif rate is None or not 0 < rate <= 1:
            fails.append(f"{label}: acceptance rate {rate} outside (0, 1]")
        notes.append(
            f"{label}: drafts {drafts:.0f}, acceptance {rate}, mean acceptance length "
            f"{sd.get('mean_acceptance_length')}, per position {sd.get('acceptance_per_pos')}"
        )
    notes.append(
        f"{label}: records {records}, window {window}, errors {errors}, probe TPOT "
        f"{m.get('probe_tpot_median_s')} s, TPOT p50/p95 {m.get('tpot_p50_s')}/{m.get('tpot_p95_s')} s, "
        f"TTFT p95 {m.get('ttft_p95_s')} s, achieved {m.get('achieved_rps')} rps, "
        f"preemptions {preemptions}, committed "
        f"{((m.get('host_after') or {}).get('windows_gpu_memory') or {}).get('committed_mb')} MB"
    )
    return fails, notes


def check(root: Path, specs: list[str]) -> tuple[list[str], list[str]]:
    fails: list[str] = []
    notes: list[str] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) < 3:
            fails.append(f"bad spec {spec!r}: want <cell>:<family>:<specdec>[:<stages>]")
            continue
        cell, family, specdec = parts[:3]
        wanted = [s for s in parts[3].split(",") if s] if len(parts) > 3 else []
        found = {
            p.parent.name: p
            for p in root.rglob(f"{cell}/seed-*/*/manifest.json")
            if "quarantine" not in p.parts  # stages the chain set aside are not the smoke
        }
        if not found:
            fails.append(f"{cell}: no manifests under {root}")
            continue
        for stage in wanted:
            if stage not in found:
                fails.append(f"{cell}/{stage}: no manifest")
        for stage in wanted or sorted(found):
            path = found.get(stage)
            if path is None:
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            f, n = check_manifest(m, family, specdec, stage, f"{cell}/{stage}")
            fails += f
            notes += n
    return fails, notes


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    fails, notes = check(Path(argv[0]), argv[1:])
    for n in notes:
        print(f"note: {n}")
    for f in fails:
        print(f"FAIL: {f}")
    print("SMOKE_OK" if not fails else "SMOKE_FAILED")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
