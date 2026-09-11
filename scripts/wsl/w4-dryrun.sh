#!/usr/bin/env bash
# CPU-only end-to-end dry run of the W4 chain against fake_vllm.py (plan Task A4); no GPU, no vLLM.
# Two fake cells of one family (dry-none, dry-ngram with the fake's synthetic drafting) go through the
# real cell chain: closed-loop {1, 8} -> r_sat -> open-loop 2 rates x 2 seeds in one session (rates of
# dry-ngram scaled from dry-none's r_sat) -> promote into a throwaway evidence week -> tables rebuilt
# from the promoted tree -> the smoke gate run over the dry-run manifests -> key fields printed -> the
# throwaway evidence removed. About 25 minutes.
set -uo pipefail
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
ROOT="$HOME/vllm-slo-lab/runs-w4-dryrun"
rm -rf "$ROOT" "$REPO/evidence/raw/w4-dryrun"
mkdir -p "$ROOT"
# fake engine: 256 slots x (132 tokens x 12 ms) = 1.58 s per request without acceleration, no batch
# slowdown, so r_sat is c / 1.58 s at the largest concurrency and the open-loop rates stay small
export FAKE_ENGINE=1 FAKE_SLOTS=256 FAKE_TOKEN_MS=12
export WARMUP=10 REWARM=5 RUN_ROOT="$ROOT" RERUN_MAX=1 PROMOTE=1 EVIDENCE_WEEK=w4-dryrun
export CLOSED_GRID="1 8" NREQ_SCALE=0.6 MULTIPLIERS="0.5 0.9" OPEN_SEEDS="1 2" STAGE_S=80
bash "$WSL/w4-cell-chain.sh" dry-none Qwen/Qwen3-8B-FP8 dry none 256
bash "$WSL/w4-cell-chain.sh" dry-ngram Qwen/Qwen3-8B-FP8 dry ngram 256 "$(cat "$ROOT/closed-loop-cells/dry-none/r_sat.txt")"
echo "--- tables from the promoted evidence"
"$LABPY" "$REPO/scripts/analyze_batch.py" "$REPO/evidence/raw/w4-dryrun/dry-none" "$REPO/evidence/raw/w4-dryrun/dry-ngram" --out "$ROOT/tables" | head -3
grep -A 12 "# Speculative decoding" "$ROOT/tables/tables.md" | head -40
echo "--- smoke gate over the dry-run manifests"
"$LABPY" "$WSL/check_w4_smoke.py" "$ROOT" dry-none:dry:none dry-ngram:dry:ngram
echo "--- key manifest fields"
"$LABPY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.glob("*-cells/*/seed-*/*/manifest.json")):
    m = json.loads(path.read_text())
    s = m.get("summary") or {}
    sd = m.get("spec_decode") or {}
    print(
        path.parent.parent.parent.name, path.parent.parent.name, path.parent.name,
        "rc", m.get("inference_perf_returncode"), "records", m.get("records"), "win", m.get("window_records"),
        "errors", s.get("errors"), "att", s.get("attainment_offered"),
        "tpot_p50", m.get("tpot_p50_s"), "specdec", m.get("specdec"), "family", m.get("family"),
        "drafts", sd.get("drafts"), "acc", sd.get("acceptance_rate"), "mal", sd.get("mean_acceptance_length"),
        "preempt", m.get("preemptions"), "shim_up_err", (m.get("shim_final") or {}).get("upstream_errors"),
        "shim_rows", m.get("shim_rows"), "extra", (m.get("engine_flags") or {}).get("extra"),
    )
PY
echo "--- promoted files"
find "$REPO/evidence/raw/w4-dryrun" -type f | sed "s#$REPO/##" | sort | head -60
rm -rf "$REPO/evidence/raw/w4-dryrun"
echo "DRYRUN DONE"
