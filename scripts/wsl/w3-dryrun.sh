#!/usr/bin/env bash
# CPU-only end-to-end dry run of the W3 chain against fake_vllm.py (plan Task A7); no GPU.
# Runs one seed through all three policies on a 2-minute profile, promotes into a throwaway
# evidence week, rebuilds the admission tables from it, prints the key manifest fields, and
# removes the throwaway evidence again.
set -uo pipefail
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
WSL="$REPO/scripts/wsl"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
ROOT="$HOME/vllm-slo-lab/runs-w3-dryrun"
rm -rf "$ROOT" "$REPO/evidence/raw/w3-dryrun"
mkdir -p "$ROOT"
cat > "$ROOT/profile.yaml" <<'YAML'
name: dryrun
kind: open_loop_multistage
stages:
  - {rate_multiplier: 0.5, duration_s: 20}
  - {rate_multiplier: 1.5, duration_s: 20}
  - {rate_multiplier: 0.5, duration_s: 80}
YAML
# fake engine: 8 slots x (132 tokens x 2 ms) ~ 30 rps; r = 30 gives 15 / 45 / 15 rps
FAKE_ENGINE=1 FAKE_SLOTS=8 FAKE_TOKEN_MS=2 SEEDS=1 PROMOTE=0 RERUN_MAX=1 \
  PROFILE="$ROOT/profile.yaml" RUN_ROOT="$ROOT" \
  bash "$WSL/w3-trace-chain.sh" dry Qwen/Qwen3-8B-FP8 256 8 30
EVIDENCE_WEEK=w3-dryrun bash "$WSL/promote-w2.sh" "$ROOT/dry/seed-1" "dry/trace/seed-1"
"$LABPY" "$REPO/scripts/analyze_batch.py" "$REPO/evidence/raw/w3-dryrun/dry" --out "$ROOT/tables" > /dev/null
"$LABPY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for policy in ("passthrough", "hard_cap", "bounded_queue"):
    p = root / "dry" / "seed-1" / f"trace-{policy}" / "manifest.json"
    if not p.exists():
        print(policy, "NO MANIFEST")
        continue
    m = json.loads(p.read_text())
    s = m.get("summary") or {}
    ph = m.get("phase_summaries") or {}
    print(
        policy,
        "rc", m.get("inference_perf_returncode"),
        "records", m.get("records"), "arrivals", (m.get("trace") or {}).get("arrivals"),
        "429", s.get("rejected_429"), "timeouts", s.get("timeouts"), "errors", s.get("errors"),
        "burst_att", (ph.get("burst") or {}).get("attainment"),
        "ttr", m.get("time_to_recover_s"), "ttr_att", m.get("time_to_recover_attainment_s"),
        "lag", m.get("first_request_after_launch_s"),
        "shim_rows", m.get("shim_rows"), "metrics_rows", m.get("metrics_rows"),
        "shim_cpu", m.get("shim_cpu_s"), "wall", m.get("load_wall_s"),
        "shim_final", (m.get("shim_final") or {}).get("rejected"),
    )
PY
echo "--- tables.md (head)"
head -12 "$ROOT/tables/tables.md"
echo "--- promoted files"
find "$REPO/evidence/raw/w3-dryrun" -type f | sed "s#$REPO/##" | sort | head -40
rm -rf "$REPO/evidence/raw/w3-dryrun"
echo "DRYRUN DONE"
