#!/usr/bin/env bash
# Promote `vllm bench serve` cross-check results into evidence/raw/w2/<cell>/crosscheck/ as
# summary-only JSON.
#
# The full --save-detailed output carries per-request arrays, and `generated_texts` holds every
# completion the model wrote for the random-token prompts (6,000 x 132 tokens per file). On
# 2026-09-10 those texts contained regurgitated training-data fragments — other people's build
# paths and website paths — which do not belong in a public repository, and each file was
# 10-22 MB. The same policy already applies to inference-perf, whose per-request JSON stays out of
# the repo with its sha256 in the manifest: keep every scalar metric, drop the arrays, record the
# array lengths and the sha256 of the untouched original, which stays under runs-w2.
# usage: promote-crosscheck.sh [cell]      (default fp8)
set -uo pipefail
CELL="${1:-fp8}"
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
PY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
SRC="$HOME/vllm-slo-lab/runs-w2/crosscheck/$CELL"
DST="$REPO/evidence/raw/w2/$CELL/crosscheck"
mkdir -p "$DST"
rm -f "$DST"/*.json
"$PY" - "$SRC" "$DST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for path in sorted(src.glob("*.json")):
    if path.name == "quiet_gpu.json":
        continue
    raw = path.read_bytes()
    data = json.loads(raw)
    summary = {k: v for k, v in data.items() if not isinstance(v, list)}
    summary["_per_request_arrays_dropped"] = {k: len(v) for k, v in data.items() if isinstance(v, list)}
    errors = data.get("errors") or []
    summary["_error_count"] = sum(1 for e in errors if e)
    summary["_original_sha256"] = hashlib.sha256(raw).hexdigest()
    summary["_original_bytes"] = len(raw)
    summary["_original_path"] = f"~/vllm-slo-lab/runs-w2/crosscheck/{src.name}/{path.name}"
    (dst / path.name).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.name}: {len(raw):,} bytes -> {(dst / path.name).stat().st_size:,} bytes summary, "
          f"errors {summary['_error_count']}, sha256 {summary['_original_sha256'][:12]}")
PY
[ -f "$SRC/quiet_gpu.json" ] && sed -e "s#$HOME#~#g" "$SRC/quiet_gpu.json" > "$DST/quiet_gpu.json"
[ -f "$SRC/serve.log" ] && sed -e "s#$HOME#~#g" "$SRC/serve.log" | "$PY" "$REPO/scripts/redact.py" redact - -o "$DST/vllm.log" 2>/dev/null
du -sh "$DST"
if grep -rl "/home/" "$DST" >/dev/null 2>&1; then echo "WARNING: /home/ still present"; grep -rl "/home/" "$DST"; else echo "no /home/ strings"; fi
