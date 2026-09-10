#!/usr/bin/env bash
# Promote one batch (runs/<batch>/seed-N) into the repo's evidence tree.
# usage: promote-w2.sh <batch-dir> <dest-rel-under-evidence/raw/w2> [session-tag]
# A batch dir keeps only its latest server session's serve.log and quiet_gpu.json, so a later session
# in the same dir (refine-cell.sh) passes a session tag and lands as vllm-<tag>.log /
# quiet_gpu-<tag>.json instead of replacing the main run's files. io-pressure.log is appended at
# the source, so it is always a superset and keeps its name.
# Copies the digest-bearing small files; drops the 10+ MB per_request_lifecycle_metrics.json
# (records.jsonl is the canonical form and manifest.json carries its sha256). Home path -> ~,
# serve.log passes through scripts/redact.py as vllm.log; records.jsonl and vllm*.log are then
# gzipped by scripts/compress_evidence.py (ADR 0011).
set -euo pipefail
SRC="$1"; REL="$2"; TAG="${3:-}"
SFX=""; [ -n "$TAG" ] && SFX="-$TAG"
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
PY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
DEST="$REPO/evidence/raw/w2/$REL"
mkdir -p "$DEST"
norm() { sed -e "s#$HOME#~#g" "$1" > "$2"; }
[ -f "$SRC/quiet_gpu.json" ] && norm "$SRC/quiet_gpu.json" "$DEST/quiet_gpu$SFX.json"
[ -f "$SRC/io-pressure.log" ] && norm "$SRC/io-pressure.log" "$DEST/io-pressure.log"
if [ -f "$SRC/serve.log" ]; then
  sed -e "s#$HOME#~#g" "$SRC/serve.log" | "$PY" "$REPO/scripts/redact.py" redact - -o "$DEST/vllm$SFX.log"
fi
for stage in "$SRC"/*/; do
  [ -f "$stage/manifest.json" ] || continue
  name=$(basename "$stage"); mkdir -p "$DEST/$name/ipf"
  for f in manifest.json records.jsonl power.csv power-warmup.csv metrics.csv warmup-ttft.json inference-perf.yaml; do
    [ -f "$stage/$f" ] && norm "$stage/$f" "$DEST/$name/$f"
  done
  [ -f "$stage/inference-perf.log" ] && sed -e "s#$HOME#~#g" "$stage/inference-perf.log" | "$PY" "$REPO/scripts/redact.py" redact - -o "$DEST/$name/inference-perf.log"
  for f in config.yaml summary_lifecycle_metrics.json stage_0_lifecycle_metrics.json; do
    [ -f "$stage/ipf/$f" ] && norm "$stage/ipf/$f" "$DEST/$name/ipf/$f"
  done
done
echo "promoted -> evidence/raw/w2/$REL"; du -sh "$DEST"; find "$DEST" -type f | wc -l
"$PY" "$REPO/scripts/compress_evidence.py" "$DEST"   # records.jsonl, vllm*.log -> .gz (ADR 0011)
if grep -rl "/home/" "$DEST" || find "$DEST" -name "*.gz" -exec zgrep -l "/home/" {} +; then
  echo "WARNING: home path remains"
else
  echo "home paths clean"
fi
