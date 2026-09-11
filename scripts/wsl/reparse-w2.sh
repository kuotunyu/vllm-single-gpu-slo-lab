#!/usr/bin/env bash
# W5 item 1: re-adapt every committed W2 stage from its raw loadgen file with the current adapter
# (server completion_tokens instead of inference-perf's re-tokenized count, ADR 0012 appendix), into
# a parallel tree, then compare with the committed records stage by stage and at r_SLO level.
# Reads about 117 GB under ~/vllm-slo-lab (sha256 to match each manifest's raw_sha256, then JSON
# parse), so run it only when no measurement is in progress. Resumable: stages whose new
# records.jsonl exists are skipped; the sha256 index is cached.
#   usage: reparse-w2.sh            env: NEW_ROOT (~/vllm-slo-lab/reparse-w2), RAW_ROOTS (dirs to index)
set -uo pipefail
REPO="/mnt/d/AI-Portfolio/CC_github部隊/vllm-single-gpu-slo-lab"
LABCLI="$HOME/vllm-slo-lab/.venv-slolab/bin/slo-lab"
LABPY="$HOME/vllm-slo-lab/.venv-slolab/bin/python"
EVIDENCE="$REPO/evidence/raw/w2"
NEW_ROOT="${NEW_ROOT:-$HOME/vllm-slo-lab/reparse-w2}"
RAW_ROOTS="${RAW_ROOTS:-$HOME/vllm-slo-lab/runs-w2 $HOME/vllm-slo-lab/runs}"
INDEX="$NEW_ROOT/raw-sha256.txt"
mkdir -p "$NEW_ROOT"
log() { echo "[$(date +%H:%M:%S)] [reparse-w2] $*"; }

# 1. index every raw per-request file by sha256 (cached; the slow part is reading the files once)
if [ ! -s "$INDEX" ]; then
  log "indexing raw files under: $RAW_ROOTS"
  # shellcheck disable=SC2086
  find $RAW_ROOTS -name per_request_lifecycle_metrics.json -type f -print0 2>/dev/null \
    | xargs -0 -n 1 sha256sum > "$INDEX.tmp" && mv "$INDEX.tmp" "$INDEX"
  log "indexed $(wc -l < "$INDEX") raw files"
fi

# 2. re-adapt each committed stage whose raw file is found by digest
n_done=0; n_skip=0; n_missing=0
while IFS= read -r manifest; do
  stage_dir=$(dirname "$manifest")
  rel="${stage_dir#"$EVIDENCE"/}"
  out="$NEW_ROOT/$rel/records.jsonl"
  if [ -f "$out" ]; then n_skip=$((n_skip + 1)); continue; fi
  sha=$("$LABPY" -c "import json,sys; print((json.load(open(sys.argv[1])).get('raw_sha256') or {}).get('per_request_lifecycle_metrics.json',''))" "$manifest")
  [ -n "$sha" ] || { log "no raw_sha256 in $rel"; n_missing=$((n_missing + 1)); continue; }
  raw=$(awk -v s="$sha" '$1 == s {print substr($0, index($0, $2)); exit}' "$INDEX")
  if [ -z "$raw" ]; then log "raw file not found for $rel ($sha)"; n_missing=$((n_missing + 1)); continue; fi
  mkdir -p "$(dirname "$out")"
  "$LABCLI" reparse-records "$raw" --out "$out" >/dev/null && n_done=$((n_done + 1)) || log "FAILED $rel"
done < <(find "$EVIDENCE" -name manifest.json | sort)
log "re-parsed $n_done stage(s), skipped $n_skip already done, $n_missing without a raw file"

# 3. compare with the committed records
"$LABCLI" reparse-compare "$EVIDENCE" "$NEW_ROOT" --out "$NEW_ROOT/compare.json"
log "REPARSE DONE -> $NEW_ROOT/compare.json"
