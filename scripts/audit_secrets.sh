#!/usr/bin/env sh
# Pre-commit / CI wrapper around the secrets scanner (design spec §7).
set -eu
cd "$(dirname "$0")/.."
exec uv run python scripts/redact.py audit .
