"""Print ``seed:value`` for every suspect stage in an analysis directory (used by w2-cell-chain.sh).

usage: python list_suspects.py <analysis_dir> cl|ol
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    out, kind = Path(sys.argv[1]), sys.argv[2]
    path = out / ("closed_loop.json" if kind == "cl" else "open_loop.json")
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["rows"]:
        if row.get("suspect"):
            # open-loop stage dirs are named with the awk "%.2f" rate string (ol-rate-26.20)
            value = row["concurrency"] if kind == "cl" else f"{float(row['offered_rps']):.2f}"
            print(f"{row['seed']}:{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
