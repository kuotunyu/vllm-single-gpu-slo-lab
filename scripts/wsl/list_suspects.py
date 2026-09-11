"""Print ``seed:value`` for every suspect (or short) stage in an analysis directory (chain helper).

usage: python list_suspects.py <analysis_dir> cl|ol [suspect|short]

``short`` lists stages whose measurement window is empty (``window_records`` 0 or missing): a
closed-loop stage that finished before the 60 s discard period ended. W4's accelerated cells run
faster than the W2-sized ``num_requests`` allowed for (CPU dry run, 2026-09-11: the 2.2x faster
fake engine emptied both closed-loop windows and the cell had no r_sat), so the W4 chain re-runs
such stages with more requests instead of stopping the cell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    out, kind = Path(sys.argv[1]), sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "suspect"
    path = out / ("closed_loop.json" if kind == "cl" else "open_loop.json")
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["rows"]:
        flagged = (not row.get("window_records")) if mode == "short" else row.get("suspect")
        if flagged:
            # open-loop stage dirs are named with the awk "%.2f" rate string (ol-rate-26.20)
            value = row["concurrency"] if kind == "cl" else f"{float(row['offered_rps']):.2f}"
            print(f"{row['seed']}:{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
