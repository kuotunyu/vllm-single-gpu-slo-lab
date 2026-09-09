"""Print r_sat (rounded to 2 decimals) or the r_SLO line of an analysis JSON (used by w2-cell-chain.sh).

usage: python read_r_sat.py <closed_loop.json> <cell>
       python read_r_sat.py <open_loop.json> <cell> r_slo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cell = data["per_cell"].get(sys.argv[2]) or {}
    if len(sys.argv) > 3 and sys.argv[3] == "r_slo":
        print(f"r_slo {cell.get('r_slo')} suspects {cell.get('suspect_rates_excluded')}")
        return 0
    r_sat = cell.get("r_sat_rps")
    if r_sat is None:
        return 1
    print(f"{r_sat:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
