"""Print ``<seed> <policy>`` for every suspect trace stage in an admission.json (W3 chain helper).

A missing or unreadable file prints nothing, so the chain treats it as "no suspects" and the
final analysis step reports the real problem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> int:
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8")).get("rows", [])
    except (OSError, ValueError):
        return 0
    for row in rows:
        if row.get("suspect"):
            print(row.get("seed"), row.get("policy"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
