"""Aggregate stage manifests from batch directories into tables (thin CLI over slo_lab).

usage: python scripts/analyze_batch.py <batch_dir> [<batch_dir> ...] --out analysis/tables/<name>

The logic lives in ``slo_lab.batch_analysis`` so that ``slo-lab reproduce-lite`` can rebuild the
committed tables from ``analysis/tables/index.json`` and ``make reproduce`` can diff them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slo_lab.batch_analysis import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dirs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = run([Path(d) for d in args.batch_dirs], Path(args.out))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
