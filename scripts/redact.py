#!/usr/bin/env python
"""Redact secrets in a log or audit the tree (design spec §7).

    uv run python scripts/redact.py audit [ROOT]          # exit 2 if anything is found
    uv run python scripts/redact.py redact IN [-o OUT]    # IN may be '-' for stdin

Logic lives in `slo_lab.redact` so it is unit-tested; this file is only the entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slo_lab.redact import format_findings, redact_text, scan_tree


def _audit(root: Path) -> int:
    findings = scan_tree(root)
    if findings:
        print(format_findings(findings), file=sys.stderr)
        print(f"audit-secrets: {len(findings)} finding(s) under {root}", file=sys.stderr)
        return 2
    print(f"audit-secrets: clean ({root})")
    return 0


def _redact(src: str, out: str | None) -> int:
    text = sys.stdin.read() if src == "-" else Path(src).read_text(encoding="utf-8")
    redacted, n = redact_text(text)
    if out is None:
        sys.stdout.write(redacted)
    else:
        Path(out).write_text(redacted, encoding="utf-8", newline="\n")
    print(f"redact: {n} replacement(s)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_audit = sub.add_parser("audit", help="scan a directory tree; non-zero exit on findings")
    p_audit.add_argument("root", nargs="?", default=".")
    p_redact = sub.add_parser("redact", help="replace secrets with placeholders")
    p_redact.add_argument("src")
    p_redact.add_argument("-o", "--out")
    args = parser.parse_args(argv)
    if args.cmd == "audit":
        return _audit(Path(args.root).resolve())
    return _redact(args.src, args.out)


if __name__ == "__main__":
    sys.exit(main())
