"""Gzip the bulky text evidence in place: per-request records and vLLM server logs (ADR 0011).

    python scripts/compress_evidence.py [DIR ...]        # default: evidence/raw

Compresses every ``records.jsonl`` and every server log (``vllm.log``, ``vllm-<tag>.log``) under
the given directories to ``<name>.gz``, verifies the round trip byte for byte, and only then
removes the plain file. Output is deterministic (level 9, mtime 0, no stored file name), so the
same input always produces the same bytes and re-running changes nothing. A plain file next to an
existing ``.gz`` wins: promotion writes plain files, so the plain copy is the newer one.

Readers never need to know which form is on disk: ``slo_lab.slo.open_evidence_text`` opens
``records.jsonl`` or ``records.jsonl.gz``, and ``make audit-secrets`` scans inside ``.gz``.
Standard library only, so the WSL promotion scripts can call it with any Python.
"""

from __future__ import annotations

import gzip
import io
import sys
from pathlib import Path

LEVEL = 9


def is_target(path: Path) -> bool:
    name = path.name
    if name == "records.jsonl":
        return True
    return name.startswith("vllm") and name.endswith(".log")


def gzip_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=LEVEL, fileobj=buf, mtime=0) as gz:
        gz.write(data)
    return buf.getvalue()


def compress_file(path: Path) -> tuple[int, int]:
    """Replace ``path`` with ``path.gz``; returns (plain bytes, compressed bytes)."""
    data = path.read_bytes()
    packed = gzip_bytes(data)
    if gzip.decompress(packed) != data:
        raise RuntimeError(f"round trip mismatch: {path}")
    target = path.with_name(path.name + ".gz")
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(packed)
    tmp.replace(target)
    path.unlink()
    return len(data), len(packed)


def compress_tree(roots: list[Path]) -> tuple[int, int, int]:
    files = before = after = 0
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.is_file() and is_target(path):
                plain, packed = compress_file(path)
                files += 1
                before += plain
                after += packed
    return files, before, after


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path("evidence/raw")]
    missing = [str(r) for r in roots if not r.exists()]
    if missing:
        print(f"compress-evidence: not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    files, before, after = compress_tree(roots)
    mib = 1024 * 1024
    print(
        f"compress-evidence: {files} file(s), {before / mib:.1f} MiB -> {after / mib:.1f} MiB"
        if files
        else "compress-evidence: nothing to compress"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
