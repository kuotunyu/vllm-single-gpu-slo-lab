"""Secrets boundary (design spec §7): redaction of logs and an audit scanner for the tree.

Patterns: IPv4 addresses (loopback and 0.0.0.0 allowed), private-key PEM headers, SSH public
keys, RunPod API keys, RunPod SSH hosts, Hugging Face tokens and e-mail addresses (GitHub
noreply and example.com allowed). `redact_text` swaps matches for placeholders; `scan_tree`
reports them without ever printing the matched value in full.

Pattern literals are assembled at import time so this file never matches itself.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

_DASHES = "-" * 5


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    placeholder: str
    allow: Callable[[str], bool] | None = None


def _is_loopback_or_unspecified(ip: str) -> bool:
    return ip.startswith("127.") or ip == "0.0.0.0"


def _is_public_placeholder_email(addr: str) -> bool:
    domain = addr.rsplit("@", 1)[-1].lower()
    return domain.endswith("noreply.github.com") or domain in {"example.com", "example.org"}


PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "ipv4",
        # Four dotted octets not embedded in a longer token: the WSL2 kernel string that
        # `platform.platform()` writes into every manifest (`Linux-6.6.114.1-microsoft-...`)
        # is a version; an address followed by `:port` or a full stop still counts.
        re.compile(r"(?<![\w-])(?<!\d\.)(?:\d{1,3}\.){3}\d{1,3}(?![\w-]|\.\d)"),
        "<IP>",
        allow=_is_loopback_or_unspecified,
    ),
    Pattern(
        "private_key",
        re.compile(_DASHES + r"BEGIN [A-Z0-9 ]*PRIVATE KEY" + _DASHES),
        "<PRIVATE_KEY_BLOCK>",
    ),
    Pattern(
        "ssh_public_key",
        re.compile(r"\bssh-(?:rsa|ed25519|dss|ecdsa-sha2-nistp\d{3})\s+AAAA[0-9A-Za-z+/=]{20,}"),
        "<SSH_PUBLIC_KEY>",
    ),
    Pattern("runpod_api_key", re.compile(r"\brpa_[A-Za-z0-9]{20,}\b"), "<RUNPOD_API_KEY>"),
    Pattern(
        "runpod_api_key_assignment",
        re.compile(r"RUNPOD_API_KEY\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
        "RUNPOD_API_KEY=<REDACTED>",
    ),
    Pattern(
        "runpod_host",
        re.compile(r"\b[a-z0-9]{6,}-[0-9]{3,6}\.proxy\.runpod\.net\b|\bssh\.runpod\.io\b"),
        "<RUNPOD_HOST>",
    ),
    Pattern("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), "<HF_TOKEN>"),
    Pattern(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "<EMAIL>",
        allow=_is_public_placeholder_email,
    ),
)

DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".venv-manim",  # the explainer animation's environment (scripts/manim/README.md)
        "venv",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        "__pycache__",
        "node_modules",
        "runs",
        "temp",
        "tmp",
        "secrets",
        ".worktrees",
    }
)


class Finding(NamedTuple):
    path: Path
    line_no: int
    pattern: str
    masked: str


def _mask(value: str) -> str:
    return value[:4] + "…" if len(value) > 6 else "…"


def _matches(text: str) -> Iterable[tuple[Pattern, re.Match[str]]]:
    for pat in PATTERNS:
        for m in pat.regex.finditer(text):
            if pat.allow is not None and pat.allow(m.group(0)):
                continue
            yield pat, m


def redact_text(text: str) -> tuple[str, int]:
    """Replace every disallowed match with its placeholder; returns (text, replacements)."""
    total = 0
    for pat in PATTERNS:

        def _sub(m: re.Match[str], pat: Pattern = pat) -> str:
            nonlocal total
            if pat.allow is not None and pat.allow(m.group(0)):
                return m.group(0)
            total += 1
            return pat.placeholder

        text = pat.regex.sub(_sub, text)
    return text, total


def scan_text(text: str, path: Path = Path("-")) -> list[Finding]:
    findings: list[Finding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        findings.extend(
            Finding(path, line_no, pat.name, _mask(m.group(0))) for pat, m in _matches(line)
        )
    return findings


def _looks_binary(head: bytes) -> bool:
    return b"\0" in head


_GZIP_MAGIC = b"\x1f\x8b"
# Evidence stores server logs and per-request records gzip-compressed (ADR 0011), and a single
# server log runs to 11 MB; the old 2 MB cap silently skipped exactly those files. The caps below
# cover every committed file with room to spare; the decompressed cap bounds a hostile archive.
DEFAULT_MAX_BYTES = 64_000_000
MAX_DECOMPRESSED_BYTES = 512_000_000


# Skipped only at the repository root: Manim's working directory (partial movie files, cached
# text SVGs). ``docs/media`` holds the committed animations and stays in the scan.
DEFAULT_EXCLUDE_TOP_DIRS: frozenset[str] = frozenset({"media"})

# Third-party benchmark text committed verbatim (TMMLU+ is MIT, ADR 0003). Its networking exam
# questions quote example addresses (subnet masks, 192.168.x.x), which are question content, not
# infrastructure; every other pattern (keys, tokens, e-mail) still applies to these files.
DATASET_TEXT_DIRS: tuple[str, ...] = ("eval/tmmluplus/",)
DATASET_ALLOWED_PATTERNS: frozenset[str] = frozenset({"ipv4"})


def _is_gzip(path: Path, head: bytes) -> bool:
    return path.suffix == ".gz" and head.startswith(_GZIP_MAGIC)


def iter_text_files(
    root: Path,
    *,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
    exclude_top_dirs: frozenset[str] = DEFAULT_EXCLUDE_TOP_DIRS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Iterable[Path]:
    """Every scannable file under ``root``: plain text, plus gzip-compressed text (``*.gz``)."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).parts
        if any(part in exclude_dirs for part in rel[:-1]):
            continue
        if len(rel) > 1 and rel[0] in exclude_top_dirs:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            with path.open("rb") as fh:
                head = fh.read(8192)
        except OSError:
            continue
        if _is_gzip(path, head) or not _looks_binary(head):
            yield path


def _read_for_scan(path: Path) -> str | None:
    """Text content for the scanner; decompresses real gzip; None if unreadable or binary.

    Decided by magic bytes, not the name, so a plain file called ``*.gz`` is still scanned.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(len(_GZIP_MAGIC))
    except OSError:
        return None
    if _is_gzip(path, head):
        try:
            with gzip.open(path, "rb") as fh:
                data = fh.read(MAX_DECOMPRESSED_BYTES + 1)
        except (OSError, EOFError):
            return None
        if len(data) > MAX_DECOMPRESSED_BYTES or _looks_binary(data[:8192]):
            return None
        return data.decode("utf-8", errors="replace")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def scan_tree(
    root: Path,
    *,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_text_files(root, exclude_dirs=exclude_dirs, max_bytes=max_bytes):
        text = _read_for_scan(path)
        if text is None:
            continue
        rel = path.relative_to(root)
        found = scan_text(text, rel)
        if rel.as_posix().startswith(DATASET_TEXT_DIRS):
            found = [f for f in found if f.pattern not in DATASET_ALLOWED_PATTERNS]
        findings.extend(found)
    return findings


def format_findings(findings: Iterable[Finding]) -> str:
    return "\n".join(f"{f.path}:{f.line_no}: {f.pattern} ({f.masked})" for f in findings)
