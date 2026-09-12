"""Secrets scanner tests. Fixtures build secret-looking strings at runtime so this file itself
never contains anything the scanner would flag (the repo dogfood test below scans it)."""

from pathlib import Path

from slo_lab.redact import (
    DEFAULT_EXCLUDE_DIRS,
    format_findings,
    redact_text,
    scan_text,
    scan_tree,
)

REPO = Path(__file__).resolve().parents[1]

PUBLIC_IP = ".".join(["203", "0", "113", "7"])
LAN_IP = ".".join(["192", "168", "1", "20"])
HF_TOKEN = "hf_" + "A" * 34
RUNPOD_KEY = "rpa_" + "K" * 32
PEM_HEADER = "-" * 5 + "BEGIN OPENSSH PRIVATE KEY" + "-" * 5
SSH_PUB = "ssh-ed25519 " + "AAAA" + "C3NzaC1lZDI1NTE5" + "Q" * 30 + " user@host"
RUNPOD_HOST = "abc123def456" + "-" + "22001" + ".proxy.runpod.net"
EMAIL = "owner@" + "corp.test"


def names(findings):
    return [f.pattern for f in findings]


def test_scan_text_flags_each_pattern_once_per_occurrence():
    text = "\n".join(
        [
            f"ssh -p 22001 root@{PUBLIC_IP}",
            f"export HF_TOKEN={HF_TOKEN}",
            f"key {RUNPOD_KEY}",
            PEM_HEADER,
            SSH_PUB,
            f"host {RUNPOD_HOST}",
            f"contact {EMAIL}",
            f"lan {LAN_IP}",
        ]
    )
    found = scan_text(text, Path("x.log"))
    assert names(found) == [
        "ipv4",
        "hf_token",
        "runpod_api_key",
        "private_key",
        "ssh_public_key",
        "runpod_host",
        "email",
        "ipv4",
    ]
    assert [f.line_no for f in found] == [1, 2, 3, 4, 5, 6, 7, 8]
    # the matched value is never echoed in full
    assert all("…" in f.masked and len(f.masked) <= 5 for f in found)
    assert "x.log:2: hf_token" in format_findings(found)


def test_scan_text_allows_loopback_unspecified_versions_and_noreply_email():
    text = "\n".join(
        [
            "listen 127.0.0.1:8001 and 0.0.0.0",
            "vllm 0.28.0 torch 2.13.0+cu130 driver 591.86",
            "Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.39",
            "61350295+kuotunyu@users.noreply.github.com",
            "someone@example.com",
        ]
    )
    assert scan_text(text) == []
    # addresses next to punctuation are still addresses
    assert names(scan_text(f"peer {LAN_IP}. then host={PUBLIC_IP}:8013")) == ["ipv4", "ipv4"]


def test_runpod_key_assignment_pattern():
    assert names(scan_text("RUNPOD_API_KEY=" + "x" * 20)) == ["runpod_api_key_assignment"]
    assert scan_text("RUNPOD_API_KEY=<REDACTED>") == []


def test_redact_text_replaces_with_placeholders_and_counts():
    text = f"ssh root@{PUBLIC_IP} -p 22001 via {RUNPOD_HOST}; token {HF_TOKEN}; me {EMAIL}"
    out, n = redact_text(text)
    assert n == 4
    assert out == "ssh root@<IP> -p 22001 via <RUNPOD_HOST>; token <HF_TOKEN>; me <EMAIL>"
    unchanged, zero = redact_text("bind 127.0.0.1 ok")
    assert zero == 0 and unchanged == "bind 127.0.0.1 ok"


def test_scan_tree_skips_excluded_dirs_and_binaries(tmp_path):
    (tmp_path / "clean.md").write_text("nothing here\n", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "dirty.log").write_text(f"connect {PUBLIC_IP}\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(f"url = {PUBLIC_IP}\n", encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "x.txt").write_text(HF_TOKEN + "\n", encoding="utf-8")
    # the animation environment and Manim's working directory (2026-09-13) are skipped too
    (tmp_path / ".venv-manim" / "Lib").mkdir(parents=True)
    (tmp_path / ".venv-manim" / "Lib" / "site.py").write_text(f"# {PUBLIC_IP}\n", encoding="utf-8")
    (tmp_path / "media" / "texts").mkdir(parents=True)
    (tmp_path / "media" / "texts" / "t.svg").write_text(f"<!-- {PUBLIC_IP} -->\n", encoding="utf-8")
    # a nested media/ (the committed animations under docs/) is scanned
    (tmp_path / "docs" / "media").mkdir(parents=True)
    (tmp_path / "docs" / "media" / "note.txt").write_text(f"{PUBLIC_IP}\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\0\0" + PUBLIC_IP.encode() + b"\0")
    findings = scan_tree(tmp_path)
    assert [(str(f.path).replace("\\", "/"), f.pattern) for f in findings] == [
        ("docs/media/note.txt", "ipv4"),
        ("logs/dirty.log", "ipv4"),
    ]
    assert scan_tree(tmp_path, exclude_dirs=DEFAULT_EXCLUDE_DIRS | {"logs", "docs"}) == []


def test_scan_tree_reads_gzip_and_files_over_the_old_2mb_cap(tmp_path):
    import gzip

    (tmp_path / "vllm.log.gz").write_bytes(gzip.compress(f"peer {PUBLIC_IP}\n".encode()))
    big = "x" * 80 + "\n"
    (tmp_path / "big.log").write_text(big * 30_000 + f"{HF_TOKEN}\n", encoding="utf-8")
    assert (tmp_path / "big.log").stat().st_size > 2_000_000
    (tmp_path / "blob.gz").write_bytes(gzip.compress(b"\0\0" + PUBLIC_IP.encode()))
    (tmp_path / "fake.gz").write_text(f"not gzip {PUBLIC_IP}\n", encoding="utf-8")
    found = sorted((str(f.path).replace("\\", "/"), f.pattern) for f in scan_tree(tmp_path))
    # fake.gz is plain text with a .gz name: still scanned as text, never silently skipped
    assert found == [("big.log", "hf_token"), ("fake.gz", "ipv4"), ("vllm.log.gz", "ipv4")]


def test_dataset_text_allows_example_addresses_but_not_secrets(tmp_path):
    d = tmp_path / "eval" / "tmmluplus"
    d.mkdir(parents=True)
    (d / "full.jsonl").write_text(
        f'{{"question": "subnet of {LAN_IP}?", "note": "{HF_TOKEN}"}}\n', encoding="utf-8"
    )
    (tmp_path / "notes.md").write_text(f"lab box {LAN_IP}\n", encoding="utf-8")
    found = sorted((str(f.path).replace("\\", "/"), f.pattern) for f in scan_tree(tmp_path))
    assert found == [("eval/tmmluplus/full.jsonl", "hf_token"), ("notes.md", "ipv4")]


def test_repository_tree_is_clean():
    findings = scan_tree(REPO)
    assert findings == [], format_findings(findings)
