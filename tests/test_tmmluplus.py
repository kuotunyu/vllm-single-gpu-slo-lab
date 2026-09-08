"""TMMLU+ slice construction is deterministic, disjoint, proportional, and frozen by hash."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slo_lab.tmmluplus import (
    Item,
    build_slices,
    load_test_items,
    parse_letter,
    read_slice,
    slice_digest,
    write_slices,
)


def _write_subject(dir_: Path, subject: str, n: int) -> None:
    lines = ["question,A,B,C,D,answer"]
    for i in range(n):
        lines.append(f"{subject} q{i},a{i},b{i},c{i},d{i},{'ABCD'[i % 4]}")
    (dir_ / f"{subject}_test.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    _write_subject(d, "accounting", 60)
    _write_subject(d, "dentistry", 30)
    _write_subject(d, "physics", 10)
    return d


def test_slices_are_disjoint_proportional_and_deterministic(data_dir: Path) -> None:
    subjects = load_test_items(data_dir)
    first = build_slices(subjects, n_items=20, n_slices=3, seed=7)
    second = build_slices(subjects, n_items=20, n_slices=3, seed=7)
    assert [slice_digest(s) for s in first] == [slice_digest(s) for s in second]
    keys = [{(i.subject, i.index) for i in s} for s in first]
    assert all(len(s) == 20 for s in first)
    assert not (keys[0] & keys[1]) and not (keys[1] & keys[2]) and not (keys[0] & keys[2])
    per_subject = {sub: sum(1 for i in first[0] if i.subject == sub) for sub in subjects}
    assert per_subject == {"accounting": 12, "dentistry": 6, "physics": 2}
    assert build_slices(subjects, n_items=20, n_slices=3, seed=8)[0] != first[0]


def test_slices_refuse_when_subjects_cannot_stay_disjoint(data_dir: Path) -> None:
    subjects = load_test_items(data_dir)
    with pytest.raises(ValueError, match="disjoint"):
        build_slices(subjects, n_items=50, n_slices=3, seed=1)


def test_write_and_read_round_trip_with_manifest(data_dir: Path, tmp_path: Path) -> None:
    subjects = load_test_items(data_dir)
    slices = build_slices(subjects, n_items=20, n_slices=2, seed=3)
    out = tmp_path / "slices"
    manifest = write_slices(slices, out, seed=3, source="test")
    stored = json.loads((out / "slices.json").read_text(encoding="utf-8"))
    assert stored == manifest
    assert manifest["slices"][0]["sha256"] == slice_digest(slices[0])
    assert read_slice(out / "slice-2.jsonl") == list(slices[1])


def test_prompt_and_letter_parsing() -> None:
    item = Item("x", 0, "問題？", "甲", "乙", "丙", "丁", "C")
    assert "A. 甲" in item.prompt() and item.prompt().endswith("答案：")
    assert parse_letter("<think>\n\n</think>\n\nC") == "C"
    assert parse_letter(" 答案是 B。") == "B"
    assert parse_letter("D. 丁") == "D"
    assert parse_letter("沒有字母") is None


def test_bad_answer_letter_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad_test.csv").write_text(
        "question,A,B,C,D,answer\nq,a,b,c,d,E\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not A-D"):
        load_test_items(tmp_path)
