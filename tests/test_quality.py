"""Paired quality comparison: same items, two cells (spec §5, preregistration "同題配對")."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slo_lab.quality import compare, mcnemar_exact, read_scored


def _write(path: Path, correct: list[bool], *, errors: set[int] | None = None) -> Path:
    errors = errors or set()
    path.write_text(
        json.dumps(
            {
                "model": path.stem,
                "n": len(correct),
                "records": [
                    {
                        "subject": "law" if i % 2 else "accounting",
                        "index": i,
                        "answer": "B",
                        "predicted": "B" if ok else "A",
                        "correct": ok,
                        "error": "ConnectionResetError" if i in errors else None,
                    }
                    for i, ok in enumerate(correct)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_mcnemar_exact_is_symmetric_and_bounded() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(9, 1) == pytest.approx(2 * (10 + 1) / 1024)
    # more discordance on one side is less likely under the null
    assert mcnemar_exact(20, 0) < mcnemar_exact(15, 5) < mcnemar_exact(11, 9)
    # The full TMMLU+ set produces thousands of discordant items and 2**n overflows a float
    # above n = 1024, so the tail is summed in log space. Cross-checked against the normal
    # approximation with continuity correction, z = (|d| - 1) / sqrt(n):
    #   1600 vs 1500 -> z = 99/55.68 = 1.778 -> p = 0.0754
    #   9000 vs 8500 -> z = 499/132.3 = 3.772 -> p = 1.62e-4
    assert mcnemar_exact(1600, 1500) == pytest.approx(0.0754, abs=5e-4)
    assert mcnemar_exact(9000, 8500) == pytest.approx(1.62e-4, rel=0.02)
    assert mcnemar_exact(1550, 1550) == 1.0
    assert mcnemar_exact(3000, 1000) == pytest.approx(2.49e-229, rel=0.01)
    assert mcnemar_exact(5000, 0) == 0.0  # below the smallest double, reported as zero


def test_read_scored_keys_on_subject_and_index(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.json", [True, False, True])
    scored = read_scored(p)
    assert scored == {("accounting", 0): True, ("law", 1): False, ("accounting", 2): True}


def test_compare_reports_paired_delta_and_exact_p(tmp_path: Path) -> None:
    # 100 items: identical on 90, a-only right on 2, b-only right on 8
    a = [True] * 90 + [True, True] + [False] * 8
    b = [True] * 90 + [False, False] + [True] * 8
    result = compare(_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b), n_boot=200)
    assert result.n_shared == 100
    assert result.a_correct == 92 and result.b_correct == 98
    assert result.a_accuracy == 0.92 and result.b_accuracy == 0.98
    assert result.discordant_a_only == 2 and result.discordant_b_only == 8
    assert result.delta == pytest.approx(0.06)
    assert result.p_value == pytest.approx(mcnemar_exact(2, 8))
    lo, hi = result.delta_ci95
    assert lo < result.delta < hi
    # the paired interval must be far tighter than two independent Wilson intervals on 0.92/0.98
    assert hi - lo < 0.15


def test_compare_excludes_items_that_errored_on_either_side(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.json", [True, True, False, True], errors={1})
    b = _write(tmp_path / "b.json", [True, False, False, True], errors={3})
    result = compare(a, b, n_boot=50)
    assert result.n_shared == 2  # items 0 and 2 survive
    assert result.excluded_errors == 2
    assert result.a_correct == 1 and result.b_correct == 1


def test_rebuild_paired_tables_uses_bf16_as_baseline_and_is_deterministic(tmp_path: Path) -> None:
    from slo_lab.quality import find_full_sets, rebuild_paired_tables

    for cell, correct in (
        ("bf16", [True] * 80 + [False] * 20),
        ("fp8", [True] * 78 + [False] * 22),
        ("awq", [True] * 70 + [False] * 30),
    ):
        d = tmp_path / "evidence" / "raw" / "w2" / cell / "tmmluplus"
        d.mkdir(parents=True)
        _write(d / "full.json", correct)
    assert sorted(find_full_sets(tmp_path)) == ["awq", "bf16", "fp8"]
    out = tmp_path / "analysis" / "tables" / "w2-quality-paired"
    assert rebuild_paired_tables(tmp_path, out) == ["awq", "fp8"]
    first = (out / "paired.md").read_text(encoding="utf-8")
    payload = json.loads((out / "paired.json").read_text(encoding="utf-8"))
    assert payload["baseline"] == "bf16"
    assert [r["b"] for r in payload["rows"]] == ["awq", "fp8"]
    assert payload["rows"][0]["delta_b_minus_a"] == pytest.approx(-0.10)
    rebuild_paired_tables(tmp_path, out)
    assert (out / "paired.md").read_text(encoding="utf-8") == first
    assert "McNemar" in first and "| awq |" in first


def test_rebuild_paired_tables_needs_two_cells(tmp_path: Path) -> None:
    from slo_lab.quality import rebuild_paired_tables

    assert rebuild_paired_tables(tmp_path, tmp_path / "out") == []


def test_compare_refuses_when_the_item_sets_do_not_overlap(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.json", [True, True])
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"records": [{"subject": "z", "index": 99, "correct": True}]}), "utf-8")
    with pytest.raises(ValueError, match="no shared items"):
        compare(a, b)
