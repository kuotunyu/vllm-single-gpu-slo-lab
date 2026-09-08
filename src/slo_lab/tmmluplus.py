"""TMMLU+ slices for the quality track (spec §5; ADR 0003: ikala/tmmluplus, MIT).

The dataset ships one ``data/<subject>_test.csv`` per subject (67 subjects, 19,680 items) with
columns ``question, A, B, C, D, answer``. The quality track scores each precision on *n*
disjoint slices of 200 items drawn proportionally across subjects with a fixed seed, so the
three "seeds" are three disjoint item sets rather than three decoding seeds (greedy decoding
makes decoding seeds meaningless). Slices are frozen as JSONL plus a SHA-256 list so the
committed evidence pins exactly which items were asked.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Item:
    subject: str
    index: int
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str

    def prompt(self) -> str:
        """Traditional-Chinese four-option prompt; the model must answer with one letter."""
        return (
            "以下是一道單選題，請直接回答正確選項的英文字母（A、B、C 或 D），不要解釋。\n\n"
            f"題目：{self.question}\n"
            f"A. {self.A}\nB. {self.B}\nC. {self.C}\nD. {self.D}\n"
            "答案："
        )


def read_subject_csv(path: Path) -> list[Item]:
    subject = path.name.removesuffix("_test.csv")
    items: list[Item] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            answer = (row.get("answer") or "").strip().upper()
            if answer not in LETTERS:
                raise ValueError(
                    f"{path.name} row {index}: answer {row.get('answer')!r} is not A-D"
                )
            items.append(
                Item(
                    subject=subject,
                    index=index,
                    question=(row.get("question") or "").strip(),
                    A=(row.get("A") or "").strip(),
                    B=(row.get("B") or "").strip(),
                    C=(row.get("C") or "").strip(),
                    D=(row.get("D") or "").strip(),
                    answer=answer,
                )
            )
    return items


def load_test_items(data_dir: Path) -> dict[str, list[Item]]:
    """Return ``{subject: items}`` for every ``*_test.csv`` under ``data_dir`` (sorted)."""
    files = sorted(data_dir.glob("*_test.csv"))
    if not files:
        raise FileNotFoundError(f"no *_test.csv under {data_dir}")
    return {path.name.removesuffix("_test.csv"): read_subject_csv(path) for path in files}


def _proportional_counts(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder apportionment of ``total`` items across subjects by size."""
    population = sum(sizes.values())
    if population < total:
        raise ValueError(f"only {population} items available, {total} requested")
    raw = {subject: total * size / population for subject, size in sizes.items()}
    counts = {subject: int(value) for subject, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(sizes, key=lambda s: (-(raw[s] - counts[s]), s))
    for subject in order[:remainder]:
        counts[subject] += 1
    return counts


def build_slices(
    subjects: dict[str, Sequence[Item]],
    *,
    n_items: int = 200,
    n_slices: int = 3,
    seed: int = 20260908,
) -> list[list[Item]]:
    """Draw ``n_slices`` disjoint slices of ``n_items``, each proportional across subjects.

    Deterministic in (subject order, seed): one shuffle per subject, then consecutive blocks are
    assigned to slices, so slice k never shares an item with slice j.
    """
    sizes = {subject: len(items) for subject, items in subjects.items()}
    counts = _proportional_counts(sizes, n_items)
    rng = random.Random(seed)
    slices: list[list[Item]] = [[] for _ in range(n_slices)]
    for subject in sorted(subjects):
        pool = list(subjects[subject])
        rng.shuffle(pool)
        need = counts[subject] * n_slices
        if need > len(pool):
            raise ValueError(
                f"{subject}: {need} items needed for {n_slices} disjoint slices, {len(pool)} available"
            )
        for k in range(n_slices):
            block = pool[k * counts[subject] : (k + 1) * counts[subject]]
            slices[k].extend(block)
    for k, block in enumerate(slices):
        assert len(block) == n_items, f"slice {k} has {len(block)} items"
    return slices


def slice_digest(items: Iterable[Item]) -> str:
    payload = "\n".join(f"{item.subject}:{item.index}:{item.answer}" for item in items) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_slices(
    slices: Sequence[Sequence[Item]], out_dir: Path, *, seed: int, source: str
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": source,
        "seed": seed,
        "n_slices": len(slices),
        "slices": [],
    }
    for k, block in enumerate(slices, start=1):
        path = out_dir / f"slice-{k}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in block:
                handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        subjects = sorted({item.subject for item in block})
        manifest["slices"].append(  # type: ignore[union-attr]
            {
                "file": path.name,
                "n_items": len(block),
                "n_subjects": len(subjects),
                "sha256": slice_digest(block),
            }
        )
    (out_dir / "slices.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def read_slice(path: Path) -> list[Item]:
    return [
        Item(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_letter(text: str) -> str | None:
    """First standalone A-D letter in a model reply (after any think block)."""
    body = text.split("</think>")[-1] if "</think>" in text else text
    for ch in body.strip().upper():
        if ch in LETTERS:
            return ch
        if ch.isalnum() and ch not in LETTERS:
            continue
    return None
