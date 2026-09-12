# W2 Knee and W4 Spec-Decode Animations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two more Manim Community explainer animations (W2 knee / four precisions; W4 speculative decoding), same plain style as the revised W3 one, embedded in the README as illustrations.

**Architecture:** A shared `scripts/manim/style.py` (font, colours, text helpers, bottom note, source card) used by all three scenes. Each new scene has a manim-free loader (`w2_data.py`, `w4_data.py`) whose values are pinned to the tables by CI tests. Scenes only draw. Outputs go to `docs/media/`; nothing enters `make reproduce` or CI.

**Tech Stack:** Python 3.12, uv, `manim==0.21.0` in `.venv-manim`, ffmpeg; `slo_lab.plots.attainment_series` / `tpot_series` for the W2 curves.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-13-w2-w4-animations-design.md`.
- No title cards, no decorative effects: plain `FadeIn`, `Create` for curves, `GrowFromEdge` for bars only.
- No LaTeX: `DecimalNumber(..., mob_class=Text)`, axes with `label_constructor=Text`; units in labels, never `unit=`.
- Every number drawn comes from the loaders; the loaders read only `analysis/tables/`.
- Repo conventions as in the W3 plan (kuotunyu commits, no trailers, ruff, tests, audit by exit code, `newline="\n"`).
- `.venv-manim/` and `media/` are ignored by git and by the secrets audit.

---

### Task 1: Shared style module and W3 refactor

**Files:**
- Create: `scripts/manim/style.py`
- Modify: `scripts/manim/w3_admission.py` (import the shared constants and helpers instead of defining them)

**Interfaces:**
- Produces: `FONT`, `BG`, `INK`, `MUTED`, `RED`, `GREEN`, `BLUE`, `AMBER`, `text(s, size=28, color=INK, weight="NORMAL") -> Text`, `bottom_note(lines: Sequence[str]) -> VGroup` (muted 16 pt lines stacked at the bottom edge), `source_card(lines: Sequence[str]) -> VGroup` (centred 24 pt lines, last one muted 28 pt), `number(value, decimals, size, color) -> DecimalNumber` (Text-rendered).

- [ ] **Step 1: Write `scripts/manim/style.py`**

```python
"""Shared look of the explainer animations: font, colours, text helpers (no LaTeX)."""

from __future__ import annotations

from collections.abc import Sequence

from manim import DOWN, DecimalNumber, Text, VGroup

FONT = "Microsoft JhengHei"
BG = "#101418"
INK = "#e8ecf1"
MUTED = "#8a94a3"
RED = "#ff4d4d"
GREEN = "#7bc96f"
BLUE = "#4c9be8"
AMBER = "#c9a227"
ORANGE = "#e4572e"
GREY = "#9aa5b1"


def text(s: str, size: int = 28, color: str = INK, weight: str = "NORMAL") -> Text:
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def number(value: float, decimals: int = 2, size: int = 26, color: str = INK) -> DecimalNumber:
    """A DecimalNumber drawn with Pango text; units belong in a separate label."""
    return DecimalNumber(
        value, num_decimal_places=decimals, font_size=size, color=color, mob_class=Text
    )


def bottom_note(lines: Sequence[str]) -> VGroup:
    group = VGroup(*[text(s, 16, MUTED) for s in lines]).arrange(DOWN, buff=0.08)
    return group.to_edge(DOWN, buff=0.2)


def source_card(lines: Sequence[str]) -> VGroup:
    mobs = [text(s, 24) for s in lines[:-1]] + [text(lines[-1], 28, MUTED)]
    return VGroup(*mobs).arrange(DOWN, buff=0.3)
```

- [ ] **Step 2: Refactor `w3_admission.py`**

Replace its `FONT`, `BG`, `INK`, `MUTED`, `RED`, `GREEN` constants and `_t` with `from style import FONT, BG, INK, MUTED, RED, GREEN, text as _t, source_card` (keep `COLORS`, `LABELS` local); build the end card with `source_card([...])` and the bottom note with `bottom_note([...])`. Behaviour unchanged.

- [ ] **Step 3: Verify the W3 scene still imports and renders one frame**

Run: `uv run ruff format scripts/manim && uv run ruff check . && .venv-manim/Scripts/manim -ql -s scripts/manim/w3_admission.py W3AdmissionBurst --media_dir <scratchpad>/media`
Expected: a PNG of the last frame (the source card) without errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/manim/style.py scripts/manim/w3_admission.py
GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "scripts/manim/style: shared font, colours and text helpers for the explainer scenes; w3_admission uses it" < /dev/null
```

---

### Task 2: `w2_data.py` loader with CI test

**Files:**
- Create: `scripts/manim/w2_data.py`
- Test: `tests/test_w2_anim_data.py` (load by path with `sys.modules` registration, as in `tests/test_w3_anim_data.py`)

**Interfaces:**
- Consumes: `slo_lab.plots.attainment_series`, `tpot_series` (each `list[Series]`, `Series.name`, `Series.points`).
- Produces:
  - `CELLS = ("bf16", "fp8", "awq", "gptq")`, `LABELS`, `OPEN_TABLES`, `CLOSED_TABLES`
  - `Cell(name, label, attainment, tpot_p95_ms, ttft_p95_s, r_slo, r_sat, wh_per_m_tokens, tmmlu_accuracy, tmmlu_delta, mcnemar_p)`
  - `KneeData(cells: dict[str, Cell], sensitivity_fp8: dict[tuple[float, float], float], rate_max: float)`
  - `load_knee(root: Path) -> KneeData`

- [ ] **Step 1: Write the failing test**

```python
"""The W2 knee animation's loader reads exactly the committed tables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("w2_data", REPO / "scripts" / "manim" / "w2_data.py")
assert _spec is not None and _spec.loader is not None
w2_data = importlib.util.module_from_spec(_spec)
sys.modules["w2_data"] = w2_data
_spec.loader.exec_module(w2_data)


@pytest.fixture(scope="module")
def knee():
    return w2_data.load_knee(REPO)


def test_curves_and_r_slo_match_the_tables(knee) -> None:
    assert list(knee.cells) == ["bf16", "fp8", "awq", "gptq"]
    fp8 = knee.cells["fp8"]
    assert dict(fp8.attainment)[28.39] == pytest.approx(0.9102, abs=1e-4)
    assert dict(fp8.attainment)[26.2] == 1.0 and dict(fp8.attainment)[43.67] == 0.0
    assert [c.r_slo for c in knee.cells.values()] == [10.26, 26.2, 22.61, 22.7]
    assert knee.cells["fp8"].r_sat == pytest.approx(41.35, abs=0.01)
    assert knee.cells["bf16"].r_sat == pytest.approx(11.40, abs=0.01)
    assert dict(fp8.tpot_p95_ms)[26.2] < 50 < dict(fp8.tpot_p95_ms)[32.75]
    assert max(t for r, t in fp8.ttft_p95_s if r <= 30.57) < 0.2
    assert knee.rate_max >= 87.34


def test_energy_quality_and_sensitivity(knee) -> None:
    wh = {c.name: c.wh_per_m_tokens for c in knee.cells.values()}
    assert wh["bf16"] == pytest.approx(74.6, abs=0.1)
    assert wh["fp8"] == pytest.approx(31.2, abs=0.1)
    assert wh["awq"] == pytest.approx(36.2, abs=0.1)
    assert wh["gptq"] == pytest.approx(36.3, abs=0.1)
    assert knee.cells["bf16"].tmmlu_accuracy == pytest.approx(0.5911, abs=1e-4)
    assert knee.cells["bf16"].tmmlu_delta is None and knee.cells["bf16"].mcnemar_p is None
    assert knee.cells["fp8"].tmmlu_delta == pytest.approx(-0.0002, abs=1e-4)
    assert knee.cells["fp8"].mcnemar_p == pytest.approx(0.945, abs=0.005)
    assert knee.cells["gptq"].tmmlu_delta == pytest.approx(-0.0208, abs=1e-4)
    s = knee.sensitivity_fp8
    assert s[(1.0, 0.03)] == 24.02 and s[(1.0, 0.05)] == 26.2 and s[(2.0, 0.1)] == 30.57
    assert len(s) == 9
```

- [ ] **Step 2: Run it to verify it fails** — `uv run pytest -q tests/test_w2_anim_data.py` → `FileNotFoundError`.

- [ ] **Step 3: Write the loader**

```python
"""Everything the W2 knee scene draws, read once from analysis/tables (no manim import)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slo_lab.plots import attainment_series, tpot_series

CELLS = ("bf16", "fp8", "awq", "gptq")
LABELS = {"bf16": "BF16", "fp8": "FP8", "awq": "AWQ", "gptq": "GPTQ-Int4"}
OPEN_TABLES = {c: f"w2-{c}-open-loop" for c in CELLS}
CLOSED_TABLES = {
    "bf16": "w2-bf16-closed-loop",
    "fp8": "w2-fp8-closed-loop-v3",
    "awq": "w2-awq-closed-loop",
    "gptq": "w2-gptq-closed-loop",
}


@dataclass(frozen=True)
class Cell:
    name: str
    label: str
    attainment: list[tuple[float, float]]  # (offered rps, min attainment over seeds)
    tpot_p95_ms: list[tuple[float, float]]  # (offered rps, mean over seeds)
    ttft_p95_s: list[tuple[float, float]]
    r_slo: float
    r_sat: float
    wh_per_m_tokens: float
    tmmlu_accuracy: float
    tmmlu_delta: float | None  # vs BF16, None for the baseline
    mcnemar_p: float | None


@dataclass(frozen=True)
class KneeData:
    cells: dict[str, Cell]
    sensitivity_fp8: dict[tuple[float, float], float]  # (ttft_s, tpot_s) -> r_slo
    rate_max: float


def _json(root: Path, table: str, name: str) -> dict[str, Any]:
    return json.loads((root / "analysis" / "tables" / table / name).read_text(encoding="utf-8"))


def _ttft_series(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    by_rate: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if not r.get("suspect") and r.get("ttft_p95_s") is not None:
            by_rate[float(r["offered_rps"])].append(float(r["ttft_p95_s"]))
    return [(rate, statistics.mean(v)) for rate, v in sorted(by_rate.items())]


def _wh_per_m_tokens(rows: list[dict[str, Any]], r_slo: float) -> float:
    at = [float(r["tok_per_wh"]) for r in rows if abs(float(r["offered_rps"]) - r_slo) < 0.01]
    return 1e6 / statistics.mean(at)


def _sensitivity(per_cell: dict[str, Any]) -> dict[tuple[float, float], float]:
    out: dict[tuple[float, float], float] = {}
    for key, value in per_cell["sensitivity"]["r_slo_by_threshold"].items():
        ttft, tpot = (float(part.split("=")[1]) for part in key.split("|"))
        out[(ttft, tpot)] = float(value)
    return out


def load_knee(root: Path) -> KneeData:
    quality = {r["b"]: r for r in _json(root, "w2-quality-paired", "paired.json")["rows"]}
    baseline_acc = next(iter(quality.values()))["a_accuracy"]
    cells: dict[str, Cell] = {}
    sensitivity: dict[tuple[float, float], float] = {}
    rate_max = 0.0
    for name in CELLS:
        table_dir = root / "analysis" / "tables" / OPEN_TABLES[name]
        open_json = _json(root, OPEN_TABLES[name], "open_loop.json")
        per_cell = open_json["per_cell"][name]
        att = next(s for s in attainment_series([table_dir]) if s.name == name).points
        tpot = next(s for s in tpot_series([table_dir]) if s.name == name).points
        rows = open_json["rows"]
        r_slo = float(per_cell["r_slo"])
        closed = _json(root, CLOSED_TABLES[name], "closed_loop.json")["per_cell"][name]
        q = quality.get(name)
        cells[name] = Cell(
            name,
            LABELS[name],
            att,
            tpot,
            _ttft_series(rows),
            r_slo,
            float(closed["r_sat_rps"]),
            _wh_per_m_tokens(rows, r_slo),
            float(q["b_accuracy"]) if q else float(baseline_acc),
            float(q["delta_b_minus_a"]) if q else None,
            float(q["mcnemar_exact_p"]) if q else None,
        )
        rate_max = max(rate_max, max(r for r, _ in att))
        if name == "fp8":
            sensitivity = _sensitivity(per_cell)
    return KneeData(cells, sensitivity, rate_max)
```

- [ ] **Step 4: Run the test to verify it passes** — `uv run ruff format scripts/manim/w2_data.py tests/test_w2_anim_data.py && uv run ruff check . && uv run pytest -q tests/test_w2_anim_data.py` → 2 passed.

- [ ] **Step 5: Commit** — `git add scripts/manim/w2_data.py tests/test_w2_anim_data.py` and commit as "scripts/manim/w2_data: loader for the W2 knee animation (curves via slo_lab.plots, r_SLO, r_sat, energy, TMMLU+, sensitivity); CI test pins it to the tables".

---

### Task 3: `w4_data.py` loader with CI test

**Files:**
- Create: `scripts/manim/w4_data.py`
- Test: `tests/test_w4_anim_data.py`

**Interfaces:**
- Produces:
  - `CELLS = (("fp8", "fp8-ngram", "8B FP8 + n-gram"), ("q4b", "q4b-eagle3", "4B + EAGLE-3"), ("q4b", "q4b-ngram", "4B + n-gram"))`, `CONCURRENCIES = (1, 8, 32, 128, 256)`
  - `OpenPoint(offered_rps, tpot_p50_diff_ms, tpot_p95_diff_ms, attainment_diff, p50_consistent, p95_consistent)`
  - `SpecCell(family, name, label, acceptance, mean_acceptance_length, r_sat, base_r_sat, closed: dict[int, float | None], open: list[OpenPoint])`
  - `SpecData(cells: list[SpecCell])`, `load_specdec(root) -> SpecData`

- [ ] **Step 1: Write the failing test**

```python
"""The W4 spec-decode animation's loader reads exactly the committed tables."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("w4_data", REPO / "scripts" / "manim" / "w4_data.py")
assert _spec is not None and _spec.loader is not None
w4_data = importlib.util.module_from_spec(_spec)
sys.modules["w4_data"] = w4_data
_spec.loader.exec_module(w4_data)


@pytest.fixture(scope="module")
def spec():
    return w4_data.load_specdec(REPO)


def test_acceptance_closed_loop_and_r_sat(spec) -> None:
    by = {c.name: c for c in spec.cells}
    assert list(by) == ["fp8-ngram", "q4b-eagle3", "q4b-ngram"]
    assert by["fp8-ngram"].acceptance == pytest.approx(0.5197, abs=1e-3)
    assert by["q4b-eagle3"].acceptance == pytest.approx(0.2654, abs=1e-3)
    assert by["q4b-ngram"].mean_acceptance_length == pytest.approx(2.599, abs=1e-3)
    assert by["fp8-ngram"].closed[1] == pytest.approx(1.724, abs=1e-3)
    assert by["fp8-ngram"].closed[128] == pytest.approx(0.85, abs=1e-3)
    assert by["fp8-ngram"].closed[256] is None  # paged, excluded by the suspect rule
    assert by["q4b-eagle3"].closed[256] == pytest.approx(0.604, abs=1e-3)
    assert set(by["q4b-ngram"].closed) == set(w4_data.CONCURRENCIES)
    assert by["fp8-ngram"].r_sat == pytest.approx(26.67, abs=0.01)
    assert by["fp8-ngram"].base_r_sat == pytest.approx(40.17, abs=0.01)
    assert by["q4b-ngram"].base_r_sat == pytest.approx(47.62, abs=0.01)


def test_open_loop_points_stop_at_half_of_the_base_r_sat(spec) -> None:
    by = {c.name: c for c in spec.cells}
    fp8 = by["fp8-ngram"].open
    assert [p.offered_rps for p in fp8] == [4.02, 10.04, 20.09]
    assert fp8[2].tpot_p95_diff_ms == pytest.approx(3.81, abs=0.01)
    assert fp8[0].tpot_p50_diff_ms == pytest.approx(-3.56, abs=0.01)
    assert all(abs(p.attainment_diff) < 0.002 for p in fp8)
    eagle = by["q4b-eagle3"].open
    assert [p.offered_rps for p in eagle] == [4.76, 11.9, 23.81]
    assert eagle[0].tpot_p95_diff_ms == pytest.approx(-2.21, abs=0.01)
    assert eagle[2].tpot_p95_diff_ms == pytest.approx(10.2, abs=0.01)
    assert eagle[2].p95_consistent is True
```

- [ ] **Step 2: Run it to verify it fails.**

- [ ] **Step 3: Write the loader**

```python
"""Everything the W4 spec-decode scene draws, read once from analysis/tables (no manim import)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CELLS = (
    ("fp8", "fp8-ngram", "8B FP8 + n-gram"),
    ("q4b", "q4b-eagle3", "4B + EAGLE-3"),
    ("q4b", "q4b-ngram", "4B + n-gram"),
)
CONCURRENCIES = (1, 8, 32, 128, 256)
OPEN_MAX_MULTIPLIER = 0.5  # paired open-loop points are shown up to 0.5 x the base r_sat


@dataclass(frozen=True)
class OpenPoint:
    offered_rps: float
    tpot_p50_diff_ms: float
    tpot_p95_diff_ms: float
    attainment_diff: float
    p50_consistent: bool
    p95_consistent: bool


@dataclass(frozen=True)
class SpecCell:
    family: str
    name: str
    label: str
    acceptance: float
    mean_acceptance_length: float
    r_sat: float
    base_r_sat: float
    closed: dict[int, float | None]  # concurrency -> rps ratio vs none (None: excluded)
    open: list[OpenPoint]


@dataclass(frozen=True)
class SpecData:
    cells: list[SpecCell]


def _family(root: Path, family: str) -> dict[str, Any]:
    path = root / "analysis" / "tables" / f"w4-{family}-specdec" / "specdec.json"
    return json.loads(path.read_text(encoding="utf-8"))["families"][family]


def _open_points(cell: dict[str, Any], base_r_sat: float) -> list[OpenPoint]:
    out: list[OpenPoint] = []
    for rate, v in sorted(cell["open_paired_by_rate"].items(), key=lambda kv: float(kv[0])):
        offered = float(v["offered_rps"])
        if offered > OPEN_MAX_MULTIPLIER * base_r_sat * 1.02:
            continue
        p50, p95, att = v["tpot_p50_diff_s"], v["tpot_p95_diff_s"], v["attainment_diff"]
        if p50["mean"] is None or p95["mean"] is None:
            continue
        out.append(
            OpenPoint(
                offered,
                1000.0 * float(p50["mean"]),
                1000.0 * float(p95["mean"]),
                float(att["mean"] or 0.0),
                bool(p50["consistent_sign"]),
                bool(p95["consistent_sign"]),
            )
        )
    return out


def load_specdec(root: Path) -> SpecData:
    cells: list[SpecCell] = []
    for family, name, label in CELLS:
        fam = _family(root, family)
        base = fam["cells"][fam["base_cell"]]
        cell = fam["cells"][name]
        closed: dict[int, float | None] = {c: None for c in CONCURRENCIES}
        for e in cell["closed_paired"]:
            if e.get("rps_ratio") is not None and int(e["concurrency"]) in closed:
                closed[int(e["concurrency"])] = float(e["rps_ratio"])
        cells.append(
            SpecCell(
                family,
                name,
                label,
                float(cell["acceptance_rate_mean"]),
                float(cell["mean_acceptance_length_mean"]),
                float(cell["r_sat_rps"]),
                float(base["r_sat_rps"]),
                closed,
                _open_points(cell, float(base["r_sat_rps"])),
            )
        )
    return SpecData(cells)
```

- [ ] **Step 4: Run the test to verify it passes.** If `fp8-ngram`'s open points differ (the table has exactly 4.02 / 10.04 / 20.09 below 0.5 × 40.17), print them and fix the expectation to the table.

- [ ] **Step 5: Commit** — "scripts/manim/w4_data: loader for the W4 spec-decode animation (acceptance, closed-loop rps ratios, paired open-loop TPOT diffs up to 0.5 x base r_sat); CI test pins it to the tables".

---

### Task 4: Scene `W2Knee`

**Files:**
- Create: `scripts/manim/w2_knee.py`

- [ ] **Step 1: Write the scene**

```python
"""W2 explainer: attainment vs offered rate for four precisions, the FP8 knee is TPOT-bound,
the SLO sensitivity grid, and the four-precision scoreboard (ADR 0007-0009). Illustration
only: every number comes from ``w2_data``.

Render: .venv-manim/Scripts/manim -qh --disable_caching scripts/manim/w2_knee.py W2Knee
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    Scene,
    Table,
    Text,
    VGroup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    AMBER,
    BG,
    BLUE,
    GREEN,
    GREY,
    INK,
    MUTED,
    ORANGE,
    RED,
    bottom_note,
    source_card,
    text,
)
from w2_data import CELLS, KneeData, load_knee

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"bf16": GREY, "fp8": BLUE, "awq": ORANGE, "gptq": AMBER}
NOTE = [
    "--gpu-memory-utilization 0.82，WSL2，與桌面共用的一張 RTX 4090，Qwen3-8B，108 → 132 tokens，"
    "open-loop Poisson 每點 5 min × 3 seeds",
]


def _axes(x_max: float, y_max: float, y_step: float, y_decimals: int) -> Axes:
    return Axes(
        x_range=[0, x_max, 10],
        y_range=[0, y_max, y_step],
        x_length=10.5,
        y_length=4.6,
        axis_config={
            "include_numbers": True,
            "label_constructor": Text,
            "font_size": 18,
            "color": MUTED,
            "decimal_number_config": {"num_decimal_places": 0},
        },
        y_axis_config={"decimal_number_config": {"num_decimal_places": y_decimals}},
    ).shift(DOWN * 0.2)


class W2Knee(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG
        data = load_knee(ROOT)
        self.add(bottom_note(NOTE))
        self.knee(data)
        self.clear()
        self.add(bottom_note(NOTE))
        self.tpot_bound(data)
        self.clear()
        self.add(bottom_note(NOTE))
        self.sensitivity(data)
        self.clear()
        self.add(bottom_note(NOTE))
        self.scoreboard(data)
        self.clear()
        self.play(
            FadeIn(
                source_card(
                    [
                        "示意動畫：數字出自 analysis/tables/w2-*-open-loop、w2-*-closed-loop、w2-quality-paired",
                        "由 make reproduce 從提交的證據重建；ADR 0007、0008、0009",
                        "github.com/kuotunyu/vllm-single-gpu-slo-lab",
                    ]
                )
            ),
            run_time=0.8,
        )
        self.wait(2.5)

    # ---- 1. attainment vs rate, four precisions (20 s)
    def knee(self, data: KneeData) -> None:
        head = text(
            "SLO attainment 對 offered rate（每個 rate 取 3 seeds 的最小值）", 28, weight="BOLD"
        )
        head.to_edge(UP, buff=0.35)
        axes = _axes(90, 1.0, 0.25, 2)
        x_lab = text("offered rate（req/s）", 18, MUTED).next_to(axes.x_axis, DOWN, buff=0.35)
        y_lab = text("attainment", 18, MUTED).next_to(axes.y_axis, UP, buff=0.15)
        target = DashedLine(axes.c2p(0, 0.95), axes.c2p(90, 0.95), color=MUTED, dash_length=0.12)
        target_lab = text("95 %", 16, MUTED).next_to(target, RIGHT, buff=0.1)
        self.play(FadeIn(head), Create(axes), FadeIn(x_lab), FadeIn(y_lab), run_time=1.0)
        self.play(Create(target), FadeIn(target_lab), run_time=0.5)
        legend = VGroup()
        for name in CELLS:
            cell = data.cells[name]
            xs = [r for r, _ in cell.attainment]
            ys = [a for _, a in cell.attainment]
            graph = axes.plot_line_graph(
                xs,
                ys,
                line_color=COLORS[name],
                add_vertex_dots=True,
                vertex_dot_radius=0.045,
                vertex_dot_style={"fill_color": COLORS[name]},
                stroke_width=3,
            )
            self.play(Create(graph["line_graph"]), FadeIn(graph["vertex_dots"]), run_time=2.2)
            marker = DashedLine(
                axes.c2p(cell.r_slo, 0), axes.c2p(cell.r_slo, 0.95), color=COLORS[name]
            )
            tag = text(f"r_SLO {cell.r_slo:g}", 18, COLORS[name]).next_to(marker, UP, buff=0.05)
            self.play(Create(marker), FadeIn(tag), run_time=0.6)
            item = text(f"{cell.label}  r_SLO {cell.r_slo:g} rps", 20, COLORS[name])
            legend.add(item)
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(
                UP + RIGHT, buff=0.5
            ).shift(DOWN * 0.7)
            self.play(FadeIn(item), run_time=0.3)
        self.wait(2.0)

    # ---- 2. the FP8 knee is TPOT-bound (12 s)
    def tpot_bound(self, data: KneeData) -> None:
        fp8 = data.cells["fp8"]
        head = text("FP8 的膝點是 TPOT，不是排隊", 28, weight="BOLD").to_edge(UP, buff=0.35)
        axes = _axes(50, 100, 25, 0)
        x_lab = text("offered rate（req/s）", 18, MUTED).next_to(axes.x_axis, DOWN, buff=0.35)
        y_lab = text("TPOT p95（ms，3 seeds 平均）", 18, MUTED).next_to(axes.y_axis, UP, buff=0.15)
        pts = [(r, t) for r, t in fp8.tpot_p95_ms if r <= 50]
        graph = axes.plot_line_graph(
            [r for r, _ in pts],
            [t for _, t in pts],
            line_color=BLUE,
            add_vertex_dots=True,
            vertex_dot_radius=0.045,
            vertex_dot_style={"fill_color": BLUE},
            stroke_width=3,
        )
        slo = DashedLine(axes.c2p(0, 50), axes.c2p(50, 50), color=RED, dash_length=0.12)
        slo_lab = text("SLO 50 ms", 16, RED).next_to(slo, RIGHT, buff=0.1)
        self.play(FadeIn(head), Create(axes), FadeIn(x_lab), FadeIn(y_lab), run_time=1.0)
        self.play(Create(slo), FadeIn(slo_lab), run_time=0.5)
        self.play(Create(graph["line_graph"]), FadeIn(graph["vertex_dots"]), run_time=2.5)
        marker = DashedLine(axes.c2p(fp8.r_slo, 0), axes.c2p(fp8.r_slo, 50), color=BLUE)
        self.play(
            Create(marker),
            FadeIn(text(f"r_SLO {fp8.r_slo:g}", 18, BLUE).next_to(marker, UP, buff=0.05)),
            run_time=0.6,
        )
        ttft_max = max(t for r, t in fp8.ttft_p95_s if r <= 30.57)
        lines = [
            f"26 到 31 rps 之間 TPOT p95 從 30 ms 爬過 50 ms；同一段 TTFT p95 最高 {ttft_max:.2f} s（門檻 1 s）",
            "容量由 decode 的步長決定：這張卡的膝點是 TPOT，不是佇列",
        ]
        note = (
            VGroup(*[text(s, 22) for s in lines])
            .arrange(DOWN, buff=0.12)
            .next_to(axes, DOWN, buff=0.75)
        )
        self.play(FadeIn(note), run_time=0.6)
        self.wait(3.5)

    # ---- 3. sensitivity grid (10 s)
    def sensitivity(self, data: KneeData) -> None:
        head = text("同一份紀錄換 SLO 門檻重算 r_SLO（FP8）", 28, weight="BOLD").to_edge(
            UP, buff=0.35
        )
        ttfts, tpots = (0.5, 1.0, 2.0), (0.03, 0.05, 0.1)
        base = data.sensitivity_fp8[(1.0, 0.05)]
        rows = [[f"{data.sensitivity_fp8[(a, b)]:g}" for b in tpots] for a in ttfts]
        table = (
            Table(
                rows,
                row_labels=[text(f"TTFT ≤ {a:g} s", 22, MUTED) for a in ttfts],
                col_labels=[text(f"TPOT ≤ {int(b * 1000)} ms", 22, MUTED) for b in tpots],
                top_left_entry=text("r_SLO（rps）", 20, MUTED),
                element_to_mobject=lambda s: text(s, 26),
                include_outer_lines=False,
                line_config={"stroke_color": MUTED, "stroke_width": 1},
                h_buff=0.7,
                v_buff=0.4,
            )
            .scale(0.85)
            .next_to(head, DOWN, buff=0.5)
        )
        for i, a in enumerate(ttfts):
            for j, b in enumerate(tpots):
                v = data.sensitivity_fp8[(a, b)]
                if v != base:
                    table.get_entries((i + 2, j + 2)).set_color(RED if v < base else GREEN)
        self.play(
            FadeIn(head),
            Create(table.get_horizontal_lines()),
            Create(table.get_vertical_lines()),
            run_time=0.8,
        )
        self.play(FadeIn(table.get_labels()), run_time=0.4)
        for i in range(3):
            self.play(FadeIn(table.get_rows()[i + 1][1:]), run_time=0.5)
        note = text(
            "r_SLO 對 TPOT 門檻敏感（30 ms → 24.0），對 TTFT 門檻不敏感（0.5–2 s 都一樣）", 22
        )
        note.next_to(table, DOWN, buff=0.6)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(3.5)

    # ---- 4. scoreboard (10 s)
    def scoreboard(self, data: KneeData) -> None:
        head = text(
            "四精度總表（SLO：TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms）", 28, weight="BOLD"
        ).to_edge(UP, buff=0.35)
        rows = []
        for name in CELLS:
            c = data.cells[name]
            delta = (
                "—"
                if c.tmmlu_delta is None
                else f"{100 * c.tmmlu_delta:+.2f} pts（p = {c.mcnemar_p:.3g}）"
            )
            rows.append(
                [
                    c.label,
                    f"{c.r_slo:g}",
                    f"{c.r_sat:.2f}",
                    f"{c.wh_per_m_tokens:.1f}",
                    f"{c.tmmlu_accuracy:.4f}",
                    delta,
                ]
            )
        table = (
            Table(
                rows,
                col_labels=[
                    text(s, 20, MUTED)
                    for s in [
                        "精度",
                        "r_SLO（rps）",
                        "r_sat（rps）",
                        "Wh／百萬 token",
                        "TMMLU+",
                        "對 BF16 配對差",
                    ]
                ],
                element_to_mobject=lambda s: text(s, 22),
                include_outer_lines=False,
                line_config={"stroke_color": MUTED, "stroke_width": 1},
                h_buff=0.5,
                v_buff=0.35,
            )
            .scale(0.78)
            .next_to(head, DOWN, buff=0.5)
        )
        for i, name in enumerate(CELLS):
            table.get_rows()[i + 1][0].set_color(COLORS[name])
        self.play(
            FadeIn(head),
            Create(table.get_horizontal_lines()),
            Create(table.get_vertical_lines()),
            run_time=0.8,
        )
        self.play(FadeIn(table.get_col_labels()), run_time=0.4)
        for i in range(4):
            self.play(FadeIn(table.get_rows()[i + 1]), run_time=0.5)
        note = text(
            "FP8：SLO 容量是 BF16 的 2.55 倍、每 token 能耗 42 %、品質無法區分", 22
        ).next_to(table, DOWN, buff=0.6)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(3.5)
```

- [ ] **Step 2: Low-quality render, inspect frames every 6 s, fix layout.** Run `-ql`, extract with `ffmpeg -vf fps=1/6`, Read the PNGs. Typical fixes: legend overlapping curves (move to the lower-right), table too wide (scale), note overlapping the axes.

- [ ] **Step 3: Lint and commit** — "scripts/manim/w2_knee: W2 explainer scene (knee curves, TPOT-bound knee, sensitivity grid, scoreboard)".

---

### Task 5: Scene `W4Specdec`

**Files:**
- Create: `scripts/manim/w4_specdec.py`

- [ ] **Step 1: Write the scene**

```python
"""W4 explainer: acceptance, closed-loop throughput ratio by concurrency, paired open-loop
TPOT differences, and the conclusion (ADR 0015-0016). Illustration only: every number comes
from ``w4_data``.

Render: .venv-manim/Scripts/manim -qh --disable_caching scripts/manim/w4_specdec.py W4Specdec
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    FadeIn,
    GrowFromEdge,
    Line,
    Rectangle,
    Scene,
    VGroup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import BG, BLUE, GREEN, INK, MUTED, ORANGE, RED, bottom_note, source_card, text
from w4_data import CONCURRENCIES, SpecData, load_specdec

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"fp8-ngram": BLUE, "q4b-eagle3": ORANGE, "q4b-ngram": GREEN}
NOTE = [
    "Shakespeare 自然文字 prompt 108 → 132 tokens，Qwen3 預設取樣，全部經 passthrough shim，"
    "--gpu-memory-utilization 0.82，WSL2 上與桌面共用的一張 RTX 4090",
]


class W4Specdec(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG
        data = load_specdec(ROOT)
        self.add(bottom_note(NOTE))
        self.acceptance(data)
        self.clear()
        self.add(bottom_note(NOTE))
        self.closed_loop(data)
        self.clear()
        self.add(bottom_note(NOTE))
        self.open_loop(data)
        self.clear()
        self.add(bottom_note(NOTE))
        self.conclusion(data)
        self.clear()
        self.play(
            FadeIn(
                source_card(
                    [
                        "示意動畫：數字出自 analysis/tables/w4-fp8-specdec 與 w4-q4b-specdec",
                        "由 make reproduce 從提交的證據重建；ADR 0015、0016",
                        "github.com/kuotunyu/vllm-single-gpu-slo-lab",
                    ]
                )
            ),
            run_time=0.8,
        )
        self.wait(2.5)

    # ---- 1. mechanism and acceptance (10 s)
    def acceptance(self, data: SpecData) -> None:
        head = text(
            "speculative decoding：每步先猜 3 個 token，目標模型一次驗證，猜錯的丟掉",
            26,
            weight="BOLD",
        )
        head.to_edge(UP, buff=0.35)
        sub = text(
            "接受率 = 被接受的 draft token ÷ 猜的 token；平均接受長度 = 每步實際產出的 token 數",
            20,
            MUTED,
        )
        sub.next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(head), FadeIn(sub), run_time=0.8)
        rows = VGroup()
        for i, cell in enumerate(data.cells):
            y = 1.2 - 1.3 * i
            label = (
                text(cell.label, 24, COLORS[cell.name])
                .move_to([0, y, 0])
                .align_to([-6.6, 0, 0], LEFT)
            )
            bar = Rectangle(
                width=6.0 * cell.acceptance,
                height=0.45,
                fill_color=COLORS[cell.name],
                fill_opacity=0.9,
                stroke_width=0,
            )
            bar.align_to([-2.6, 0, 0], LEFT).set_y(y)
            val = text(
                f"接受率 {cell.acceptance:.2f} · 平均接受長度 {cell.mean_acceptance_length:.2f} token／步",
                22,
            ).next_to(bar, RIGHT, buff=0.2)
            self.play(FadeIn(label), run_time=0.3)
            self.play(GrowFromEdge(bar, LEFT), run_time=0.8)
            self.play(FadeIn(val), run_time=0.3)
            rows.add(label, bar, val)
        frame = Line([-2.6, 1.6, 0], [-2.6, -1.7, 0], color=MUTED, stroke_width=1)
        one = Line([3.4, 1.6, 0], [3.4, -1.7, 0], color=MUTED, stroke_width=1)
        self.play(
            Create(frame),
            Create(one),
            FadeIn(text("0", 16, MUTED).next_to(frame, DOWN, buff=0.1)),
            FadeIn(text("1.0", 16, MUTED).next_to(one, DOWN, buff=0.1)),
            run_time=0.4,
        )
        self.wait(2.5)

    # ---- 2. closed-loop throughput ratio by concurrency (15 s)
    def closed_loop(self, data: SpecData) -> None:
        head = text(
            "closed-loop 吞吐：加速 cell ÷ 同 family 的 none cell（rps 比）", 26, weight="BOLD"
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(head), run_time=0.6)
        base_y, unit = -1.6, 1.9  # 1.0x = 1.9 scene units
        group_x = [-4.8, -2.4, 0.0, 2.4, 4.8]
        bar_w, gap = 0.5, 0.08
        axis = Line([-6.4, base_y, 0], [6.4, base_y, 0], color=MUTED, stroke_width=1)
        one = Line([-6.4, base_y + unit, 0], [6.4, base_y + unit, 0], color=MUTED, stroke_width=1)
        one_lab = (
            text("1.0×（與 none 相同）", 16, MUTED)
            .next_to(one, UP, buff=0.05)
            .align_to([-6.4, 0, 0], LEFT)
        )
        self.play(Create(axis), Create(one), FadeIn(one_lab), run_time=0.6)
        legend = VGroup(*[text(c.label, 18, COLORS[c.name]) for c in data.cells]).arrange(
            RIGHT, buff=0.6
        )
        legend.next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(legend), run_time=0.4)
        for gx, conc in zip(group_x, CONCURRENCIES, strict=True):
            self.play(FadeIn(text(f"c = {conc}", 20).move_to([gx, base_y - 0.35, 0])), run_time=0.2)
            for k, cell in enumerate(data.cells):
                x = gx + (k - 1) * (bar_w + gap)
                ratio = cell.closed[conc]
                if ratio is None:
                    self.play(
                        FadeIn(text("分頁", 16, RED).move_to([x, base_y + 0.25, 0])), run_time=0.3
                    )
                    continue
                bar = Rectangle(
                    width=bar_w,
                    height=unit * ratio,
                    fill_color=COLORS[cell.name],
                    fill_opacity=0.9,
                    stroke_width=0,
                )
                bar.move_to([x, base_y + unit * ratio / 2, 0])
                val = text(f"{ratio:.2f}", 16, INK).next_to(bar, UP, buff=0.05)
                self.play(GrowFromEdge(bar, DOWN), FadeIn(val), run_time=0.45)
        note = text(
            "單流與小批次變快（1.12 到 1.72 倍），c = 128 起吞吐反轉為 0.6 到 0.85 倍", 22
        ).next_to(head, DOWN, buff=0.8)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(3.0)

    # ---- 3. open-loop paired differences (15 s)
    def open_loop(self, data: SpecData) -> None:
        head = text(
            "open-loop 同 rate、同 seed 配對：TPOT 中位數與 p95 對 none 的差（ms）",
            26,
            weight="BOLD",
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(head), run_time=0.6)
        panel_x = [-4.6, 0.0, 4.6]
        scale = 0.14  # scene units per ms
        for px, cell in zip(panel_x, data.cells, strict=True):
            title = text(cell.label, 22, COLORS[cell.name]).move_to([px, 2.3, 0])
            base_y = 0.0
            axis = Line([px - 1.9, base_y, 0], [px + 1.9, base_y, 0], color=MUTED, stroke_width=1)
            self.play(FadeIn(title), Create(axis), run_time=0.3)
            for j, p in enumerate(cell.open):
                x = px - 1.3 + j * 1.3
                for dx, val, color in (
                    (-0.22, p.tpot_p50_diff_ms, GREEN if p.tpot_p50_diff_ms < 0 else RED),
                    (0.22, p.tpot_p95_diff_ms, RED if p.tpot_p95_diff_ms > 0 else GREEN),
                ):
                    h = abs(val) * scale
                    bar = Rectangle(
                        width=0.36,
                        height=max(h, 0.02),
                        fill_color=color,
                        fill_opacity=0.9,
                        stroke_width=0,
                    )
                    bar.move_to([x + dx, base_y + (h / 2 if val >= 0 else -h / 2), 0])
                    lab = text(f"{val:+.1f}", 14, INK).next_to(
                        bar, UP if val >= 0 else DOWN, buff=0.04
                    )
                    self.play(
                        GrowFromEdge(bar, DOWN if val >= 0 else UP), FadeIn(lab), run_time=0.25
                    )
                self.play(
                    FadeIn(
                        text(f"{p.offered_rps:g} rps", 14, MUTED).move_to([x, base_y - 1.55, 0])
                    ),
                    run_time=0.15,
                )
            self.play(
                FadeIn(text("attainment 差 0", 16, MUTED).move_to([px, -2.0, 0])), run_time=0.2
            )
        key = text(
            "左 = 中位數差，右 = p95 差；綠 = 變快，紅 = 變慢；三個 seed 同號", 18, MUTED
        ).next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(key), run_time=0.4)
        self.wait(3.0)

    # ---- 4. conclusion (8 s)
    def conclusion(self, data: SpecData) -> None:
        lines = [
            "單流與小批次變快；c ≥ 128 吞吐反轉；同 rate 的中位數變快、p95 變慢",
            "以 p95 定義的 SLO 下，attainment 與粗網格 r_SLO 都和 none 相同：容量沒有增加",
        ]
        head = (
            VGroup(*[text(s, 26, weight="BOLD") for s in lines])
            .arrange(DOWN, buff=0.2)
            .to_edge(UP, buff=0.6)
        )
        self.play(FadeIn(head), run_time=0.8)
        rows = VGroup()
        for i, cell in enumerate(data.cells):
            line = text(
                f"{cell.label}：r_sat {cell.r_sat:.1f} rps（none {cell.base_r_sat:.1f}，{cell.r_sat / cell.base_r_sat:.2f}×）",
                24,
                COLORS[cell.name],
            )
            rows.add(line)
        rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT).next_to(head, DOWN, buff=0.8)
        for r in rows:
            self.play(FadeIn(r), run_time=0.4)
        self.wait(3.5)
```

- [ ] **Step 2: Low-quality render, inspect frames, fix layout.**

- [ ] **Step 3: Lint and commit** — "scripts/manim/w4_specdec: W4 explainer scene (acceptance, closed-loop ratio by concurrency, paired open-loop TPOT diffs, conclusion)".

---

### Task 6: Render, GIFs, README and docs, CI

- [ ] **Step 1: Render both at `-qh`, re-encode to 1080p30, build GIFs** (same ffmpeg commands as the W3 README; outputs `docs/media/w2-knee.{mp4,gif}`, `docs/media/w4-specdec.{mp4,gif}`); check sizes ≤ 8 MB.
- [ ] **Step 2: README** — after the W2 paragraph ending "4090 不計 $／百萬 token（ADR 0010）。" insert the W2 GIF and a one-line caption; after the W4 paragraph ending "高負載點量不到（ADR 0016）。" insert the W4 GIF and caption (same wording pattern as the W3 caption).
- [ ] **Step 3: `scripts/manim/README.md`** — list the three scenes and their check-lists; `evidence/README.md` sentence mentions three animations; HANDOFF row updated; W4 plan run log row.
- [ ] **Step 4: Verify** — `uv run ruff check . && uv run ruff format --check . && uv run pytest -q` (exit codes), `uv run python scripts/redact.py audit .`, `make reproduce`; commit; push; `gh run watch`.
- [ ] **Step 5: Control tower ledger line; memory.**

## Self-review

- Spec coverage: style module (T1), loaders + tests (T2, T3), scenes 2 and 3 (T4, T5), outputs/README/docs/CI (T6).
- Types: `KneeData.cells[name].attainment` list of (rate, value) used by `plot_line_graph`; `SpecCell.closed[conc]` may be None → "分頁"; `OpenPoint` fields as named.
- No placeholders; every number drawn comes from the loaders except fixed thresholds, the 2.55×/42 % sentence (README numbers, ADR 0009) and the axis limits.
