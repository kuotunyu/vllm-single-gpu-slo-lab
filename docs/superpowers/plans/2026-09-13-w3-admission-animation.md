# W3 Admission Explainer Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ~75 s Manim Community animation of the W3 burst/admission result (three policies replayed, then C = 256 vs C = 192), rendered to `docs/media/` and embedded in the README as an illustration.

**Architecture:** Pure data logic goes into the package (`slo_lab.timeline`, tested in CI). A manim-free loader (`scripts/manim/w3_data.py`) assembles everything the scene draws from the committed evidence and tables. The scene (`scripts/manim/w3_admission.py`) only draws, driven by one `ValueTracker` for the replay cursor. Rendering happens in a separate `.venv-manim`; the outputs are committed as binaries and never touched by `make reproduce`.

**Tech Stack:** Python 3.12, uv, `manim==0.21.0` (Manim Community; Pango text, no LaTeX), ffmpeg at `D:\ffmpeg\bin` for the GIF, existing `slo_lab.slo` / `slo_lab.admission_analysis` / `slo_lab.stats`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-13-w3-admission-animation-design.md`.
- The animation is an illustration: not in `make reproduce`, not in CI, no new claims-audit rows; every number it shows must equal a value in `analysis/tables/w3-fp8-admission/` or `w3-fp8-c192-admission/` (ADR 0013, 0018).
- `pyproject.toml` dependencies unchanged; manim lives only in `.venv-manim` and `scripts/manim/requirements.txt` (`manim==0.21.0`).
- Repo conventions: commits as `git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com`, `GIT_ASK_YESNO=false`, stdin closed, no Co-Authored-By; before every commit `uv run ruff format`, `uv run ruff check .`, the tests, and `uv run python scripts/redact.py audit <dir>` checked by exit code (never `| tail`).
- Files with backslashes are written with the Write/Edit tools; Python writes use `newline="\n"`.
- Text in the animation: Traditional Chinese with English terms, font `Microsoft JhengHei`; deep background `#101418`; policy colours passthrough `#e4572e`, hard_cap `#4c9be8`, bounded_queue `#7bc96f`; SLO breach red `#ff4d4d`, pass green `#7bc96f`.
- Video 1920 × 1080 @ 30 fps (`-qh`); GIF 640 px wide, 12 fps, ≤ 8 MB or cut to the replay segment.

---

### Task 1: `slo_lab.timeline` (buckets, rolling merge, resampling, phases)

**Files:**
- Create: `src/slo_lab/timeline.py`
- Test: `tests/test_timeline.py`

**Interfaces:**
- Consumes: `slo_lab.slo.RequestRecord`, `Outcome`, `Slo`, `DEFAULT_SLO`, `meets_slo`; `slo_lab.stats.percentile(values, q)`.
- Produces:
  - `Bucket(start_s, end_s, offered, met, rejected, timeouts, ok_ttfts)` frozen dataclass with properties `attainment -> float | None` and `ttft_p95_s -> float | None`.
  - `Phase(name, start_s, end_s)` frozen dataclass.
  - `bucket_records(records, *, bucket_s=10.0, end_s=None, slo=DEFAULT_SLO) -> list[Bucket]`
  - `rolling(buckets, window) -> list[Bucket]`
  - `downsample(series, step_s, end_s, *, start_s=0.0) -> list[tuple[float, float]]`
  - `phases_of(manifest) -> list[Phase]`

- [x] **Step 1: Write the failing tests**

```python
"""Time-bucketed views of a replayed trace (slo_lab.timeline)."""

from __future__ import annotations

import pytest

from slo_lab.slo import RequestRecord
from slo_lab.timeline import Bucket, Phase, bucket_records, downsample, phases_of, rolling


def _ok(i: int, offered: float, ttft: float = 0.1) -> RequestRecord:
    return RequestRecord(
        request_id=f"r{i}",
        offered_at_s=offered,
        outcome="ok",
        ttft_s=ttft,
        e2e_s=ttft + 1.0,
        output_tokens=132,
    )


def _bad(i: int, offered: float, outcome: str) -> RequestRecord:
    return RequestRecord(request_id=f"r{i}", offered_at_s=offered, outcome=outcome)


RECORDS = [
    _ok(0, 1.0, 0.1),
    _ok(1, 2.0, 0.2),
    _ok(2, 5.0, 2.0),  # TTFT above the 1 s threshold: offered, not met
    _bad(3, 12.0, "rejected_429"),
    _bad(4, 15.0, "timeout"),
    _ok(5, 25.0, 0.3),
]


def test_bucket_records_counts_attainment_and_ttft() -> None:
    b = bucket_records(RECORDS, bucket_s=10.0, end_s=30.0)
    assert [x.start_s for x in b] == [0.0, 10.0, 20.0]
    assert (b[0].offered, b[0].met, b[0].rejected, b[0].timeouts) == (3, 2, 0, 0)
    assert b[0].attainment == pytest.approx(2 / 3)
    assert b[0].ttft_p95_s == pytest.approx(1.82)  # linear-interpolation p95 of [0.1, 0.2, 2.0]
    assert (b[1].offered, b[1].met, b[1].rejected, b[1].timeouts) == (2, 0, 1, 1)
    assert b[1].attainment == 0.0 and b[1].ttft_p95_s is None
    assert (b[2].offered, b[2].met) == (1, 1)


def test_bucket_records_default_end_and_explicit_cut() -> None:
    assert len(bucket_records(RECORDS, bucket_s=10.0)) == 3  # last offer at 25 s -> end 30 s
    cut = bucket_records(RECORDS, bucket_s=10.0, end_s=20.0)
    assert len(cut) == 2 and sum(x.offered for x in cut) == 5
    empty = bucket_records([], bucket_s=10.0, end_s=30.0)
    assert len(empty) == 3 and empty[0].attainment is None and empty[0].ttft_p95_s is None


def test_bucket_records_rejects_bad_width() -> None:
    with pytest.raises(ValueError):
        bucket_records(RECORDS, bucket_s=0.0)


def test_rolling_merges_a_window_of_buckets() -> None:
    b = bucket_records(RECORDS, bucket_s=10.0, end_s=30.0)
    r = rolling(b, 2)
    assert (r[0].start_s, r[0].end_s, r[0].offered) == (0.0, 10.0, 3)
    assert (r[1].start_s, r[1].end_s, r[1].offered, r[1].met, r[1].rejected) == (0.0, 20.0, 5, 2, 1)
    assert r[1].ttft_p95_s == pytest.approx(1.82)
    assert (r[2].start_s, r[2].end_s, r[2].offered, r[2].met) == (10.0, 30.0, 3, 1)
    assert r[2].ok_ttfts == (0.3,)
    with pytest.raises(ValueError):
        rolling(b, 0)


def test_downsample_holds_the_latest_sample() -> None:
    series = [(7.0, 3.0), (2.0, 1.0), (12.0, 0.0)]
    assert downsample(series, 5.0, 15.0) == [(0.0, 0.0), (5.0, 1.0), (10.0, 3.0), (15.0, 0.0)]
    assert downsample([], 5.0, 10.0) == [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]


def test_phases_of_reads_the_manifest_trace() -> None:
    m = {
        "trace": {
            "phases": [
                {"phase": "pre", "start_s": 0, "end_s": 300},
                {"phase": "burst", "start_s": 300, "end_s": 600},
            ]
        }
    }
    assert phases_of(m) == [Phase("pre", 0.0, 300.0), Phase("burst", 300.0, 600.0)]
    assert phases_of({}) == []
    assert isinstance(Bucket(0.0, 10.0, 0, 0, 0, 0, ()), Bucket)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_timeline.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'slo_lab.timeline'`

- [x] **Step 3: Write the implementation**

```python
"""Time-bucketed views of a replayed trace, for figures and explainer animations.

Pure logic on the canonical per-request records (``slo_lab.slo.RequestRecord``): fixed-width
buckets of offered requests with attainment, TTFT p95 and rejection counts; a rolling merge of
consecutive buckets (so an animated readout does not jump bucket to bucket); a fixed-step
resampling of a queue series; and the trace phases of a stage manifest. No plotting or
animation dependency, so it is tested in CI like the rest of the package.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from slo_lab.slo import DEFAULT_SLO, Outcome, RequestRecord, Slo, meets_slo
from slo_lab.stats import percentile


@dataclass(frozen=True)
class Bucket:
    """Offered requests in ``[start_s, end_s)`` of the trace, by offer time."""

    start_s: float
    end_s: float
    offered: int
    met: int
    rejected: int
    timeouts: int
    ok_ttfts: tuple[float, ...]  # sorted TTFTs of successful requests, kept so buckets can merge

    @property
    def attainment(self) -> float | None:
        return self.met / self.offered if self.offered else None

    @property
    def ttft_p95_s(self) -> float | None:
        return percentile(self.ok_ttfts, 95) if self.ok_ttfts else None


@dataclass(frozen=True)
class Phase:
    name: str
    start_s: float
    end_s: float


def bucket_records(
    records: Iterable[RequestRecord],
    *,
    bucket_s: float = 10.0,
    end_s: float | None = None,
    slo: Slo = DEFAULT_SLO,
) -> list[Bucket]:
    """Fixed-width buckets from 0 to ``end_s`` (default: past the last offer); offers at or
    beyond ``end_s`` are dropped."""
    if bucket_s <= 0:
        raise ValueError(f"bucket_s must be positive, got {bucket_s}")
    recs = list(records)
    if end_s is None:
        last = max((r.offered_at_s for r in recs), default=0.0)
        end_s = (math.floor(last / bucket_s) + 1) * bucket_s
    n = max(1, math.ceil(end_s / bucket_s - 1e-9))
    offered = [0] * n
    met = [0] * n
    rejected = [0] * n
    timeouts = [0] * n
    ttfts: list[list[float]] = [[] for _ in range(n)]
    for r in recs:
        i = int(r.offered_at_s // bucket_s)
        if i < 0 or i >= n:
            continue
        offered[i] += 1
        if meets_slo(r, slo):
            met[i] += 1
        if r.outcome is Outcome.REJECTED_429:
            rejected[i] += 1
        elif r.outcome is Outcome.TIMEOUT:
            timeouts[i] += 1
        if r.outcome is Outcome.OK and r.ttft_s is not None:
            ttfts[i].append(r.ttft_s)
    return [
        Bucket(
            i * bucket_s,
            (i + 1) * bucket_s,
            offered[i],
            met[i],
            rejected[i],
            timeouts[i],
            tuple(sorted(ttfts[i])),
        )
        for i in range(n)
    ]


def rolling(buckets: Sequence[Bucket], window: int) -> list[Bucket]:
    """Bucket ``i`` merged with the ``window - 1`` buckets before it (fewer at the start)."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    out: list[Bucket] = []
    for i, b in enumerate(buckets):
        chunk = buckets[max(0, i - window + 1) : i + 1]
        out.append(
            Bucket(
                chunk[0].start_s,
                b.end_s,
                sum(c.offered for c in chunk),
                sum(c.met for c in chunk),
                sum(c.rejected for c in chunk),
                sum(c.timeouts for c in chunk),
                tuple(sorted(t for c in chunk for t in c.ok_ttfts)),
            )
        )
    return out


def downsample(
    series: Sequence[tuple[float, float]], step_s: float, end_s: float, *, start_s: float = 0.0
) -> list[tuple[float, float]]:
    """``(t, value)`` on a fixed grid; the value is the latest sample at or before ``t``
    (0.0 before the first sample), so an animation can look values up by frame."""
    if step_s <= 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    pts = sorted(series)
    times = [t for t, _ in pts]
    out: list[tuple[float, float]] = []
    k = 0
    while True:
        t = start_s + k * step_s
        if t > end_s + 1e-9:
            break
        i = bisect_right(times, t) - 1
        out.append((round(t, 6), pts[i][1] if i >= 0 else 0.0))
        k += 1
    return out


def phases_of(manifest: dict[str, Any]) -> list[Phase]:
    """The trace phases a stage manifest records (``trace.phases``), in order."""
    trace = manifest.get("trace") or {}
    return [
        Phase(str(p["phase"]), float(p["start_s"]), float(p["end_s"]))
        for p in trace.get("phases") or []
    ]
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run ruff format src/slo_lab/timeline.py tests/test_timeline.py && uv run ruff check . && uv run pytest -q tests/test_timeline.py`
Expected: 6 passed.

- [x] **Step 5: Commit**

```bash
git add src/slo_lab/timeline.py tests/test_timeline.py
GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "slo_lab.timeline: fixed-width buckets (attainment, TTFT p95, rejections), rolling merge, queue resampling, trace phases; 6 tests" < /dev/null
```

---

### Task 2: Manim environment, ignores, render README

**Files:**
- Create: `scripts/manim/requirements.txt`, `scripts/manim/README.md`
- Modify: `.gitignore` (add `.venv-manim/` and `media/` after the `venv/` line)
- Scratch (not committed): a one-frame font check scene under the scratchpad

**Interfaces:**
- Produces: `.venv-manim/Scripts/manim` and `.venv-manim/Scripts/python.exe` with `manim==0.21.0` and this repo installed editable (so the scene can import `slo_lab`).

- [x] **Step 1: Write `scripts/manim/requirements.txt`**

```
manim==0.21.0
```

- [x] **Step 2: Add the ignores**

Append to `.gitignore` after `venv/`:

```
.venv-manim/
media/
```

- [x] **Step 3: Create the environment (no network cost beyond the wheel download; free)**

```bash
uv venv .venv-manim --python 3.12
uv pip install --python .venv-manim/Scripts/python.exe -r scripts/manim/requirements.txt -e .
.venv-manim/Scripts/manim --version
```

Expected: `Manim Community v0.21.0`.

- [x] **Step 4: Font check (scratch scene, not committed)**

Write `<scratchpad>/fontcheck.py`:

```python
from manim import Scene, Text, config

config.background_color = "#101418"


class FontCheck(Scene):
    def construct(self):
        self.add(
            Text(
                "原生排隊 passthrough 1.5 倍突發 TTFT p95 ≤ 1 s",
                font="Microsoft JhengHei",
                font_size=36,
            )
        )
        self.wait(0.1)
```

Run: `.venv-manim/Scripts/manim -ql -s <scratchpad>/fontcheck.py FontCheck --media_dir <scratchpad>/media` and Read the PNG under `<scratchpad>/media/images/fontcheck/`.
Expected: CJK glyphs rendered (not boxes). If boxes appear, change `FONT` in Task 4 to `"Noto Sans TC"` or the system default and note it in `scripts/manim/README.md`.

- [x] **Step 5: Write `scripts/manim/README.md`**

```markdown
# 解說動畫（Manim Community）

這裡的動畫是**示意**，不是證據：每個數字都讀自 `analysis/tables/` 與 `evidence/raw/`，但影片不進 `make reproduce`、不進 CI、不列入 claims audit 的證據路徑（設計：`docs/superpowers/specs/2026-09-13-w3-admission-animation-design.md`）。

## 安裝（一次）

```bash
uv venv .venv-manim --python 3.12
uv pip install --python .venv-manim/Scripts/python.exe -r scripts/manim/requirements.txt -e .
```

不需要 LaTeX（只用 Pango 文字）；GIF 需要 ffmpeg 在 PATH。字型用 Windows 內建的 Microsoft JhengHei；其他系統把 `scripts/manim/w3_admission.py` 的 `FONT` 改成有的 CJK 字型（例如 Noto Sans TC）。

## 渲染

```bash
.venv-manim/Scripts/manim -qh --disable_caching scripts/manim/w3_admission.py W3AdmissionBurst
# -> media/videos/w3_admission/1080p30/W3AdmissionBurst.mp4（media/ 已 git-ignore）
cp media/videos/w3_admission/1080p30/W3AdmissionBurst.mp4 docs/media/w3-admission-burst.mp4
ffmpeg -y -i docs/media/w3-admission-burst.mp4 -vf "fps=12,scale=640:-1:flags=lanczos,palettegen" media/palette.png
ffmpeg -y -i docs/media/w3-admission-burst.mp4 -i media/palette.png -filter_complex "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" docs/media/w3-admission-burst.gif
```

迭代時用 `-ql`（480p15）看時間軸，最後才用 `-qh`。

## 場景與資料

- `w3_data.py`：讀 seed 1 的三個策略目錄（`evidence/raw/w3/fp8/trace/seed-1/trace-<policy>/`）、`analysis/tables/w3-fp8-admission/admission.json`、`w3-fp8-c192-admission/admission.json`，用 `slo_lab.timeline` 分桶；不 import manim，`tests/test_w3_anim_data.py` 在 CI 驗證它讀出的值與表相同。
- `w3_admission.py`：場景 `W3AdmissionBurst`，只畫。

## 核對表（渲染後人工核對）

| 影片裡的數字 | 表 |
|---|---|
| 到達率 21.8 / 65.5 / 21.8 rps、相位 300 / 600 s | `evidence/raw/w3/fp8/trace/seed-1/trace-seed-1.json` |
| 整段 attainment 0.19 / 0.57 / 0.59；goodput 5.8 / 17.5 / 18.0；拒絕率 0 / 20.9 / 19.9 %；time-to-recover 745–820 s（seed 2 未恢復）/ 0 / 5–10 s | `analysis/tables/w3-fp8-admission/admission.json` `per_cell_policy.fp8` |
| 尾聲：突發段 TPOT p95 57.8 / 45.2 ms、突發段 attainment 0.007 / 0.491、整段 0.573 / 0.780、拒絕率 20.9 / 21.8 % | 同上與 `analysis/tables/w3-fp8-c192-admission/admission.json` |
```

- [x] **Step 6: Commit**

```bash
git add .gitignore scripts/manim/requirements.txt scripts/manim/README.md
GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "scripts/manim: Manim Community 0.21 environment (separate venv, ignored), render steps and number check-list for the explainer animation" < /dev/null
```

---

### Task 3: `scripts/manim/w3_data.py` (manim-free loader) with a CI test

**Files:**
- Create: `scripts/manim/w3_data.py`
- Test: `tests/test_w3_anim_data.py` (loads the module by path, like `tests/test_compress_evidence.py`)

**Interfaces:**
- Consumes: `slo_lab.timeline` (Task 1), `slo_lab.admission_analysis.waiting_series(stage_dir, *, origin_mono=...)`, `slo_lab.slo.read_records_jsonl`.
- Produces:
  - constants `POLICIES = ("passthrough", "hard_cap", "bounded_queue")`, `TRACE_END_S = 1500.0`, `QUEUE_STEP_S = 5.0`, `BUCKET_S = 10.0`, `ROLL = 3`
  - `Lane(policy, queue, buckets, raw_buckets)` with `queue_at(t) -> float`, `bucket_at(t) -> Bucket`, `raw_bucket_index(t) -> int`
  - `ReplayData(lanes, phases, rate_rps, scoreboard, per_seed, coda, queue_max)`
  - `load_replay(root: Path, seed: int = 1) -> ReplayData`

- [x] **Step 1: Write the failing test**

```python
"""The explainer animation's loader reads exactly the committed evidence and tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("w3_data", REPO / "scripts" / "manim" / "w3_data.py")
assert _spec is not None and _spec.loader is not None
w3_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w3_data)


@pytest.fixture(scope="module")
def replay():
    return w3_data.load_replay(REPO)


def test_lanes_cover_the_trace_on_fixed_grids(replay) -> None:
    assert list(replay.lanes) == list(w3_data.POLICIES)
    for lane in replay.lanes.values():
        assert lane.queue[0] == (0.0, 0.0) or lane.queue[0][0] == 0.0
        assert lane.queue[-1][0] == 1500.0 and len(lane.queue) == 301
        assert len(lane.buckets) == 150 and len(lane.raw_buckets) == 150
        assert sum(b.offered for b in lane.raw_buckets) == 45624
    assert replay.queue_max == pytest.approx(9479.0)
    assert replay.lanes["hard_cap"].queue_at(450.0) == 0.0
    assert replay.lanes["passthrough"].queue_at(600.0) > 5000
    assert replay.lanes["passthrough"].raw_bucket_index(1499.9) == 149
    assert replay.lanes["hard_cap"].bucket_at(450.0).rejected > 0


def test_phases_rates_and_scoreboard_match_the_tables(replay) -> None:
    assert [p.name for p in replay.phases] == ["pre", "burst", "recovery"]
    assert replay.rate_rps == {"pre": 21.835, "burst": 65.505, "recovery": 21.835}
    sb = replay.scoreboard
    assert sb["passthrough"]["attainment"] == pytest.approx(0.1918, abs=5e-4)
    assert sb["hard_cap"]["attainment"] == pytest.approx(0.5729, abs=5e-4)
    assert sb["bounded_queue"]["goodput_rps"] == pytest.approx(18.01, abs=0.01)
    assert sb["hard_cap"]["rejection_rate"] == pytest.approx(0.2092, abs=5e-4)
    assert replay.per_seed["passthrough"]["time_to_recover_s"] == [820.0, None, 745.0]
    assert replay.coda["c256"]["tpot_p95_burst_s"] == pytest.approx(0.0578, abs=5e-4)
    assert replay.coda["c192"]["tpot_p95_burst_s"] == pytest.approx(0.0452, abs=5e-4)
    assert replay.coda["c192"]["attainment_burst"] == pytest.approx(0.491, abs=1e-3)
    assert replay.coda["c192"]["attainment"] == pytest.approx(0.7804, abs=5e-4)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/test_w3_anim_data.py`
Expected: FAIL (`FileNotFoundError` for `scripts/manim/w3_data.py`).

- [x] **Step 3: Write the loader**

```python
"""Everything the W3 explainer scene draws, computed once from the committed evidence.

Seed 1's replay through each policy gives the queue series and the 10 s buckets; the 3-seed
means come from ``analysis/tables``. No manim import, so ``tests/test_w3_anim_data.py`` checks
in CI that the values the animation shows are the tables' values.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slo_lab.admission_analysis import waiting_series
from slo_lab.slo import read_records_jsonl
from slo_lab.timeline import Bucket, Phase, bucket_records, downsample, phases_of, rolling

POLICIES = ("passthrough", "hard_cap", "bounded_queue")
TRACE_END_S = 1500.0
QUEUE_STEP_S = 5.0
BUCKET_S = 10.0
ROLL = 3
_METRICS = (
    "attainment",
    "goodput_rps",
    "rejection_rate",
    "attainment_burst",
    "attainment_recovery",
    "ttft_p95_burst_s",
    "time_to_recover_s",
)


@dataclass(frozen=True)
class Lane:
    policy: str
    queue: list[tuple[float, float]]  # every QUEUE_STEP_S from 0 to TRACE_END_S
    buckets: list[Bucket]  # BUCKET_S buckets rolled over ROLL buckets (readouts)
    raw_buckets: list[Bucket]  # unrolled (rejection sparks)

    def queue_at(self, t: float) -> float:
        i = min(len(self.queue) - 1, max(0, int(t // QUEUE_STEP_S)))
        return self.queue[i][1]

    def raw_bucket_index(self, t: float) -> int:
        return min(len(self.raw_buckets) - 1, max(0, int(t // BUCKET_S)))

    def bucket_at(self, t: float) -> Bucket:
        return self.buckets[self.raw_bucket_index(t)]


@dataclass(frozen=True)
class ReplayData:
    lanes: dict[str, Lane]
    phases: list[Phase]
    rate_rps: dict[str, float]
    scoreboard: dict[str, dict[str, float | None]]  # policy -> 3-seed means
    per_seed: dict[str, dict[str, list[float | None]]]  # policy -> metric -> per-seed values
    coda: dict[str, dict[str, float | None]]  # "c256" / "c192" -> hard-cap means
    queue_max: float


def _table(root: Path, name: str) -> dict[str, Any]:
    path = root / "analysis" / "tables" / name / "admission.json"
    return json.loads(path.read_text(encoding="utf-8"))["per_cell_policy"]


def _means(entry: dict[str, Any]) -> dict[str, float | None]:
    return {k: entry["mean"].get(k) for k in _METRICS}


def _tpot_burst_mean(root: Path, name: str, cell: str) -> float | None:
    """3-seed mean of the burst-stage TPOT p95 from the table's stage rows (not in the means)."""
    path = root / "analysis" / "tables" / name / "admission.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    vals = [r["tpot_p95_burst_s"] for r in rows if r["cell"] == cell and r["policy"] == "hard_cap"]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def load_replay(root: Path, seed: int = 1) -> ReplayData:
    base = root / "evidence" / "raw" / "w3" / "fp8" / "trace" / f"seed-{seed}"
    lanes: dict[str, Lane] = {}
    phases: list[Phase] = []
    for policy in POLICIES:
        d = base / f"trace-{policy}"
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if not phases:
            phases = phases_of(m)
        series = waiting_series(d, origin_mono=m.get("records_origin_monotonic_s"))
        queue = downsample(series, QUEUE_STEP_S, TRACE_END_S)
        raw = bucket_records(
            read_records_jsonl(d / "records.jsonl"), bucket_s=BUCKET_S, end_s=TRACE_END_S
        )
        lanes[policy] = Lane(policy, queue, rolling(raw, ROLL), raw)
    trace = json.loads((base / f"trace-seed-{seed}.json").read_text(encoding="utf-8"))
    rate_rps = {p["phase"]: float(p["rate_rps"]) for p in trace["phases"]}
    w3 = _table(root, "w3-fp8-admission")["fp8"]
    c192 = _table(root, "w3-fp8-c192-admission")["fp8-c192"]["hard_cap"]
    scoreboard = {p: _means(w3[p]) for p in POLICIES}
    per_seed = {
        p: {k: list(w3[p]["per_seed"][k]) for k in _METRICS if k in w3[p]["per_seed"]}
        for p in POLICIES
    }
    coda = {"c256": _means(w3["hard_cap"]), "c192": _means(c192)}
    coda["c256"]["tpot_p95_burst_s"] = _tpot_burst_mean(root, "w3-fp8-admission", "fp8")
    coda["c192"]["tpot_p95_burst_s"] = _tpot_burst_mean(root, "w3-fp8-c192-admission", "fp8-c192")
    queue_max = max(v for lane in lanes.values() for _, v in lane.queue)
    return ReplayData(lanes, phases, rate_rps, scoreboard, per_seed, coda, queue_max)


_ = bisect_right  # keep the import explicit for readers extending queue_at to arbitrary grids
```

(Remove the trailing `bisect_right` line and import if ruff flags it as unused; `queue_at` uses integer division, so it is not needed.)

- [x] **Step 4: Run the test to verify it passes**

Run: `uv run ruff format scripts/manim/w3_data.py tests/test_w3_anim_data.py && uv run ruff check . && uv run pytest -q tests/test_w3_anim_data.py`
Expected: 2 passed (loading three 45k-record files takes a few seconds). If `per_seed["passthrough"]["time_to_recover_s"]` differs in order, print `replay.per_seed` and fix the expected list to the table's order (seeds 1, 2, 3 → 820, None, 745 per `analysis/tables/w3-fp8-admission/tables.md`).

- [x] **Step 5: Commit**

```bash
git add scripts/manim/w3_data.py tests/test_w3_anim_data.py
GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "scripts/manim/w3_data: manim-free loader for the W3 explainer (seed-1 queue series and 10 s buckets, 3-seed means, C=256 vs C=192); CI test pins its values to the tables" < /dev/null
```

---

### Task 4: The scene `W3AdmissionBurst`

**Files:**
- Create: `scripts/manim/w3_admission.py`

**Interfaces:**
- Consumes: `w3_data.load_replay`, `Lane`, `ReplayData`, `POLICIES`, `TRACE_END_S`; manim 0.21 (`Scene`, `ValueTracker`, `always_redraw`, `DecimalNumber`, `Integer`, `Text`, `Rectangle`, `NumberLine`, `Line`, `Dot`, `VGroup`, `Table`, `FadeIn`, `FadeOut`, `Write`, `Create`, `GrowFromEdge`, `linear`, `rate_functions`).
- Produces: the scene class `W3AdmissionBurst` rendering ~75 s.

- [x] **Step 1: Write the scene**

```python
"""W3 explainer: one RTX 4090, a 1.5x five-minute burst, three admission policies, then
hard cap C = 256 vs C = 192 (ADR 0013, ADR 0018). Illustration only: every number shown is
read from the committed tables by ``w3_data``; the scene draws and never computes.

Render: .venv-manim/Scripts/manim -qh --disable_caching scripts/manim/w3_admission.py W3AdmissionBurst
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Integer,
    Line,
    NumberLine,
    Rectangle,
    Scene,
    Table,
    Text,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
    linear,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w3_data import POLICIES, TRACE_END_S, Lane, ReplayData, load_replay  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FONT = "Microsoft JhengHei"
BG = "#101418"
INK = "#e8ecf1"
MUTED = "#8a94a3"
RED = "#ff4d4d"
GREEN = "#7bc96f"
COLORS = {"passthrough": "#e4572e", "hard_cap": "#4c9be8", "bounded_queue": "#7bc96f"}
LABELS = {
    "passthrough": "原生排隊（passthrough）",
    "hard_cap": "hard cap + 429（C = 256）",
    "bounded_queue": "有界佇列 + 1 s 逾時（Q = 256）",
}
TIMELINE_LEN = 11.0
BAR_MAX_W = 5.6
BAR_H = 0.55
LANE_Y = (1.15, -0.35, -1.85)
BAR_LEFT_X = -2.0
REPLAY_TO_BURST_END_S = 18.0
REPLAY_RECOVERY_S = 12.0


def _t(s: str, size: int = 28, color: str = INK, weight: str = "NORMAL") -> Text:
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def _log_width(queued: float, queue_max: float) -> float:
    if queued <= 0 or queue_max <= 0:
        return 0.0
    return BAR_MAX_W * math.log10(1.0 + queued) / math.log10(1.0 + queue_max)


class W3AdmissionBurst(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG
        random.seed(3)
        data = load_replay(ROOT)
        self.title_card()
        stage = self.build_stage(data)
        self.replay(data, stage)
        self.recovery_notes(data, stage)
        self.clear()
        self.scoreboard(data)
        self.clear()
        self.coda(data)
        self.clear()
        self.end_card()

    # ---- 1. title (3 s)
    def title_card(self) -> None:
        title = _t("一張 RTX 4090、1.5 倍 5 分鐘突發、三種 admission", 44, weight="BOLD")
        sub = _t(
            "SLO：TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms（Qwen3-8B-FP8，vLLM 0.28，WSL2）", 26, MUTED
        )
        sub.next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(1.6)
        self.play(FadeOut(title), FadeOut(sub), run_time=0.5)

    # ---- 2. stage (8 s)
    def build_stage(self, data: ReplayData) -> dict:
        axis = NumberLine(
            x_range=[0, TRACE_END_S, 300],
            length=TIMELINE_LEN,
            include_numbers=True,
            font_size=20,
            color=MUTED,
            decimal_number_config={"num_decimal_places": 0},
        ).shift(UP * 2.55)
        unit = _t("秒（trace 時間）", 18, MUTED).next_to(axis, RIGHT, buff=0.15)
        shades = VGroup()
        for ph in data.phases:
            x0, x1 = axis.n2p(ph.start_s)[0], axis.n2p(ph.end_s)[0]
            fill = "#3a2326" if ph.name == "burst" else "#1a2028"
            shades.add(
                Rectangle(
                    width=x1 - x0, height=1.1, fill_color=fill, fill_opacity=0.9, stroke_width=0
                ).move_to([(x0 + x1) / 2, axis.get_y() + 0.75, 0])
            )
        steps = VGroup()
        rate_max = max(data.rate_rps.values())
        for ph in data.phases:
            r = data.rate_rps[ph.name]
            y = axis.get_y() + 0.25 + 0.85 * (r / rate_max)
            seg = Line(
                [axis.n2p(ph.start_s)[0], y, 0],
                [axis.n2p(ph.end_s)[0], y, 0],
                color=INK,
                stroke_width=3,
            )
            lab = _t(f"{r:.1f} rps", 18).next_to(seg, UP, buff=0.05)
            steps.add(VGroup(seg, lab))
        caption = _t("到達率：0.5× → 1.5×（5 分鐘）→ 0.5× r_sat", 20, MUTED).next_to(
            shades, UP, buff=0.55
        )
        self.play(FadeIn(shades), Create(axis), FadeIn(unit), run_time=1.2)
        self.play(FadeIn(caption), run_time=0.4)
        for s in steps:
            self.play(GrowFromEdge(s[0], LEFT), FadeIn(s[1]), run_time=0.6)
        lanes: dict[str, dict] = {}
        for policy, y in zip(POLICIES, LANE_Y):
            label = (
                _t(LABELS[policy], 22, COLORS[policy])
                .move_to([-4.6, y + 0.42, 0])
                .align_to([-6.9, 0, 0], LEFT)
            )
            anchor = Line(
                [BAR_LEFT_X, y - BAR_H / 2, 0],
                [BAR_LEFT_X, y + BAR_H / 2, 0],
                color=MUTED,
                stroke_width=2,
            )
            qlabel = _t("佇列", 16, MUTED).next_to(anchor, LEFT, buff=0.1)
            lanes[policy] = {"y": y, "label": label, "anchor": anchor, "qlabel": qlabel}
            self.play(FadeIn(label), Create(anchor), FadeIn(qlabel), run_time=0.45)
        scale_note = _t("佇列長條為對數尺度；讀數為近 30 s 的滾動值", 16, MUTED).to_edge(
            DOWN, buff=0.25
        )
        self.play(FadeIn(scale_note), run_time=0.4)
        self.wait(0.6)
        return {
            "axis": axis,
            "lanes": lanes,
            "shades": shades,
            "steps": steps,
            "caption": caption,
            "note": scale_note,
        }

    # ---- 3. replay (30 s)
    def replay(self, data: ReplayData, stage: dict) -> None:
        axis: NumberLine = stage["axis"]
        t = ValueTracker(0.0)
        cursor = always_redraw(
            lambda: Line(
                [axis.n2p(t.get_value())[0], axis.get_y() - 0.25, 0],
                [axis.n2p(t.get_value())[0], axis.get_y() + 1.3, 0],
                color=INK,
                stroke_width=2,
            )
        )
        clock = always_redraw(
            lambda: _t(f"t = {int(t.get_value()):4d} s", 22).move_to([5.9, axis.get_y() + 1.55, 0])
        )
        self.add(cursor, clock)
        dyn = VGroup()
        for policy in POLICIES:
            lane: Lane = data.lanes[policy]
            y = stage["lanes"][policy]["y"]
            color = COLORS[policy]
            bar = always_redraw(
                lambda lane=lane, y=y, color=color: (
                    Rectangle(
                        width=max(0.02, _log_width(lane.queue_at(t.get_value()), data.queue_max)),
                        height=BAR_H,
                        fill_color=color,
                        fill_opacity=0.85,
                        stroke_width=0,
                    )
                    .align_to([BAR_LEFT_X, 0, 0], LEFT)
                    .set_y(y)
                )
            )
            qnum = Integer(0, font_size=22, color=INK)
            qnum.add_updater(
                lambda m, lane=lane, y=y: (
                    m.set_value(int(lane.queue_at(t.get_value())))
                    .next_to(m.bar_ref, RIGHT, buff=0.12)
                    .set_y(y)
                )
            )
            qnum.bar_ref = bar
            att = DecimalNumber(0, num_decimal_places=2, font_size=24, color=INK)
            att_label = _t("attainment", 16, MUTED)
            ttft = DecimalNumber(0, num_decimal_places=2, font_size=24, color=INK)
            ttft_label = _t("TTFT p95 (s)", 16, MUTED)
            rej = Integer(0, font_size=24, color=INK)
            rej_label = _t("429（累計）", 16, MUTED)
            readouts = (
                VGroup(
                    VGroup(att_label, att).arrange(DOWN, buff=0.05),
                    VGroup(ttft_label, ttft).arrange(DOWN, buff=0.05),
                    VGroup(rej_label, rej).arrange(DOWN, buff=0.05),
                )
                .arrange(RIGHT, buff=0.55)
                .move_to([5.2, y, 0])
            )

            def upd_att(m, lane=lane):
                b = lane.bucket_at(t.get_value())
                a = b.attainment
                m.set_value(a if a is not None else 0.0)
                m.set_color(
                    GREEN
                    if a is not None and a >= 0.95
                    else (RED if a is not None and a < 0.5 else INK)
                )

            def upd_ttft(m, lane=lane):
                b = lane.bucket_at(t.get_value())
                v = b.ttft_p95_s
                m.set_value(v if v is not None else 0.0)
                m.set_color(RED if v is not None and v > 1.0 else INK)

            def upd_rej(m, lane=lane):
                i = lane.raw_bucket_index(t.get_value())
                m.set_value(sum(b.rejected for b in lane.raw_buckets[: i + 1]))

            att.add_updater(upd_att)
            ttft.add_updater(upd_ttft)
            rej.add_updater(upd_rej)
            sparks = self._sparks(lane, y, t, color)
            dyn.add(bar, qnum, readouts, sparks)
        self.add(dyn)
        burst_end = next(p.end_s for p in data.phases if p.name == "burst")
        self.play(t.animate.set_value(burst_end), run_time=REPLAY_TO_BURST_END_S, rate_func=linear)
        self.play(t.animate.set_value(TRACE_END_S), run_time=REPLAY_RECOVERY_S, rate_func=linear)
        stage["dyn"] = dyn
        stage["cursor"] = (cursor, clock)

    def _sparks(self, lane: Lane, y: float, t: ValueTracker, color: str) -> VGroup:
        """Red dots leaving the lane while a bucket has 429s; count grows with log2(rejected)."""
        group = VGroup()
        state = {"bucket": -1, "clock": 0.0}

        def updater(g: VGroup, dt: float) -> None:
            state["clock"] += dt
            i = lane.raw_bucket_index(t.get_value())
            if i != state["bucket"]:
                state["bucket"] = i
                n = lane.raw_buckets[i].rejected
                if n > 0:
                    k = min(6, 1 + int(math.log2(n)))
                    x0 = (
                        BAR_LEFT_X
                        + _log_width(
                            lane.queue_at(t.get_value()),
                            1.0 + max(1.0, lane.queue_at(t.get_value())),
                        )
                        + 0.15
                    )
                    for _ in range(k):
                        d = Dot(radius=0.05, color=RED).move_to(
                            [x0 + random.uniform(0, 0.3), y + random.uniform(-0.2, 0.2), 0]
                        )
                        d.birth = state["clock"]
                        d.vel = (random.uniform(0.6, 1.2), random.uniform(0.4, 1.0))
                        g.add(d)
            for d in list(g.submobjects):
                age = state["clock"] - d.birth
                if age > 1.0:
                    g.remove(d)
                    continue
                d.shift([d.vel[0] * dt, d.vel[1] * dt, 0])
                d.set_opacity(1.0 - age)

        group.add_updater(updater)
        return group

    # ---- 4. recovery notes (hold 4 s)
    def recovery_notes(self, data: ReplayData, stage: dict) -> None:
        ttr = data.per_seed["passthrough"]["time_to_recover_s"]
        known = sorted(v for v in ttr if v is not None)
        unrec = sum(1 for v in ttr if v is None)
        lines = [
            f"time-to-recover：原生排隊 {int(known[0])}–{int(known[-1])} s（{unrec} 個 seed 未恢復）",
            "hard cap 0 s · 有界佇列 5–10 s",
        ]
        note = (
            VGroup(*[_t(s, 22, INK) for s in lines]).arrange(DOWN, buff=0.12).move_to([0, -3.0, 0])
        )
        stage["note"].set_opacity(0)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(3.4)

    # ---- 5. scoreboard (8 s)
    def scoreboard(self, data: ReplayData) -> None:
        sb = data.scoreboard
        head = _t("整段（25 分鐘）結果，3 seeds 平均", 34, weight="BOLD").to_edge(UP, buff=0.6)

        def ttr_text(policy: str) -> str:
            vals = data.per_seed[policy]["time_to_recover_s"]
            known = sorted(v for v in vals if v is not None)
            if not known:
                return "未恢復"
            lo, hi = int(known[0]), int(known[-1])
            base = f"{lo} s" if lo == hi else f"{lo}–{hi} s"
            missing = len(vals) - len(known)
            return base + (f"（{missing} seed 未恢復）" if missing else "")

        rows = [
            [
                LABELS[p],
                f"{sb[p]['attainment']:.2f}",
                f"{sb[p]['goodput_rps']:.1f} rps",
                f"{100 * sb[p]['rejection_rate']:.1f} %",
                ttr_text(p),
            ]
            for p in POLICIES
        ]
        table = (
            Table(
                rows,
                col_labels=[
                    _t(s, 22, MUTED)
                    for s in ["策略", "attainment", "goodput", "拒絕率", "time-to-recover"]
                ],
                element_to_mobject=lambda s: _t(s, 24),
                include_outer_lines=False,
                line_config={"stroke_color": MUTED, "stroke_width": 1},
                h_buff=0.6,
                v_buff=0.35,
            )
            .scale(0.8)
            .next_to(head, DOWN, buff=0.5)
        )
        for i, p in enumerate(POLICIES):
            table.get_rows()[i + 1][0].set_color(COLORS[p])
            a = sb[p]["attainment"]
            table.get_rows()[i + 1][1].set_color(GREEN if a >= 0.5 else RED)
        self.play(FadeIn(head), run_time=0.5)
        self.play(
            Create(table.get_horizontal_lines()), Create(table.get_vertical_lines()), run_time=0.8
        )
        self.play(FadeIn(table.get_col_labels()), run_time=0.4)
        for i in range(3):
            self.play(FadeIn(table.get_rows()[i + 1]), run_time=0.6)
        takeaway = _t(
            "限流讓 attainment 提高約 0.4、突發後 10 s 內恢復；代價是拒絕 12–21 % 的請求", 22, INK
        ).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(takeaway), run_time=0.5)
        self.wait(3.2)

    # ---- 6. coda: C = 256 vs C = 192 (12 s)
    def coda(self, data: ReplayData) -> None:
        head = _t(
            "同一條到達序列，只改 hard cap 的上限：C = 256 → C = 192（ADR 0018）", 32, weight="BOLD"
        ).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.5)
        rows = []
        for key, label, y in (
            ("c256", "C = 256（closed-loop 的平台）", 1.0),
            ("c192", "C = 192", -0.9),
        ):
            m = data.coda[key]
            tpot_ms = 1000 * m["tpot_p95_burst_s"]
            lab = _t(label, 26, INK).move_to([-4.6, y + 0.7, 0])
            scale = 5.0 / 60.0
            bar = (
                Rectangle(
                    width=tpot_ms * scale,
                    height=0.5,
                    fill_color=RED if tpot_ms > 50 else GREEN,
                    fill_opacity=0.9,
                    stroke_width=0,
                )
                .align_to([-6.0, 0, 0], LEFT)
                .set_y(y)
            )
            val = _t(
                f"突發段 TPOT p95 {tpot_ms:.1f} ms", 24, RED if tpot_ms > 50 else GREEN
            ).next_to(bar, RIGHT, buff=0.2)
            nums = (
                _t(
                    f"突發段 attainment {m['attainment_burst']:.2f} · 整段 {m['attainment']:.2f} · 拒絕率 {100 * m['rejection_rate']:.1f} %",
                    22,
                    MUTED,
                )
                .move_to([-6.0 + 0.0, y - 0.6, 0])
                .align_to([-6.0, 0, 0], LEFT)
            )
            rows.append((lab, bar, val, nums))
        thr_x = -6.0 + 50 * (5.0 / 60.0)
        thr = Line([thr_x, 1.7, 0], [thr_x, -1.8, 0], color=RED, stroke_width=2)
        thr_lab = _t("SLO 50 ms", 18, RED).next_to(thr, UP, buff=0.05)
        self.play(Create(thr), FadeIn(thr_lab), run_time=0.5)
        for lab, bar, val, nums in rows:
            self.play(FadeIn(lab), run_time=0.3)
            self.play(GrowFromEdge(bar, LEFT), run_time=0.9)
            self.play(FadeIn(val), FadeIn(nums), run_time=0.5)
            self.wait(0.6)
        msg = _t(
            "上限要用突發下仍 TPOT 安全的並行取，不是 closed-loop 的平台", 26, INK, weight="BOLD"
        ).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(msg), run_time=0.5)
        self.wait(4.0)

    # ---- 7. end card (4 s)
    def end_card(self) -> None:
        lines = [
            _t(
                "示意動畫：數字出自 analysis/tables/w3-fp8-admission 與 w3-fp8-c192-admission",
                24,
                INK,
            ),
            _t("由 make reproduce 從提交的證據重建；ADR 0013、0018", 24, INK),
            _t("github.com/kuotunyu/vllm-single-gpu-slo-lab", 28, MUTED),
        ]
        g = VGroup(*lines).arrange(DOWN, buff=0.3)
        self.play(FadeIn(g), run_time=0.8)
        self.wait(3.2)
```

- [x] **Step 2: Low-quality render to check timing and layout**

Run: `.venv-manim/Scripts/manim -ql --disable_caching scripts/manim/w3_admission.py W3AdmissionBurst`
Expected: `media/videos/w3_admission/480p15/W3AdmissionBurst.mp4` written; no traceback. Then extract frames to inspect (Read the PNGs): `ffmpeg -y -i media/videos/w3_admission/480p15/W3AdmissionBurst.mp4 -vf "fps=1/8" <scratchpad>/frames/f%03d.png`. Check: labels not overlapping bars, readouts inside the frame, numbers legible, the cursor reaching 1500 s, the coda bars ending left/right of the 50 ms line as expected.

- [x] **Step 3: Fix layout issues found, re-render `-ql` until clean**

Typical fixes: move `LANE_Y`, `BAR_LEFT_X`, readout x (5.2), font sizes. Keep the timings.

- [x] **Step 4: Lint and commit the scene**

Run: `uv run ruff format scripts/manim/w3_admission.py && uv run ruff check .` (ruff runs on the repo's own venv: the file imports manim, which ruff does not need to resolve).

```bash
git add scripts/manim/w3_admission.py
GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "scripts/manim/w3_admission: W3 explainer scene (title, stage, 30 s replay of the three policies, recovery notes, scoreboard, C=256 vs C=192 coda, end card)" < /dev/null
```

---

### Task 5: Final render, GIF, README and docs, CI

**Files:**
- Create: `docs/media/w3-admission-burst.mp4`, `docs/media/w3-admission-burst.gif`
- Modify: `README.md` (after the W3 補點 paragraph in the status block), `evidence/README.md` (append), `analysis/claims_audit.md` (preamble sentence), `docs/HANDOFF.md` (已完成 table row), `docs/superpowers/plans/2026-09-11-w4-specdec.md` (run-log row)

- [x] **Step 1: High-quality render and copy**

```bash
.venv-manim/Scripts/manim -qh --disable_caching scripts/manim/w3_admission.py W3AdmissionBurst
mkdir -p docs/media
cp media/videos/w3_admission/1080p30/W3AdmissionBurst.mp4 docs/media/w3-admission-burst.mp4
ffprobe -v error -show_entries format=duration,size -of default=nw=1 docs/media/w3-admission-burst.mp4
```

Expected: duration 70–80 s, size under 15 MB.

- [x] **Step 2: GIF (two-pass palette), check size**

```bash
ffmpeg -y -i docs/media/w3-admission-burst.mp4 -vf "fps=12,scale=640:-1:flags=lanczos,palettegen=max_colors=128" media/palette.png
ffmpeg -y -i docs/media/w3-admission-burst.mp4 -i media/palette.png -filter_complex "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" docs/media/w3-admission-burst.gif
ls -l docs/media/
```

If the GIF is over 8 MB: re-run with `-ss 11 -t 42` on the input (the replay segment only) and say so in the README caption.

- [x] **Step 3: Docs**

README, after the C = 192 補點 paragraph inside the status block, add:

```markdown
>
> ![W3 admission 示意動畫](docs/media/w3-admission-burst.gif)
>
> 示意動畫（Manim Community，`scripts/manim/`）：三種 admission 重播同一條突發 trace，再比 C = 256 與 C = 192。數字全部出自 `analysis/tables/w3-*-admission/`；完整版 `docs/media/w3-admission-burst.mp4`。動畫不是證據，不在 `make reproduce` 的範圍。
```

`evidence/README.md`, append: `示意動畫（2026-09-13）：\`docs/media/\` 的 mp4／gif 由 \`scripts/manim/\` 從表與紀錄畫出，只重述表內數字，不屬於證據、不在 \`make reproduce\` 的 diff 範圍。`

`analysis/claims_audit.md`, preamble (after 共同條件 sentence): `\`docs/media/\` 的示意動畫只重述下表的數字，不新增任何 claim。`

`docs/HANDOFF.md` 已完成 table: add a row `| 解說動畫 | W3 admission 的 Manim 動畫（\`docs/media/\`），示意用，不進 reproduce／CI | 設計 \`docs/superpowers/specs/2026-09-13-w3-admission-animation-design.md\` |`.

W4 plan run log: a row for 2026-09-13 with the render facts (duration, sizes, tests 187).

- [x] **Step 4: Verify, commit, push, CI**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q && uv run python scripts/redact.py audit . && make reproduce
git add -A
GIT_ASK_YESNO=false git -c user.name=kuotunyu -c user.email=61350295+kuotunyu@users.noreply.github.com commit -m "docs/media: W3 admission explainer animation (mp4 + gif) embedded in the README as an illustration; evidence README, claims audit preamble, HANDOFF" < /dev/null
git push origin main
gh run watch $(gh run list --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status --interval 20
```

Expected: all green; `make reproduce` zero diff (media is outside `evidence/` and `analysis/`).

- [x] **Step 5: Control tower ledger line** in `_portfolio_control/docs/inventory/2026-09-02-arsenal-v3-inventory.md` (§19 補記) and commit there.

---

## Self-review

- Spec coverage: data module (Task 1), environment/README (Task 2), loader with CI test (Task 3), scene sections 1–7 (Task 4), outputs, README/evidence README/claims audit note, size rule, CI (Task 5). The spec's "寬度 ∝ 佇列長度" is refined to a log scale (passthrough peaks at 9,479 queued while the bounded queue peaks at 79; a linear bar would hide the latter) and the on-screen note says so.
- Types: `Lane.queue_at/bucket_at/raw_bucket_index`, `ReplayData.per_seed`, `coda["c256"|"c192"]["tpot_p95_burst_s"]` are used by Task 4 exactly as defined in Task 3; `Bucket.attainment/ttft_p95_s/rejected` as defined in Task 1.
- No placeholders; every number the scene prints comes from `ReplayData`, not literals, except the fixed SLO thresholds and labels.
