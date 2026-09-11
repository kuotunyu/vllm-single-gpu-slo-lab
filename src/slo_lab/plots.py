"""Dependency-free SVG figures rebuilt from the committed tables and evidence (spec §8, W5).

Three figure families, all derived from files that are already in the repo so that
``make reproduce`` can rebuild them on a CPU-only runner without matplotlib:

- attainment vs offered rate (W2, W4): per cell the minimum attainment over seeds at each
  open-loop rate, from ``analysis/tables/<name>/open_loop.json``;
- TPOT p95 vs offered rate (W4): per cell the mean over seeds, with the 50 ms SLO line;
- queue timeline (W3): vLLM waiting + shim waiting against seconds from trace start for the
  three policies of one seed, aligned on the monotonic clock like the admission analysis.

The writer is deliberately small: fixed palette, coordinates rounded to two decimals, no
timestamps, so the same inputs always yield the same bytes and the committed SVGs diff clean.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from slo_lab.admission_analysis import POLICY_ORDER, waiting_series

PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f")
WIDTH, HEIGHT = 760, 420
MARGIN = {"left": 68, "right": 170, "top": 44, "bottom": 56}
W2_CELLS = ("bf16", "fp8", "awq", "gptq")
W4_FAMILIES = ("fp8", "q4b")


@dataclass
class Series:
    name: str
    points: list[tuple[float, float]]
    dashed: bool = False


def _fmt(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _ticks(lo: float, hi: float) -> list[float]:
    step = _nice_step(hi - lo)
    first = math.ceil(lo / step - 1e-9) * step
    out = []
    t = first
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def line_chart(
    series: Sequence[Series],
    *,
    title: str,
    x_label: str,
    y_label: str,
    hlines: Sequence[tuple[float, str]] = (),
    vlines: Sequence[tuple[float, str]] = (),
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """One SVG line chart; raises on empty input rather than drawing an empty frame."""
    drawable = [s for s in series if s.points]
    if not drawable:
        raise ValueError("line_chart needs at least one series with points")
    xs = [x for s in drawable for x, _ in s.points] + [x for x, _ in vlines]
    ys = [y for s in drawable for _, y in s.points] + [y for y, _ in hlines]
    x_lo, x_hi = x_range or (min(xs), max(xs))
    y_lo, y_hi = y_range or (min(0.0, min(ys)), max(ys))
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    if y_range is None:
        y_hi *= 1.05
    left, top = MARGIN["left"], MARGIN["top"]
    plot_w = width - left - MARGIN["right"]
    plot_h = height - top - MARGIN["bottom"]

    def px(x: float) -> str:
        return _fmt(left + (x - x_lo) / (x_hi - x_lo) * plot_w)

    def py(y: float) -> str:
        return _fmt(top + plot_h - (y - y_lo) / (y_hi - y_lo) * plot_h)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-size="15" font-weight="bold">{escape(title)}</text>',
    ]
    for t in _ticks(x_lo, x_hi):
        x = px(t)
        out.append(
            f'<line x1="{x}" y1="{_fmt(top)}" x2="{x}" y2="{_fmt(top + plot_h)}" '
            'stroke="#e6e6e6" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x}" y="{_fmt(top + plot_h + 16)}" text-anchor="middle">{_fmt(t)}</text>'
        )
    for t in _ticks(y_lo, y_hi):
        y = py(t)
        out.append(
            f'<line x1="{_fmt(left)}" y1="{y}" x2="{_fmt(left + plot_w)}" y2="{y}" '
            'stroke="#e6e6e6" stroke-width="1"/>'
        )
        out.append(f'<text x="{_fmt(left - 6)}" y="{y}" text-anchor="end" dy="4">{_fmt(t)}</text>')
    out.append(
        f'<line x1="{_fmt(left)}" y1="{_fmt(top + plot_h)}" x2="{_fmt(left + plot_w)}" '
        f'y2="{_fmt(top + plot_h)}" stroke="#333" stroke-width="1"/>'
    )
    out.append(
        f'<line x1="{_fmt(left)}" y1="{_fmt(top)}" x2="{_fmt(left)}" y2="{_fmt(top + plot_h)}" '
        'stroke="#333" stroke-width="1"/>'
    )
    out.append(
        f'<text x="{_fmt(left + plot_w / 2)}" y="{_fmt(height - 14)}" text-anchor="middle">'
        f"{escape(x_label)}</text>"
    )
    out.append(
        f'<text x="16" y="{_fmt(top + plot_h / 2)}" text-anchor="middle" '
        f'transform="rotate(-90 16 {_fmt(top + plot_h / 2)})">{escape(y_label)}</text>'
    )
    for value, label in hlines:
        y = py(value)
        out.append(
            f'<line x1="{_fmt(left)}" y1="{y}" x2="{_fmt(left + plot_w)}" y2="{y}" '
            'stroke="#888" stroke-width="1" stroke-dasharray="6 4"/>'
        )
        out.append(
            f'<text x="{_fmt(left + plot_w - 4)}" y="{y}" text-anchor="end" dy="-4" fill="#555">{escape(label)}</text>'
        )
    for value, label in vlines:
        x = px(value)
        out.append(
            f'<line x1="{x}" y1="{_fmt(top)}" x2="{x}" y2="{_fmt(top + plot_h)}" '
            'stroke="#888" stroke-width="1" stroke-dasharray="6 4"/>'
        )
        out.append(
            f'<text x="{x}" y="{_fmt(top - 6)}" text-anchor="middle" fill="#555">{escape(label)}</text>'
        )
    for i, s in enumerate(drawable):
        color = PALETTE[i % len(PALETTE)]
        dash = ' stroke-dasharray="8 4"' if s.dashed else ""
        pts = " ".join(f"{px(x)},{py(y)}" for x, y in sorted(s.points))
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
        )
        if len(s.points) <= 60:  # markers on sweep points, not on 5 s time series
            for x, y in sorted(s.points):
                out.append(f'<circle cx="{px(x)}" cy="{py(y)}" r="2.5" fill="{color}"/>')
        ly = top + 8 + i * 18
        lx = left + plot_w + 14
        out.append(
            f'<line x1="{_fmt(lx)}" y1="{_fmt(ly)}" x2="{_fmt(lx + 22)}" y2="{_fmt(ly)}" '
            f'stroke="{color}" stroke-width="2"{dash}/>'
        )
        out.append(f'<text x="{_fmt(lx + 28)}" y="{_fmt(ly)}" dy="4">{escape(s.name)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _open_loop_rows(table_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in table_dirs:
        path = Path(d) / "open_loop.json"
        if path.exists():
            rows += json.loads(path.read_text(encoding="utf-8")).get("rows", [])
    return [r for r in rows if not r.get("suspect")]


def attainment_series(table_dirs: Sequence[Path]) -> list[Series]:
    """Per cell, the minimum attainment over seeds at each offered rate (the r_SLO rule's view)."""
    by_cell: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in _open_loop_rows(table_dirs):
        if r.get("attainment") is not None and r.get("offered_rps") is not None:
            by_cell[r["cell"]][float(r["offered_rps"])].append(float(r["attainment"]))
    return [
        Series(cell, [(rate, min(v)) for rate, v in sorted(rates.items())])
        for cell, rates in sorted(by_cell.items())
    ]


def tpot_series(table_dirs: Sequence[Path]) -> list[Series]:
    """Per cell, the mean TPOT p95 (ms) over seeds at each offered rate."""
    by_cell: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in _open_loop_rows(table_dirs):
        if r.get("tpot_p95_s") is not None and r.get("offered_rps") is not None:
            by_cell[r["cell"]][float(r["offered_rps"])].append(float(r["tpot_p95_s"]) * 1000.0)
    return [
        Series(cell, [(rate, sum(v) / len(v)) for rate, v in sorted(rates.items())])
        for cell, rates in sorted(by_cell.items())
    ]


def attainment_vs_rate(table_dirs: Sequence[Path], *, title: str) -> str:
    return line_chart(
        attainment_series(table_dirs),
        title=title,
        x_label="offered rate (req/s, Poisson)",
        y_label="SLO attainment (min over seeds)",
        hlines=[(0.95, "95 % target")],
        y_range=(0.0, 1.02),
    )


def tpot_vs_rate(table_dirs: Sequence[Path], *, title: str, slo_tpot_s: float = 0.05) -> str:
    return line_chart(
        tpot_series(table_dirs),
        title=title,
        x_label="offered rate (req/s, Poisson)",
        y_label="TPOT p95 (ms, mean over seeds)",
        hlines=[(slo_tpot_s * 1000.0, f"{slo_tpot_s * 1000:g} ms")],
    )


def _stage_policy_key(policy: str | None) -> int:
    return POLICY_ORDER.index(policy) if policy in POLICY_ORDER else len(POLICY_ORDER)


def queue_timeline(stage_dirs: Sequence[Path], *, title: str) -> str:
    """Queued requests over a replayed trace, one line per admission policy."""
    series: list[tuple[int, Series]] = []
    vlines: list[tuple[float, str]] = []
    for d in stage_dirs:
        d = Path(d)
        manifest = d / "manifest.json"
        if not manifest.exists():
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        origin = m.get("records_origin_monotonic_s")
        offsets = [v for v in (m.get("clock_offset_s") or {}).values() if v is not None]
        origin_unix = (
            float(origin) + sum(offsets) / len(offsets) if origin is not None and offsets else None
        )
        points = waiting_series(d, origin_mono=origin, origin_unix=origin_unix)
        if not points:
            continue
        policy = m.get("policy") or d.name
        series.append((_stage_policy_key(policy), Series(str(policy), points)))
        if not vlines:
            for phase in (m.get("trace") or {}).get("phases") or []:
                if phase.get("phase") != "pre":
                    vlines.append((float(phase["start_s"]), str(phase["phase"])))
    return line_chart(
        [s for _, s in sorted(series, key=lambda t: t[0])],
        title=title,
        x_label="seconds from trace start",
        y_label="queued requests (vLLM waiting + shim waiting)",
        vlines=vlines,
    )


def rebuild_plots(root: Path) -> list[str]:
    """Write every figure whose inputs exist under ``root`` into ``evidence/plots``; names written."""
    tables = root / "analysis" / "tables"
    out_dir = root / "evidence" / "plots"
    figures: list[tuple[str, str]] = []
    w2 = [tables / f"w2-{cell}-open-loop" for cell in W2_CELLS]
    if any((d / "open_loop.json").exists() for d in w2):
        figures.append(
            (
                "w2-attainment-vs-rate.svg",
                attainment_vs_rate(
                    w2, title="W2: SLO attainment vs offered rate, four precisions (3 seeds, min)"
                ),
            )
        )
    for cell in ("fp8", "bf16"):
        stages = [
            root / "evidence" / "raw" / "w3" / cell / "trace" / "seed-1" / f"trace-{policy}"
            for policy in POLICY_ORDER
        ]
        if any((s / "manifest.json").exists() for s in stages):
            figures.append(
                (
                    f"w3-queue-timeline-{cell}.svg",
                    queue_timeline(
                        stages, title=f"W3: queue during the burst trace, {cell}, seed 1"
                    ),
                )
            )
    for family in W4_FAMILIES:
        table = tables / f"w4-{family}-specdec"
        if (table / "open_loop.json").exists():
            figures.append(
                (
                    f"w4-tpot-vs-rate-{family}.svg",
                    tpot_vs_rate(
                        [table], title=f"W4: TPOT p95 vs offered rate, {family} (3 seeds, mean)"
                    ),
                )
            )
            figures.append(
                (
                    f"w4-attainment-vs-rate-{family}.svg",
                    attainment_vs_rate(
                        [table], title=f"W4: SLO attainment vs offered rate, {family} (min)"
                    ),
                )
            )
    if figures:
        out_dir.mkdir(parents=True, exist_ok=True)
    for name, svg in figures:
        (out_dir / name).write_text(svg, encoding="utf-8", newline="\n")
    return sorted(name for name, _ in figures)
