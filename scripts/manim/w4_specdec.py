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
from style import (
    BG,
    BLUE,
    GREEN,
    INK,
    MUTED,
    ORANGE,
    RED,
    bottom_note,
    source_card,
    text,
)
from w4_data import CONCURRENCIES, SpecData, load_specdec

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"fp8-ngram": BLUE, "q4b-eagle3": ORANGE, "q4b-ngram": GREEN}
NOTE = [
    "Shakespeare 自然文字 prompt 108 → 132 tokens，Qwen3 預設取樣，全部經 passthrough shim",
    "--gpu-memory-utilization 0.82，WSL2 上與桌面共用的一張 RTX 4090",
]


def _bar(width: float, height: float, color: str) -> Rectangle:
    return Rectangle(
        width=width, height=max(height, 0.02), fill_color=color, fill_opacity=0.9, stroke_width=0
    )


class W4Specdec(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG
        data = load_specdec(ROOT)
        for section in (self.acceptance, self.closed_loop, self.open_loop, self.conclusion):
            self.add(bottom_note(NOTE))
            section(data)
            self.clear()
        card = source_card(
            [
                "示意動畫：數字出自 analysis/tables/w4-fp8-specdec 與 w4-q4b-specdec",
                "由 make reproduce 從提交的證據重建；ADR 0015、0016",
                "github.com/kuotunyu/vllm-single-gpu-slo-lab",
            ]
        )
        self.play(FadeIn(card), run_time=0.8)
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
        left, full = -2.6, 6.0
        zero = Line([left, 1.6, 0], [left, -1.7, 0], color=MUTED, stroke_width=1)
        one = Line([left + full, 1.6, 0], [left + full, -1.7, 0], color=MUTED, stroke_width=1)
        self.play(
            Create(zero),
            Create(one),
            FadeIn(text("接受率 0", 16, MUTED).next_to(zero, DOWN, buff=0.1)),
            FadeIn(text("1.0", 16, MUTED).next_to(one, DOWN, buff=0.1)),
            run_time=0.4,
        )
        for i, cell in enumerate(data.cells):
            y = 1.1 - 1.2 * i
            label = text(cell.label, 24, COLORS[cell.name])
            label.move_to([0, y, 0]).align_to([-6.6, 0, 0], LEFT)
            bar = _bar(full * cell.acceptance, 0.45, COLORS[cell.name])
            bar.align_to([left, 0, 0], LEFT).set_y(y)
            val = text(
                f"接受率 {cell.acceptance:.2f} · 平均接受長度 {cell.mean_acceptance_length:.2f} token／步",
                20,
            )
            val.next_to(bar, RIGHT, buff=0.2)
            self.play(FadeIn(label), run_time=0.3)
            self.play(GrowFromEdge(bar, LEFT), run_time=0.8)
            self.play(FadeIn(val), run_time=0.3)
        self.wait(2.5)

    # ---- 2. closed-loop throughput ratio by concurrency (15 s)
    def closed_loop(self, data: SpecData) -> None:
        head = text(
            "closed-loop 吞吐：加速 cell ÷ 同 family 的 none cell（rps 比）", 26, weight="BOLD"
        )
        head.to_edge(UP, buff=0.35)
        self.play(FadeIn(head), run_time=0.6)
        legend = VGroup(*[text(c.label, 18, COLORS[c.name]) for c in data.cells])
        legend.arrange(RIGHT, buff=0.6).next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(legend), run_time=0.4)
        base_y, unit = -1.7, 1.8  # 1.0x = 1.8 scene units
        group_x = [-4.8, -2.4, 0.0, 2.4, 4.8]
        bar_w, gap = 0.5, 0.08
        axis = Line([-6.4, base_y, 0], [6.4, base_y, 0], color=MUTED, stroke_width=1)
        one = Line([-6.4, base_y + unit, 0], [6.4, base_y + unit, 0], color=MUTED, stroke_width=1)
        one_lab = text("1.0×（與 none 相同）", 16, MUTED)
        one_lab.next_to(one, UP, buff=0.05).align_to([6.4, 0, 0], RIGHT)
        self.play(Create(axis), Create(one), FadeIn(one_lab), run_time=0.6)
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
                bar = _bar(bar_w, unit * ratio, COLORS[cell.name])
                bar.move_to([x, base_y + unit * ratio / 2, 0])
                val = text(f"{ratio:.2f}", 16, INK).next_to(bar, UP, buff=0.05)
                self.play(GrowFromEdge(bar, DOWN), FadeIn(val), run_time=0.45)
        note = text("單流與小批次變快（1.12 到 1.72 倍），c = 128 起吞吐反轉為 0.6 到 0.85 倍", 22)
        note.next_to(legend, DOWN, buff=0.25)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(3.0)

    # ---- 3. open-loop paired differences (15 s)
    def open_loop(self, data: SpecData) -> None:
        head = text(
            "open-loop 同 rate、同 seed 配對：TPOT 中位數與 p95 對 none 的差（ms）",
            26,
            weight="BOLD",
        )
        head.to_edge(UP, buff=0.35)
        key = text("左 = 中位數差，右 = p95 差；綠 = 變快，紅 = 變慢；三個 seed 同號", 18, MUTED)
        key.next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(head), FadeIn(key), run_time=0.6)
        panel_x = [-4.6, 0.0, 4.6]
        scale = 0.13  # scene units per ms
        base_y = 0.1
        for px, cell in zip(panel_x, data.cells, strict=True):
            title = text(cell.label, 22, COLORS[cell.name]).move_to([px, 2.1, 0])
            axis = Line([px - 1.9, base_y, 0], [px + 1.9, base_y, 0], color=MUTED, stroke_width=1)
            self.play(FadeIn(title), Create(axis), run_time=0.3)
            for j, p in enumerate(cell.open):
                x = px - 1.3 + j * 1.3
                pairs = (
                    (-0.22, p.tpot_p50_diff_ms, GREEN if p.tpot_p50_diff_ms < 0 else RED),
                    (0.22, p.tpot_p95_diff_ms, RED if p.tpot_p95_diff_ms > 0 else GREEN),
                )
                for dx, val, color in pairs:
                    h = abs(val) * scale
                    bar = _bar(0.36, h, color)
                    bar.move_to([x + dx, base_y + (h / 2 if val >= 0 else -h / 2), 0])
                    lab = text(f"{val:+.1f}", 14, INK)
                    lab.next_to(bar, UP if val >= 0 else DOWN, buff=0.04)
                    self.play(
                        GrowFromEdge(bar, DOWN if val >= 0 else UP), FadeIn(lab), run_time=0.25
                    )
                rate = text(f"{p.offered_rps:g} rps", 14, MUTED).move_to([x, base_y - 1.45, 0])
                self.play(FadeIn(rate), run_time=0.15)
            att = text("attainment 差 0", 16, MUTED).move_to([px, base_y - 1.85, 0])
            self.play(FadeIn(att), run_time=0.2)
        self.wait(3.0)

    # ---- 4. conclusion (8 s)
    def conclusion(self, data: SpecData) -> None:
        lines = [
            "單流與小批次變快；c ≥ 128 吞吐反轉；同 rate 的中位數變快、p95 變慢",
            "以 p95 定義的 SLO 下，attainment 與粗網格 r_SLO 都和 none 相同：容量沒有增加",
        ]
        head = VGroup(*[text(s, 26, weight="BOLD") for s in lines]).arrange(DOWN, buff=0.2)
        head.to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.8)
        rows = VGroup()
        for cell in data.cells:
            ratio = cell.r_sat / cell.base_r_sat
            rows.add(
                text(
                    f"{cell.label}：r_sat {cell.r_sat:.1f} rps（none {cell.base_r_sat:.1f}，{ratio:.2f}×）",
                    24,
                    COLORS[cell.name],
                )
            )
        rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT).next_to(head, DOWN, buff=0.8)
        for r in rows:
            self.play(FadeIn(r), run_time=0.4)
        self.wait(3.5)
