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
    "--gpu-memory-utilization 0.82，WSL2，與桌面共用的一張 RTX 4090，Qwen3-8B",
    "108 → 132 tokens，open-loop Poisson 每點 5 min × 3 seeds",
]


def _axes(x_max: float, y_max: float, y_step: float, y_decimals: int) -> Axes:
    return Axes(
        x_range=[0, x_max, 10],
        y_range=[0, y_max, y_step],
        x_length=10.5,
        y_length=4.2,
        axis_config={
            "include_numbers": True,
            "label_constructor": Text,
            "font_size": 18,
            "color": MUTED,
            "decimal_number_config": {"num_decimal_places": 0},
        },
        y_axis_config={"decimal_number_config": {"num_decimal_places": y_decimals}},
    ).shift(UP * 0.05)


def _graph(axes: Axes, points: list[tuple[float, float]], color: str):
    return axes.plot_line_graph(
        [r for r, _ in points],
        [v for _, v in points],
        line_color=color,
        add_vertex_dots=True,
        vertex_dot_radius=0.045,
        vertex_dot_style={"fill_color": color},
        stroke_width=3,
    )


class W2Knee(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG
        data = load_knee(ROOT)
        for section in (self.knee, self.tpot_bound, self.sensitivity, self.scoreboard):
            self.add(bottom_note(NOTE))
            section(data)
            self.clear()
        card = source_card(
            [
                "示意動畫：數字出自 analysis/tables/w2-*-open-loop、w2-*-closed-loop、w2-quality-paired",
                "由 make reproduce 從提交的證據重建；ADR 0007、0008、0009",
                "github.com/kuotunyu/vllm-single-gpu-slo-lab",
            ]
        )
        self.play(FadeIn(card), run_time=0.8)
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
            graph = _graph(axes, cell.attainment, COLORS[name])
            self.play(Create(graph["line_graph"]), FadeIn(graph["vertex_dots"]), run_time=2.2)
            marker = DashedLine(
                axes.c2p(cell.r_slo, 0), axes.c2p(cell.r_slo, 0.95), color=COLORS[name]
            )
            self.play(Create(marker), run_time=0.6)
            item = text(f"{cell.label}  r_SLO {cell.r_slo:g} rps", 20, COLORS[name])
            legend.add(item)
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            legend.move_to(axes.c2p(72, 0.62))
            self.play(FadeIn(item), run_time=0.3)
        self.wait(2.0)

    # ---- 2. the FP8 knee is TPOT-bound (12 s)
    def tpot_bound(self, data: KneeData) -> None:
        fp8 = data.cells["fp8"]
        head = text(
            "FP8 的 TPOT p95（ms，3 seeds 平均）對 offered rate：膝點是 TPOT，不是排隊",
            26,
            weight="BOLD",
        )
        head.to_edge(UP, buff=0.35)
        axes = _axes(50, 100, 25, 0)
        x_lab = text("offered rate（req/s）", 18, MUTED).next_to(axes.x_axis, DOWN, buff=0.35)
        pts = [(r, min(t, 100.0)) for r, t in fp8.tpot_p95_ms if r <= 50]
        graph = _graph(axes, pts, BLUE)
        slo = DashedLine(axes.c2p(0, 50), axes.c2p(50, 50), color=RED, dash_length=0.12)
        slo_lab = text("SLO 50 ms", 16, RED).next_to(slo, RIGHT, buff=0.1)
        self.play(FadeIn(head), Create(axes), FadeIn(x_lab), run_time=1.0)
        self.play(Create(slo), FadeIn(slo_lab), run_time=0.5)
        self.play(Create(graph["line_graph"]), FadeIn(graph["vertex_dots"]), run_time=2.5)
        marker = DashedLine(axes.c2p(fp8.r_slo, 0), axes.c2p(fp8.r_slo, 50), color=BLUE)
        tag = text(f"r_SLO {fp8.r_slo:g}", 16, BLUE).next_to(marker, UP, buff=0.08)
        self.play(Create(marker), FadeIn(tag), run_time=0.6)
        ttft_max = max(t for r, t in fp8.ttft_p95_s if r <= 30.57)
        lines = [
            f"26 到 31 rps 之間 TPOT p95 從 30 ms 爬過 50 ms；同一段 TTFT p95 最高 {ttft_max:.2f} s"
            "（門檻 1 s）",
            "容量由 decode 的步長決定：這張卡的膝點是 TPOT，不是佇列",
        ]
        note = VGroup(*[text(s, 22) for s in lines]).arrange(DOWN, buff=0.12)
        note.next_to(head, DOWN, buff=0.25)
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
        table = Table(
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
        table.scale(0.85).next_to(head, DOWN, buff=0.5)
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
            "r_SLO 對 TPOT 門檻敏感（30 ms → 24.0），對 TTFT 門檻不敏感（0.5 到 2 s 都一樣）", 22
        )
        note.next_to(table, DOWN, buff=0.6)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(3.5)

    # ---- 4. scoreboard (10 s)
    def scoreboard(self, data: KneeData) -> None:
        head = text("四精度總表（SLO：TTFT p95 ≤ 1 s 且 TPOT p95 ≤ 50 ms）", 28, weight="BOLD")
        head.to_edge(UP, buff=0.35)
        rows = []
        for name in CELLS:
            c = data.cells[name]
            if c.tmmlu_delta is None:
                delta = "—"
            else:
                delta = f"{100 * c.tmmlu_delta:+.2f} pts（p = {c.mcnemar_p:.3g}）"
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
        labels = [
            "精度",
            "r_SLO（rps）",
            "r_sat（rps）",
            "Wh／百萬 token",
            "TMMLU+",
            "對 BF16 配對差",
        ]
        table = Table(
            rows,
            col_labels=[text(s, 20, MUTED) for s in labels],
            element_to_mobject=lambda s: text(s, 22),
            include_outer_lines=False,
            line_config={"stroke_color": MUTED, "stroke_width": 1},
            h_buff=0.5,
            v_buff=0.35,
        )
        table.scale(0.78).next_to(head, DOWN, buff=0.5)
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
        note = text("FP8：SLO 容量是 BF16 的 2.55 倍、每 token 能耗 42 %、品質無法區分", 22)
        note.next_to(table, DOWN, buff=0.6)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(3.5)
