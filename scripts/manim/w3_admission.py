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
    always_redraw,
    linear,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w3_data import POLICIES, TRACE_END_S, Lane, ReplayData, load_replay

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
BAR_MAX_W = 4.2
BAR_H = 0.5
LANE_Y = (1.05, -0.45, -1.95)
LABEL_LEFT_X = -6.9
BAR_LEFT_X = -6.9
READOUT_X = 3.9
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
            label_constructor=Text,
        ).shift(UP * 2.45)
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
        caption = _t(
            "到達率：0.5× → 1.5×（5 分鐘）→ 0.5× r_sat；橫軸為 trace 秒數", 20, MUTED
        ).next_to(shades, UP, buff=0.55)
        self.play(FadeIn(shades), Create(axis), run_time=1.2)
        self.play(FadeIn(caption), run_time=0.4)
        for s in steps:
            self.play(GrowFromEdge(s[0], LEFT), FadeIn(s[1]), run_time=0.6)
        lanes: dict[str, dict] = {}
        for policy, y in zip(POLICIES, LANE_Y, strict=True):
            label = _t(LABELS[policy], 22, COLORS[policy])
            label.move_to([0, y + 0.5, 0]).align_to([LABEL_LEFT_X, 0, 0], LEFT)
            anchor = Line(
                [BAR_LEFT_X, y - BAR_H / 2, 0],
                [BAR_LEFT_X, y + BAR_H / 2, 0],
                color=MUTED,
                stroke_width=2,
            )
            lanes[policy] = {"y": y, "label": label, "anchor": anchor}
            self.play(FadeIn(label), Create(anchor), run_time=0.45)
        scale_note = _t(
            "佇列長條為對數尺度（vLLM waiting + shim waiting）；讀數為近 30 s 的滾動值", 16, MUTED
        )
        scale_note.to_edge(DOWN, buff=0.25)
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
        clock_label = _t("t（秒）=", 20, MUTED).move_to([5.35, axis.get_y() + 1.55, 0])
        clock = Integer(0, font_size=24, color=INK, mob_class=Text)
        clock.add_updater(
            lambda m: m.set_value(int(t.get_value())).next_to(clock_label, RIGHT, buff=0.12)
        )
        self.add(cursor, clock_label, clock)
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
            qnum = Integer(0, font_size=20, color=INK, mob_class=Text)

            def upd_q(m, lane=lane, y=y, bar=bar):
                m.set_value(int(lane.queue_at(t.get_value())))
                m.next_to(bar, RIGHT, buff=0.12).set_y(y)

            qnum.add_updater(upd_q)
            att = DecimalNumber(0, num_decimal_places=2, font_size=26, color=INK, mob_class=Text)
            ttft = DecimalNumber(0, num_decimal_places=2, font_size=26, color=INK, mob_class=Text)
            rej = Integer(0, font_size=26, color=INK, mob_class=Text)
            cols = VGroup(
                VGroup(_t("attainment", 15, MUTED), att).arrange(DOWN, buff=0.06),
                VGroup(_t("TTFT p95（s）", 15, MUTED), ttft).arrange(DOWN, buff=0.06),
                VGroup(_t("429 累計", 15, MUTED), rej).arrange(DOWN, buff=0.06),
            ).arrange(RIGHT, buff=0.35, aligned_edge=UP)
            cols.move_to([READOUT_X + 0.6, y, 0])
            anchors = [c[1].get_center() for c in cols]

            def upd_att(m, lane=lane, pos=anchors[0]):
                b = lane.bucket_at(t.get_value())
                a = b.attainment
                m.set_value(a if a is not None else 0.0)
                if a is None:
                    m.set_color(INK)
                else:
                    m.set_color(GREEN if a >= 0.95 else (RED if a < 0.5 else INK))
                m.move_to(pos)

            def upd_ttft(m, lane=lane, pos=anchors[1]):
                b = lane.bucket_at(t.get_value())
                v = b.ttft_p95_s
                m.set_value(v if v is not None else 0.0)
                m.set_color(RED if v is not None and v > 1.0 else INK)
                m.move_to(pos)

            def upd_rej(m, lane=lane, pos=anchors[2]):
                i = lane.raw_bucket_index(t.get_value())
                m.set_value(sum(b.rejected for b in lane.raw_buckets[: i + 1]))
                m.move_to(pos)

            att.add_updater(upd_att)
            ttft.add_updater(upd_ttft)
            rej.add_updater(upd_rej)
            sparks = self._sparks(lane, y, t, data.queue_max)
            dyn.add(bar, qnum, cols, sparks)
        self.add(dyn)
        burst_end = next(p.end_s for p in data.phases if p.name == "burst")
        self.play(t.animate.set_value(burst_end), run_time=REPLAY_TO_BURST_END_S, rate_func=linear)
        self.play(t.animate.set_value(TRACE_END_S), run_time=REPLAY_RECOVERY_S, rate_func=linear)
        stage["dyn"] = dyn
        stage["cursor"] = (cursor, clock_label, clock)

    def _sparks(self, lane: Lane, y: float, t: ValueTracker, queue_max: float) -> VGroup:
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
                    x0 = BAR_LEFT_X + _log_width(lane.queue_at(t.get_value()), queue_max) + 0.9
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
        note = VGroup(*[_t(s, 22, INK) for s in lines]).arrange(DOWN, buff=0.12)
        note.move_to([0, -3.1, 0])
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
        table = Table(
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
        table.scale(0.8).next_to(head, DOWN, buff=0.5)
        for i, p in enumerate(POLICIES):
            table.get_rows()[i + 1][0].set_color(COLORS[p])
            a = sb[p]["attainment"]
            table.get_rows()[i + 1][1].set_color(GREEN if a >= 0.5 else RED)
        self.play(FadeIn(head), run_time=0.5)
        self.play(
            Create(table.get_horizontal_lines()),
            Create(table.get_vertical_lines()),
            run_time=0.8,
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
            "同一條到達序列，只改 hard cap 的上限：C = 256 → C = 192（ADR 0018）",
            26,
            weight="BOLD",
        ).to_edge(UP, buff=0.6)
        self.play(FadeIn(head), run_time=0.5)
        left_x = -6.0
        scale = 5.0 / 60.0  # scene units per ms of TPOT p95
        rows = []
        for key, label, y in (
            ("c256", "C = 256（closed-loop 的平台）", 1.0),
            ("c192", "C = 192", -0.9),
        ):
            m = data.coda[key]
            tpot_ms = 1000 * m["tpot_p95_burst_s"]
            over = tpot_ms > 50
            lab = _t(label, 26, INK).move_to([0, y + 0.7, 0]).align_to([left_x, 0, 0], LEFT)
            bar = Rectangle(
                width=tpot_ms * scale,
                height=0.5,
                fill_color=RED if over else GREEN,
                fill_opacity=0.9,
                stroke_width=0,
            )
            bar.align_to([left_x, 0, 0], LEFT).set_y(y)
            val = _t(f"突發段 TPOT p95 {tpot_ms:.1f} ms", 24, RED if over else GREEN)
            val.next_to(bar, RIGHT, buff=0.2)
            nums = _t(
                f"突發段 attainment {m['attainment_burst']:.2f} · 整段 {m['attainment']:.2f} · "
                f"拒絕率 {100 * m['rejection_rate']:.1f} %",
                22,
                MUTED,
            )
            nums.move_to([0, y - 0.6, 0]).align_to([left_x, 0, 0], LEFT)
            rows.append((lab, bar, val, nums))
        thr_x = left_x + 50 * scale
        thr = Line([thr_x, 1.35, 0], [thr_x, -1.9, 0], color=RED, stroke_width=2)
        thr_lab = _t("SLO 50 ms", 18, RED).next_to(thr, DOWN, buff=0.08)
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
            _t("示意動畫：數字出自 analysis/tables/w3-fp8-admission 與 w3-fp8-c192-admission", 24),
            _t("由 make reproduce 從提交的證據重建；ADR 0013、0018", 24),
            _t("github.com/kuotunyu/vllm-single-gpu-slo-lab", 28, MUTED),
        ]
        g = VGroup(*lines).arrange(DOWN, buff=0.3)
        self.play(FadeIn(g), run_time=0.8)
        self.wait(3.2)
