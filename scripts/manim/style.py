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
