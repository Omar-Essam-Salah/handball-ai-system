"""
arabic_text.py — correct Arabic text rendering on OpenCV (BGR numpy) frames.

cv2.putText cannot shape Arabic: it draws isolated, unconnected letter forms
in logical (typing) order, which reads as garbled nonsense once more than one
letter is adjacent. This module shapes the text properly (arabic_reshaper
picks the correct joined glyph form per letter position) and reorders it for
right-to-left display (python-bidi), then rasterizes with Pillow — which,
unlike OpenCV, has real font shaping — and composites the result back onto
the BGR frame.

    from arabic_text import draw_text
    draw_text(frame, "سوء تمركز", (640, 300), size=22, color=(0, 0, 255))

Mixed Arabic/Latin/number strings (e.g. "المسدد رقم 84") work correctly —
arabic_reshaper + bidi handle the LTR numeral runs embedded in RTL text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import arabic_reshaper
import cv2
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

# Windows ships Arabic-capable glyphs in these; both are near-universal.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]
_font_cache: dict[tuple[str, int], "ImageFont.FreeTypeFont"] = {}


def _resolve_font_path(font_path: Optional[str]) -> str:
    candidates = [font_path] if font_path else []
    candidates += _FONT_CANDIDATES
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError(
        "No Arabic-capable TTF font found. Pass font_path= explicitly, or "
        "install one (e.g. Noto Naskh Arabic) and point font_path at it.")


def _get_font(size: int, font_path: Optional[str]) -> "ImageFont.FreeTypeFont":
    resolved = _resolve_font_path(font_path)
    key = (resolved, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(resolved, size)
    return _font_cache[key]


def shape(text: str) -> str:
    """Reshape + reorder Arabic text into correct visual (display) order.
    Safe to call on mixed Arabic/Latin/digit strings or pure-Latin strings."""
    return get_display(arabic_reshaper.reshape(text))


def text_size(text: str, size: int = 22, font_path: Optional[str] = None,
             stroke_width: int = 0) -> tuple[int, int]:
    """(width, height) the shaped text will occupy at this font size."""
    font = _get_font(size, font_path)
    tmp = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(tmp)
    box = draw.textbbox((0, 0), shape(text), font=font, stroke_width=stroke_width)
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    size: int = 22,
    color: tuple[int, int, int] = (255, 255, 255),   # BGR
    outline_color: tuple[int, int, int] = (0, 0, 0),  # BGR
    outline_px: int = 3,
    anchor: str = "la",           # PIL anchor: l/m/r + a/m/d (see Pillow docs)
    font_path: Optional[str] = None,
    already_shaped: bool = False,
) -> np.ndarray:
    """Draw Arabic/mixed text onto a BGR frame IN PLACE and return it.

    `org` is interpreted per `anchor` — default "la" = left, ascender
    (matches cv2.putText's baseline-left convention closely enough for HUD
    labels). Use anchor="ma" to center horizontally on org[0].
    """
    font = _get_font(size, font_path)
    shaped = text if already_shaped else shape(text)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(
        org, shaped, font=font,
        fill=tuple(int(c) for c in color[::-1]),               # BGR->RGB
        stroke_width=outline_px,
        stroke_fill=tuple(int(c) for c in outline_color[::-1]),
        anchor=anchor,
    )
    frame[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return frame


def wrap_text(text: str, max_width: int, size: int = 20,
             font_path: Optional[str] = None) -> list[str]:
    """Greedy word-wrap on the LOGICAL (unshaped) string, measuring each
    candidate line's shaped width. Must wrap before shaping — splitting an
    already-shaped/reordered string mid-line breaks letter joining."""
    words = text.split(" ")
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and text_size(trial, size, font_path)[0] > max_width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def draw_text_block(
    frame: np.ndarray,
    lines: list[str],
    org: tuple[int, int],
    size: int = 20,
    line_spacing: float = 1.35,
    color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
    outline_px: int = 3,
    anchor: str = "ra",
    font_path: Optional[str] = None,
    max_width: Optional[int] = None,
) -> np.ndarray:
    """Multi-line Arabic block (e.g. a report card). One PIL round-trip for
    the whole block — cheaper than calling draw_text per line.

    Right-anchored ("ra") by default: `org` is the block's TOP-RIGHT corner
    and text grows leftward from it, which is the only anchoring that can't
    silently run off-canvas for RTL paragraphs of unknown length — a
    left anchor ("la") extends unpredictably far to the right instead.
    Pass max_width to auto-wrap long paragraph lines (wrapped before
    shaping, so letter joining inside each wrapped line stays correct).
    """
    font = _get_font(size, font_path)
    if max_width is not None:
        expanded: list[str] = []
        for line in lines:
            expanded.extend(wrap_text(line, max_width, size, font_path) if line else [""])
        lines = expanded
    shaped_lines = [shape(t) for t in lines]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    x, y = org
    dy = size * line_spacing
    for i, line in enumerate(shaped_lines):
        draw.text(
            (x, y + i * dy), line, font=font,
            fill=tuple(int(c) for c in color[::-1]),
            stroke_width=outline_px,
            stroke_fill=tuple(int(c) for c in outline_color[::-1]),
            anchor=anchor,
        )
    frame[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return frame
