"""
ihf_notation.py — IHF-standard tactical drawing primitives (OpenCV).

Implements the International Handball Federation coaching-board notation on
video frames:

    Pass                dashed arrow      passer → receiver
    Player movement     solid arrow       along the movement vector
    Shot on goal        thick double arrow along the shot trajectory
    Dribble             wavy (sinusoidal) line along the player's path
    Zones               shaded translucent polygons with a label
    Defensive block     lines connecting adjacent defenders + spacing labels

All functions draw IN PLACE on a BGR uint8 frame and also return it, so they
compose:  draw_pass(draw_zone(img, ...), ...).

Coordinates are frame pixels (int or float).  No court knowledge here — pure
rendering.  Colors are BGR.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

# ── Default palette (BGR) ────────────────────────────────────────────────────
COL_PASS      = (60, 220, 255)    # amber       — pass arrows
COL_MOVE      = (255, 200, 80)    # light blue  — off-ball movement
COL_SHOT      = (60, 60, 255)     # red         — shot trajectory
COL_DRIBBLE   = (80, 255, 160)    # green       — dribble path
COL_BLOCK     = (255, 120, 255)   # magenta     — defensive block links
COL_ZONE_CREA = (50, 200, 255)    # amber       — creation / assist zone
COL_ZONE_FIN  = (80, 80, 255)     # red         — finishing zone
COL_VULN      = (0, 140, 255)     # orange      — vulnerable (uncovered) space
COL_ERROR     = (0, 0, 255)       # red         — positioning error / defensive gap
COL_HIGHLIGHT = (0, 235, 255)     # yellow      — flagged-player highlight box

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([1.0, 0.0])


def _arrow_head(img, tip, direction, size, color, thickness):
    """Open V-shaped arrow head at `tip`, pointing along `direction`."""
    d = _unit(np.asarray(direction, np.float64))
    for sign in (+1, -1):
        ang = math.radians(28) * sign
        c, s = math.cos(ang), math.sin(ang)
        w = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])
        base = np.asarray(tip, np.float64) - w * size
        cv2.line(img, _pt(tip), _pt(base), color, thickness, cv2.LINE_AA)


# ── 1. Pass: dashed arrow ────────────────────────────────────────────────────

def draw_pass_arrow(img: np.ndarray, p_from, p_to,
                    color=COL_PASS, thickness: int = 2,
                    dash_len: int = 12, gap_len: int = 9) -> np.ndarray:
    """IHF pass notation: dashed arrow from the passer to the receiver."""
    a = np.asarray(p_from, np.float64)
    b = np.asarray(p_to,   np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 4:
        return img
    u = d / length
    head = 12 + 2 * thickness
    # dashes stop short of the head
    pos, end = 0.0, max(0.0, length - head)
    while pos < end:
        seg_end = min(pos + dash_len, end)
        cv2.line(img, _pt(a + u * pos), _pt(a + u * seg_end),
                 color, thickness, cv2.LINE_AA)
        pos = seg_end + gap_len
    _arrow_head(img, b, u, head, color, thickness)
    return img


# ── 2. Player movement: solid arrow ─────────────────────────────────────────

def draw_move_arrow(img: np.ndarray, p_from, p_to,
                    color=COL_MOVE, thickness: int = 2) -> np.ndarray:
    """IHF off-ball movement: plain solid arrow along the movement vector."""
    a = np.asarray(p_from, np.float64)
    b = np.asarray(p_to,   np.float64)
    if np.linalg.norm(b - a) < 4:
        return img
    cv2.line(img, _pt(a), _pt(b), color, thickness, cv2.LINE_AA)
    _arrow_head(img, b, b - a, 11 + 2 * thickness, color, thickness)
    return img


# ── 3. Shot: thick double arrow ──────────────────────────────────────────────

def draw_shot_arrow(img: np.ndarray, p_from, p_to,
                    color=COL_SHOT, thickness: int = 3,
                    rail_offset: int = 4) -> np.ndarray:
    """IHF shot notation: two parallel thick rails + one big arrow head."""
    a = np.asarray(p_from, np.float64)
    b = np.asarray(p_to,   np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 6:
        return img
    u = d / length
    n = np.array([-u[1], u[0]])                 # normal
    head = 18 + 2 * thickness
    b_short = b - u * head * 0.75               # rails stop before the head
    for sign in (+1, -1):
        off = n * rail_offset * sign
        cv2.line(img, _pt(a + off), _pt(b_short + off),
                 color, thickness, cv2.LINE_AA)
    _arrow_head(img, b, u, head, color, thickness + 1)
    return img


# ── 4. Dribble: wavy line ────────────────────────────────────────────────────

def draw_dribble_path(img: np.ndarray, path: Sequence,
                      color=COL_DRIBBLE, thickness: int = 2,
                      amplitude: float = 6.0, wavelength: float = 26.0) -> np.ndarray:
    """IHF dribble notation: sinusoidal wave rendered along the actual path.

    `path` is a polyline of (x, y) points (the player's trajectory).
    """
    pts = [np.asarray(p, np.float64) for p in path]
    if len(pts) < 2:
        return img
    # Resample the polyline at ~4 px steps, then displace along the normal.
    resampled: list[np.ndarray] = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        steps = max(1, int(seg_len / 4))
        for i in range(steps):
            resampled.append(a + seg * (i / steps))
    resampled.append(pts[-1])

    wave: list[tuple[int, int]] = []
    dist = 0.0
    for i, p in enumerate(resampled):
        if i > 0:
            dist += float(np.linalg.norm(resampled[i] - resampled[i - 1]))
        nxt = resampled[min(i + 1, len(resampled) - 1)]
        prv = resampled[max(i - 1, 0)]
        u = _unit(nxt - prv)
        n = np.array([-u[1], u[0]])
        offset = amplitude * math.sin(2 * math.pi * dist / wavelength)
        wave.append(_pt(p + n * offset))
    cv2.polylines(img, [np.array(wave, np.int32)], False, color,
                  thickness, cv2.LINE_AA)
    return img


# ── 5. Shaded zones ──────────────────────────────────────────────────────────

def draw_zone(img: np.ndarray, polygon: Sequence, color,
              alpha: float = 0.28, label: str = "",
              border: bool = True) -> np.ndarray:
    """Translucent shaded polygon with an optional outlined label."""
    pts = np.array([_pt(p) for p in polygon], np.int32)
    if len(pts) < 3:
        return img
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    if border:
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    if label:
        cx, cy = pts.mean(axis=0).astype(int)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 2)
        org = (cx - tw // 2, cy)
        cv2.putText(img, label, org, _FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, org, _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def zone_around(points: Iterable, pad: float = 46.0) -> list[tuple[float, float]]:
    """Convex hull around `points`, padded outward — a quick tactical zone."""
    arr = np.array(list(points), np.float32).reshape(-1, 2)
    if len(arr) == 1:
        x, y = arr[0]
        return [(x - pad, y - pad), (x + pad, y - pad),
                (x + pad, y + pad), (x - pad, y + pad)]
    hull = cv2.convexHull(arr).reshape(-1, 2)
    c = hull.mean(axis=0)
    out = []
    for p in hull:
        d = _unit(p - c)
        out.append(tuple(p + d * pad))
    return out


# ── 6. Defensive block structure ────────────────────────────────────────────

def draw_defensive_block(img: np.ndarray, defender_pts: Sequence,
                         color=COL_BLOCK, thickness: int = 2,
                         show_spacing: bool = True,
                         px_per_m: Optional[float] = None) -> np.ndarray:
    """Connect adjacent defenders (sorted along the block) to show the wall
    structure and, optionally, print the gap size between each pair.

    `defender_pts`  — (x, y) foot positions of the defending team.
    `px_per_m`      — if given, gaps are labelled in metres, else in pixels.
    """
    pts = [np.asarray(p, np.float64) for p in defender_pts]
    if len(pts) < 2:
        return img
    # Sort along the principal axis of the block (usually across the court)
    arr = np.array(pts)
    mean = arr.mean(axis=0)
    axis = np.linalg.svd(arr - mean)[2][0]      # first right-singular vector
    order = np.argsort((arr - mean) @ axis)
    arr = arr[order]

    for a, b in zip(arr[:-1], arr[1:]):
        cv2.line(img, _pt(a), _pt(b), color, thickness, cv2.LINE_AA)
        if show_spacing:
            gap_px = float(np.linalg.norm(b - a))
            label = (f"{gap_px / px_per_m:.1f}m" if px_per_m
                     else f"{gap_px:.0f}px")
            mid = _pt((a + b) / 2 + np.array([0, -8]))
            cv2.putText(img, label, mid, _FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, mid, _FONT, 0.45, color, 1, cv2.LINE_AA)
    for p in arr:
        cv2.circle(img, _pt(p), 5, color, -1, cv2.LINE_AA)
        cv2.circle(img, _pt(p), 7, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# ── 7. Positioning-error marker (bold X) ────────────────────────────────────

def draw_x_marker(img: np.ndarray, point, size: int = 13,
                  color=COL_ERROR, thickness: int = 3) -> np.ndarray:
    """Bold X at `point` flagging a positioning/coverage error. Drawn with a
    black outline first so it stays legible over any court/kit color."""
    x, y = _pt(point)
    for th, col in ((thickness + 2, (0, 0, 0)), (thickness, color)):
        cv2.line(img, (x - size, y - size), (x + size, y + size), col, th, cv2.LINE_AA)
        cv2.line(img, (x - size, y + size), (x + size, y - size), col, th, cv2.LINE_AA)
    return img


# ── 8. Dashed ellipse (defensive-gap outline) ───────────────────────────────

def draw_dashed_ellipse(img: np.ndarray, center, axes,
                        color=COL_ERROR, thickness: int = 2,
                        angle: float = 0.0,
                        dash_deg: float = 14.0, gap_deg: float = 10.0) -> np.ndarray:
    """Dashed ellipse outline — IHF notation for a highlighted danger/gap
    zone that isn't a filled polygon (e.g. an open defensive corridor)."""
    a = 0.0
    while a < 360.0:
        a2 = min(a + dash_deg, 360.0)
        cv2.ellipse(img, _pt(center), (int(axes[0]), int(axes[1])), angle,
                    a, a2, color, thickness, cv2.LINE_AA)
        a += dash_deg + gap_deg
    return img


# ── 9. Passing lane (creation-zone corridor) ────────────────────────────────

def passing_lane_polygon(p_from, p_to, width_start: float = 26.0,
                         width_end: float = 52.0) -> list[tuple[float, float]]:
    """Tapered corridor polygon from passer to receiver — the created lane."""
    a = np.asarray(p_from, np.float64)
    b = np.asarray(p_to,   np.float64)
    u = _unit(b - a)
    n = np.array([-u[1], u[0]])
    return [tuple(a + n * width_start), tuple(b + n * width_end),
            tuple(b - n * width_end), tuple(a - n * width_start)]
