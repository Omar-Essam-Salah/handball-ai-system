"""
shot_prediction.py — multiple candidate shot-target arrows ("خيارات التسديد")
for an attacker who has the ball near the goal, BEFORE (or independent of)
the actual shot outcome.

This is deliberately geometric, not a learned/hallucinated prediction: given
the shooter's position, the goal mouth, the goalkeeper's real tracked
position, and any defenders standing in a shooting lane, it scores points
along the goal line by how OPEN they are (far from the keeper, not blocked
by a defender's body) and returns the top N as distinct candidate targets —
exactly the "he could shoot here, here, or here" breakdown a coach draws on
a tactics board. Each option is rendered as a thin, light dashed arrow,
visually distinct from the thick double arrow used for the shot that
actually happened (ihf_notation.draw_shot_arrow).

    from shot_prediction import compute_shot_options, render_shot_options

    options = compute_shot_options(
        shooter_point=(910, 380), goal_x=1250,
        goal_y_top=310, goal_y_bottom=450,
        goalkeeper_point=(1230, 390),
        defender_points=[(1050, 400)],
    )
    render_shot_options(frame, (910, 380), options)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from arabic_text import draw_text
from ihf_notation import draw_pass_arrow

COL_OPTION = (230, 210, 130)   # light cyan-ish blue (BGR) — distinct from the
                              # solid-red "actual shot" arrow and the orange
                              # "actual pass/lane" arrow already in use
BLOCK_PENALTY   = 500.0
BLOCK_RADIUS_PX = 38.0


def _point_to_segment_dist(a, b, p) -> float:
    a, b, p = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(p, np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def compute_shot_options(
    shooter_point: tuple[float, float],
    goal_x: float,
    goal_y_top: float,
    goal_y_bottom: float,
    goalkeeper_point: Optional[tuple[float, float]] = None,
    defender_points: Optional[list[tuple[float, float]]] = None,
    n_options: int = 3,
    min_spacing_px: float = 35.0,
    edge_margin_px: float = 10.0,
) -> list[tuple[float, float]]:
    """Return up to `n_options` (x, y) target points along the goal mouth,
    ranked by openness. Real geometry only — no learned model, so results
    are always explainable ("this option scored highest because the keeper
    was N px away and no defender's body was on that line")."""
    defender_points = defender_points or []
    lo, hi = goal_y_top + edge_margin_px, goal_y_bottom - edge_margin_px
    if hi <= lo:
        return []
    candidates = np.linspace(lo, hi, 25)

    scores = np.zeros(len(candidates))
    for i, y in enumerate(candidates):
        target = (goal_x, float(y))
        score = 0.0
        if goalkeeper_point is not None:
            score += float(np.hypot(goal_x - goalkeeper_point[0], y - goalkeeper_point[1]))
        for dpt in defender_points:
            if _point_to_segment_dist(shooter_point, target, dpt) < BLOCK_RADIUS_PX:
                score -= BLOCK_PENALTY
        # mild preference for the corners — realistic keeper-beating targets
        edge_bonus = min(abs(y - lo), abs(y - hi)) * -0.15 + (hi - lo) * 0.15
        scores[i] = score + edge_bonus

    order = np.argsort(scores)[::-1]
    chosen: list[tuple[float, float]] = []
    for idx in order:
        y = float(candidates[idx])
        if all(abs(y - c[1]) > min_spacing_px for c in chosen):
            chosen.append((goal_x, y))
        if len(chosen) >= n_options:
            break
    return chosen


def render_shot_options(
    frame: np.ndarray,
    shooter_point: tuple[float, float],
    options: list[tuple[float, float]],
    color=COL_OPTION,
    label_prefix: str = "خيار",
) -> np.ndarray:
    """Draw each candidate as a thin dashed arrow + a small numbered label at
    its tip. Call BEFORE drawing the real shot arrow so the real one
    (thicker, red, drawn on top) stays visually dominant."""
    for i, target in enumerate(options, start=1):
        draw_pass_arrow(frame, shooter_point, target, color=color, thickness=1,
                        dash_len=8, gap_len=6)
        tx, ty = target
        draw_text(frame, f"{label_prefix} {i}", (int(tx) - 6, int(ty)),
                  size=15, color=(255, 255, 255), outline_color=(0, 0, 0),
                  anchor="rm")
    return frame
