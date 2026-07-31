"""
tactical_annotator.py — composes defensive_gap_detector (geometry) +
ihf_notation (drawing primitives) + arabic_text (correct Arabic glyphs) into
the exact annotation style requested: yellow highlight boxes over flagged
defenders, a red X + Arabic label on each one, a dashed red ellipse over the
biggest defensive gap, and orange dashed lane arrows tracing the ball's path
to goal.

Entirely deterministic and free to run on every play — see
defensive_gap_detector.py's docstring for why no LLM is involved here.

    from tactical_annotator import render_defensive_analysis

    frame = render_defensive_analysis(
        frame, defenders, attackers, goal_x=1250, attack_direction=+1,
        ball_path=[(600, 380), (1150, 340)],
    )
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from arabic_text import draw_text
from defensive_gap_detector import GapFinding, analyze_defensive_shape
from ihf_notation import (COL_ERROR, COL_HIGHLIGHT, COL_VULN, draw_dashed_ellipse,
                          draw_pass_arrow, draw_x_marker, draw_zone, zone_around)

LABEL_SIZE  = 18
X_TO_PLAYER = 0.0            # X marker sits directly on the flagged point


def _cluster_points(points: list[tuple], radius: float = 220.0) -> list[list[int]]:
    """Greedy proximity clustering — groups flagged players close enough
    together to share one highlight box (matches the reference composition,
    where the two gap-boundary defenders share a box and the isolated
    beaten defender gets his own)."""
    n = len(points)
    assigned = [-1] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if assigned[i] != -1:
            continue
        cluster = [i]
        assigned[i] = len(clusters)
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            if np.hypot(points[i][0] - points[j][0],
                       points[i][1] - points[j][1]) <= radius:
                cluster.append(j)
                assigned[j] = assigned[i]
        clusters.append(cluster)
    return clusters


def render_defensive_analysis(
    frame: np.ndarray,
    defenders: list[dict],
    attackers: list[dict],
    goal_x: float,
    attack_direction: int = +1,
    ball_path: Optional[list[tuple]] = None,
    box_half_size: float = 70.0,
) -> np.ndarray:
    """Draw the full defensive-analysis overlay on `frame` (in place) and
    return it. No-ops (returns frame unchanged) if there isn't enough tracked
    data to say anything meaningful."""
    finding = analyze_defensive_shape(defenders, attackers, goal_x, attack_direction)
    if finding is None:
        return frame

    # 1. Attack lane arrows (ball path toward goal) — drawn first so player
    #    markers/labels sit on top of them, not the other way round.
    if ball_path and len(ball_path) >= 2:
        for p_from, p_to in zip(ball_path[:-1], ball_path[1:]):
            draw_pass_arrow(frame, p_from, p_to, color=COL_VULN, thickness=2)

    # 2. Yellow highlight boxes around clusters of flagged players
    if finding.flagged:
        pts = [f.point for f in finding.flagged]
        for cluster in _cluster_points(pts):
            box_pts = [f.point for k, f in enumerate(finding.flagged) if k in cluster]
            zone = zone_around(box_pts, pad=box_half_size)
            draw_zone(frame, zone, COL_HIGHLIGHT, alpha=0.30, border=True)

    # 3. Dashed red gap ellipse + label
    if finding.gap_px >= 1.0:
        draw_dashed_ellipse(frame, finding.gap_center, finding.gap_axes, COL_ERROR)
        lx, ly = finding.gap_center
        draw_text(frame, "ثغرة دفاعية كبيرة", (int(lx), int(ly + finding.gap_axes[1] + 6)),
                  size=LABEL_SIZE, color=(255, 255, 255), anchor="ma")

    # 4. X marker + label per flagged defender
    for f in finding.flagged:
        draw_x_marker(frame, f.point, color=COL_ERROR)
        px, py = f.point
        draw_text(frame, f.label, (int(px), int(py + 24)),
                  size=LABEL_SIZE, color=(255, 255, 255), anchor="ma")

    return frame
