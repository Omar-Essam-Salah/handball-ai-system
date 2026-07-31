"""
defensive_gap_detector.py — deterministic geometry for the "image-2 style"
tactical annotation: biggest gap in the defensive block + which specific
defenders are responsible. No LLM involved — this is pure coordinate math
over our own tracked player positions, so it runs free, instantly, and on
every single play with zero API quota cost.

Two independent findings, mirroring the reference annotation exactly:

    1. GAP finding      — sort defenders along the block's principal axis
                          (same technique ihf_notation.draw_defensive_block
                          uses for spacing labels); the largest consecutive
                          gap is "ثغرة دفاعية كبيرة", and the two defenders
                          bounding it are flagged "سوء تمركز" if the gap
                          exceeds GAP_FLAG_PX.

    2. BEATEN-defender finding — an attacker who has advanced closer to the
                          goal line than the nearest defender assigned to
                          their lane, while that defender is still laterally
                          near them, is a coverage breakdown: "إخفاق في
                          التغطية" on that defender.

Both operate in FRAME PIXEL coordinates (this is a per-frame visual
annotation tool, not a match-long metric analysis — for that, see
transition_analysis.py, which already does metre-based coverage over time).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GAP_FLAG_PX      = 140.0    # consecutive-defender gap large enough to flag
BEATEN_MARGIN_PX = 60.0     # attacker must be this much deeper than the defender
BEATEN_LANE_PX    = 130.0   # ...and still within this lateral distance of them


@dataclass
class FlaggedPlayer:
    tid:    int
    point:  tuple[float, float]
    reason: str            # "gap" | "beaten"
    label:  str            # Arabic label to render


@dataclass
class GapFinding:
    gap_center: tuple[float, float]
    gap_axes:   tuple[float, float]     # ellipse (rx, ry) for rendering
    gap_px:     float
    flagged:    list[FlaggedPlayer] = field(default_factory=list)


def _sort_along_block(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order points along the block's principal axis (PCA via SVD) — same
    approach as ihf_notation.draw_defensive_block, kept independent here so
    this module has no rendering dependency."""
    mean = pts.mean(axis=0)
    axis = np.linalg.svd(pts - mean)[2][0]
    order = np.argsort((pts - mean) @ axis)
    return pts[order], order


def analyze_defensive_shape(
    defenders: list[dict],          # [{tid, cx, cy}, ...]
    attackers: list[dict],          # [{tid, cx, cy}, ...]
    goal_x: float,
    attack_direction: int = +1,     # +1: attackers advance toward +x; -1: toward -x
    gap_flag_px: float = GAP_FLAG_PX,
    beaten_margin_px: float = BEATEN_MARGIN_PX,
    beaten_lane_px: float = BEATEN_LANE_PX,
) -> GapFinding | None:
    """Pure geometry — no video, no LLM. Returns None if there aren't enough
    defenders tracked this frame to say anything meaningful."""
    if len(defenders) < 3:
        return None

    pts = np.array([[d["cx"], d["cy"]] for d in defenders], np.float64)
    tids = [d["tid"] for d in defenders]
    sorted_pts, order = _sort_along_block(pts)
    sorted_tids = [tids[i] for i in order]

    gaps = np.linalg.norm(np.diff(sorted_pts, axis=0), axis=1)
    i = int(np.argmax(gaps))
    a, b = sorted_pts[i], sorted_pts[i + 1]
    gap_px = float(gaps[i])
    mid = tuple(((a + b) / 2).tolist())

    finding = GapFinding(
        gap_center=mid,
        gap_axes=(max(30.0, gap_px / 2), max(24.0, gap_px / 3.2)),
        gap_px=gap_px,
    )

    if gap_px >= gap_flag_px:
        for idx, pt in ((i, a), (i + 1, b)):
            finding.flagged.append(FlaggedPlayer(
                tid=sorted_tids[idx], point=tuple(pt.tolist()),
                reason="gap", label="سوء تمركز"))

    def depth(x: float) -> float:
        return x * attack_direction

    already_flagged = {f.tid for f in finding.flagged}
    for atk in attackers:
        ax, ay = atk["cx"], atk["cy"]
        # nearest defender by lateral (perpendicular-ish) distance
        d_idx = int(np.argmin(np.abs(pts[:, 1] - ay))) if len(pts) else None
        if d_idx is None:
            continue
        dx, dy = pts[d_idx]
        lateral = abs(dy - ay)
        if (depth(ax) > depth(dx) + beaten_margin_px and lateral <= beaten_lane_px
                and tids[d_idx] not in already_flagged):
            finding.flagged.append(FlaggedPlayer(
                tid=tids[d_idx], point=(float(dx), float(dy)),
                reason="beaten", label="إخفاق في التغطية"))
            already_flagged.add(tids[d_idx])

    return finding
