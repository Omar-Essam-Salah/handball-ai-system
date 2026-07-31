"""
Biomechanics analyzer — per-player technique metrics from YOLOv8-pose keypoints.

Consumes the 17-point COCO skeleton already attached to every player track and
derives broadcast-style biomechanical readouts:

    • jump height        — feet lift above their own ground baseline (cm)
    • throwing-arm angle — elbow angle (shoulder–elbow–wrist) of the raised arm
    • release height     — wrist height above the hip, normalised by body height
    • trunk torque       — angular separation between the shoulder line and hip
                            line (the "X-factor" rotational-power proxy)

Scale: a player's pixel bounding-box height is mapped to an assumed real stature
(DEFAULT_STATURE_CM) to convert pixels → centimetres. This is an estimate (no
per-player calibration), but consistent enough for relative technique comparison.

Stateful per track-id: ground baseline + peak jump are tracked over time.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

# COCO-17 indices
NOSE = 0
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16

KP_CONF = 0.30
DEFAULT_STATURE_CM = 185.0


def _angle(a, b, c) -> float | None:
    """Angle at point b (degrees) for the a-b-c joint."""
    if a is None or b is None or c is None:
        return None
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    na = math.hypot(bax, bay); nc = math.hypot(bcx, bcy)
    if na < 1e-6 or nc < 1e-6:
        return None
    cosang = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (na * nc)))
    return math.degrees(math.acos(cosang))


def _line_angle(p, q) -> float | None:
    if p is None or q is None:
        return None
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


@dataclass
class _TrackState:
    foot_baseline: float = 0.0           # ground level (max foot-y seen, robust)
    foot_hist: deque = field(default_factory=lambda: deque(maxlen=60))
    peak_jump_cm: float = 0.0
    airborne: bool = False


class BiomechAnalyzer:
    JUMP_TRIGGER_CM = 8.0    # ignore sub-8cm noise as "jumping"

    def __init__(self, stature_cm: float = DEFAULT_STATURE_CM):
        self.stature_cm = float(stature_cm)
        self._state: dict[int, _TrackState] = {}

    @staticmethod
    def _kp(kpts, idx):
        if kpts is None or len(kpts) <= idx:
            return None
        k = kpts[idx]
        return (float(k[0]), float(k[1])) if float(k[2]) >= KP_CONF else None

    def update(self, track_id: int, kpts, px_height: float) -> dict | None:
        """Update one player's biomech state; return current metrics (or None
        if the pose is too incomplete to analyse)."""
        if kpts is None or len(kpts) < 17 or px_height < 20:
            return None
        st = self._state.setdefault(track_id, _TrackState())
        cm_per_px = self.stature_cm / float(px_height)

        l_sh, r_sh = self._kp(kpts, L_SH), self._kp(kpts, R_SH)
        l_hip, r_hip = self._kp(kpts, L_HIP), self._kp(kpts, R_HIP)
        l_an, r_an = self._kp(kpts, L_ANK), self._kp(kpts, R_ANK)

        # ── Jump height ──
        foot_pts = [p[1] for p in (l_an, r_an) if p is not None]
        jump_cm = 0.0
        if foot_pts:
            foot_y = max(foot_pts)                    # lowest foot (largest y)
            st.foot_hist.append(foot_y)
            # ground baseline = robust max (deepest the feet have been)
            st.foot_baseline = max(st.foot_hist) if st.foot_hist else foot_y
            jump_px = max(0.0, st.foot_baseline - foot_y)
            jump_cm = jump_px * cm_per_px
            st.airborne = jump_cm >= self.JUMP_TRIGGER_CM
            if jump_cm > st.peak_jump_cm:
                st.peak_jump_cm = jump_cm

        # ── Throwing arm: pick the arm whose wrist is highest (smallest y) ──
        arms = []
        for sh, el, wr in ((self._kp(kpts, L_SH), self._kp(kpts, L_EL), self._kp(kpts, L_WR)),
                           (self._kp(kpts, R_SH), self._kp(kpts, R_EL), self._kp(kpts, R_WR))):
            if sh and el and wr:
                arms.append((wr[1], sh, el, wr))
        arm_angle = release_cm = None
        if arms:
            arms.sort(key=lambda a: a[0])             # smallest wrist-y first = highest
            _, sh, el, wr = arms[0]
            arm_angle = _angle(sh, el, wr)
            hip_y = None
            hips = [p[1] for p in (l_hip, r_hip) if p is not None]
            if hips:
                hip_y = sum(hips) / len(hips)
                release_cm = max(0.0, (hip_y - wr[1])) * cm_per_px  # wrist above hip

        # ── Trunk torque: shoulder line vs hip line separation ──
        torque = None
        sa = _line_angle(l_sh, r_sh)
        ha = _line_angle(l_hip, r_hip)
        if sa is not None and ha is not None:
            d = abs(sa - ha) % 180.0
            torque = min(d, 180.0 - d)

        return {
            "jump_cm":     round(jump_cm, 1),
            "peak_jump_cm": round(st.peak_jump_cm, 1),
            "airborne":    st.airborne,
            "arm_angle":   round(arm_angle, 0) if arm_angle is not None else None,
            "release_cm":  round(release_cm, 1) if release_cm is not None else None,
            "torque":      round(torque, 0) if torque is not None else None,
        }

    def drop(self, active_ids: set[int]) -> None:
        """Forget state for tracks that are gone (keeps memory bounded)."""
        for tid in [t for t in self._state if t not in active_ids]:
            self._state.pop(tid, None)
