"""
Ball Kalman Filter — trajectory-aware ball tracking.

Why this exists
───────────────
Frame-by-frame YOLO detection has fundamental limits no filter can fix:
    * the same body part fires as "ball" intermittently (head, elbow)
    * fast motion blurs make the ball invisible
    * painted markers fire identically every frame

A Kalman filter is the proven solution because it operates in TIME, not in
single frames. It tracks ball state (position + velocity + acceleration),
predicts where the ball SHOULD be next, and fuses detections with that
prediction. Detections far from the prediction are *gated out* automatically.

This is what every professional sports tracker uses (Hawk-Eye, TrackMan, Opta).

State vector
────────────
    x = [px, py, vx, vy, ax, ay]   (6-d)

    px, py     — ball pixel position
    vx, vy     — ball velocity (px / frame)
    ax, ay     — ball acceleration (constant-acceleration model)

Process model
─────────────
    p_{t+1} = p_t + v_t + 0.5 * a_t
    v_{t+1} = v_t + a_t
    a_{t+1} = a_t   (assumed constant — process noise absorbs change)

Measurement model
─────────────────
    z = [px, py]   (we measure position only; velocity inferred from history)

Gating
──────
A detection at distance d from the prediction has Mahalanobis distance:
    M = sqrt( (z - H @ x_pred).T @ S^{-1} @ (z - H @ x_pred) )
where S is the innovation covariance. If M > GATE_M, the detection is rejected
as "not consistent with the trajectory" — this is what kills head/marker FPs.

Usage
─────
    kf = BallKalman()

    # every frame:
    kf.predict()                          # extrapolate state
    if yolo_detection:
        if kf.gate(z=(cx, cy)):
            kf.update((cx, cy), conf=det.confidence)
    px, py = kf.position()                # use this everywhere instead of raw YOLO
"""

from __future__ import annotations
import math
from typing import Optional

import numpy as np


class BallKalman:
    # ── Tuning constants ────────────────────────────────────────────────
    # Process noise (how much the ball can deviate from constant-acceleration
    # model). Higher = more responsive, less smooth. Lower = more lag.
    Q_POS = 1.0      # position process noise
    Q_VEL = 4.0      # velocity process noise
    Q_ACC = 16.0     # acceleration process noise

    # Measurement noise (how much we trust YOLO). Smaller = trust YOLO more.
    R_BASE = 4.0     # at conf=1.0 we trust the detection completely
    R_AT_LOW_CONF = 60.0   # at conf=0.2 we trust it much less

    # Gating threshold — Mahalanobis distance. ~3.0 = 99.7% confidence ellipse.
    GATE_M = 3.5

    # Track lifecycle
    INIT_HITS_REQUIRED = 2     # need 2 consecutive close detections to "lock"
    MAX_MISSED_FRAMES  = 25    # after this many missed detections, reset filter

    def __init__(self):
        self._x = None             # 6-d state vector
        self._P = None             # 6x6 covariance
        self.locked = False        # True after INIT_HITS_REQUIRED hits
        self._init_hits = 0
        self._missed = 0
        # State-transition matrix (constant-acceleration, dt=1)
        self._F = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0,   0.5],
            [0, 0, 1, 0, 1,   0],
            [0, 0, 0, 1, 0,   1],
            [0, 0, 0, 0, 1,   0],
            [0, 0, 0, 0, 0,   1],
        ], dtype=np.float64)
        self._H = np.array([
            [1, 0, 0, 0, 0, 0],    # measure px
            [0, 1, 0, 0, 0, 0],    # measure py
        ], dtype=np.float64)
        # Process noise covariance
        self._Q = np.diag([self.Q_POS, self.Q_POS,
                           self.Q_VEL, self.Q_VEL,
                           self.Q_ACC, self.Q_ACC]).astype(np.float64)

    # ────────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self._x = None
        self._P = None
        self.locked = False
        self._init_hits = 0
        self._missed = 0

    @property
    def has_state(self) -> bool:
        return self._x is not None

    @property
    def missed_frames(self) -> int:
        """Consecutive predict() steps since the last update() (0 = fresh)."""
        return int(self._missed)

    def position(self) -> Optional[tuple[float, float]]:
        """Best estimate of current ball position. None if no track yet."""
        if self._x is None:
            return None
        return float(self._x[0]), float(self._x[1])

    def velocity(self) -> tuple[float, float]:
        if self._x is None:
            return 0.0, 0.0
        return float(self._x[2]), float(self._x[3])

    def speed(self) -> float:
        vx, vy = self.velocity()
        return math.hypot(vx, vy)

    def predict(self) -> None:
        """Step the filter forward one frame (no measurement)."""
        if self._x is None:
            return
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self._missed += 1
        if self._missed > self.MAX_MISSED_FRAMES:
            self.reset()

    def gate(self, z: tuple[float, float]) -> bool:
        """Returns True if a detection at z is consistent with our prediction.
        On the very first frame (no state yet) we always accept."""
        if self._x is None:
            return True
        z_arr = np.array([z[0], z[1]], dtype=np.float64)
        z_pred = self._H @ self._x
        S = self._H @ self._P @ self._H.T + np.eye(2) * self.R_BASE
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return True
        innov = z_arr - z_pred
        M2 = float(innov.T @ S_inv @ innov)
        if M2 < 0:
            return True
        return math.sqrt(M2) <= self.GATE_M

    def update(self, z: tuple[float, float], conf: float = 0.5) -> None:
        """Fuse a detection at z into the state estimate.
        Confidence scales the measurement noise."""
        z_arr = np.array([z[0], z[1]], dtype=np.float64)

        # Cold-start: initialise state from the first detection
        if self._x is None:
            self._x = np.array([z_arr[0], z_arr[1], 0, 0, 0, 0], dtype=np.float64)
            self._P = np.diag([10.0, 10.0, 100.0, 100.0, 200.0, 200.0])
            self._init_hits = 1
            self._missed = 0
            return

        # Measurement noise scales with confidence: low conf → big R
        c = max(0.05, min(1.0, conf))
        r = self.R_BASE + (self.R_AT_LOW_CONF - self.R_BASE) * (1.0 - c)
        R = np.eye(2) * r

        z_pred = self._H @ self._x
        innov = z_arr - z_pred
        S = self._H @ self._P @ self._H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self._P @ self._H.T @ S_inv
        self._x = self._x + K @ innov
        self._P = (np.eye(6) - K @ self._H) @ self._P
        self._missed = 0
        self._init_hits = min(self._init_hits + 1, self.INIT_HITS_REQUIRED + 5)
        if self._init_hits >= self.INIT_HITS_REQUIRED:
            self.locked = True

    def stats(self) -> dict:
        return {
            "locked":   self.locked,
            "has_state": self.has_state,
            "missed":   self._missed,
            "speed":    round(self.speed(), 1),
            "pos":      self.position(),
        }
