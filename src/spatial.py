"""
Spatial analytics — heatmaps, team shape (formations), and shot maps.

All operate in IMAGE space (pixels). If a homography is available the pipeline
can pass court coordinates instead, but pixel space is enough for the live
overlays and relative tactical readouts.
"""
from __future__ import annotations

import math
import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────
class HeatMap:
    """Decaying 2-D occupancy accumulator → JET overlay. One per team (or all)."""

    def __init__(self, frame_w: int, frame_h: int, cell: int = 16, decay: float = 0.995):
        self.cell = max(8, int(cell))
        self.gw = max(1, frame_w // self.cell)
        self.gh = max(1, frame_h // self.cell)
        self.grid = np.zeros((self.gh, self.gw), np.float32)
        self.decay = float(decay)

    def add(self, x: float, y: float, weight: float = 1.0) -> None:
        gx, gy = int(x // self.cell), int(y // self.cell)
        if 0 <= gx < self.gw and 0 <= gy < self.gh:
            self.grid[gy, gx] += weight

    def step(self) -> None:
        self.grid *= self.decay

    def render(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        m = float(self.grid.max())
        if m < 1e-3:
            return frame
        norm = np.clip(self.grid / m * 255.0, 0, 255).astype(np.uint8)
        big = cv2.resize(norm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        k = max(3, int(self.cell) | 1)
        big = cv2.GaussianBlur(big, (k, k), 0)
        cm = cv2.applyColorMap(big, cv2.COLORMAP_JET)
        mask = (big > 14)[..., None]
        blended = cv2.addWeighted(frame, 1.0 - alpha, cm, alpha, 0)
        return np.where(mask, blended, frame)


# ──────────────────────────────────────────────────────────────────────────
class TeamShape:
    """Per-team formation metrics: centroid, spread, width/depth, defensive line."""

    @staticmethod
    def analyse(players: list[tuple[float, float, str]], frame_w: int, frame_h: int) -> dict:
        out: dict = {}
        for team in ("team_a", "team_b"):
            pts = np.array([(x, y) for (x, y, t) in players if t == team], np.float32)
            if len(pts) < 2:
                out[team] = None
                continue
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            spread = float(np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1)).mean())
            width = float(pts[:, 0].max() - pts[:, 0].min())
            depth = float(pts[:, 1].max() - pts[:, 1].min())
            out[team] = {
                "centroid": [round(cx, 1), round(cy, 1)],
                "spread_px": round(spread, 1),
                "width_pct": round(100 * width / max(frame_w, 1), 1),
                "depth_pct": round(100 * depth / max(frame_h, 1), 1),
                "n": int(len(pts)),
                "compactness": round(100 * (1 - min(1.0, spread / (0.25 * frame_w))), 0),
            }
        return out


# ──────────────────────────────────────────────────────────────────────────
class ShotMap:
    """Collects shots/goals as normalised court points for a virtual-court plot."""

    def __init__(self, frame_w: int, frame_h: int):
        self.fw, self.fh = float(frame_w), float(frame_h)
        self.shots: list[dict] = []

    def add(self, x: float, y: float, team: str, is_goal: bool) -> None:
        self.shots.append({
            "x": round(max(0.0, min(1.0, x / self.fw)), 4),
            "y": round(max(0.0, min(1.0, y / self.fh)), 4),
            "team": team, "goal": bool(is_goal),
        })
        if len(self.shots) > 200:
            self.shots = self.shots[-200:]

    def data(self) -> list[dict]:
        return list(self.shots)
