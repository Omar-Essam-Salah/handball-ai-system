"""
gk_kpi.py — Goalkeeper performance KPI tracker with a persistent overlay.

    Save% = saves / shots_on_target × 100
    shots_on_target = saves + goals          (off-target shots excluded)

Feed it resolved MatchEvents from HandballEngine:

    kpi = GoalkeeperKPI()
    for ev in new_events:
        kpi.register_event(ev.event_type.value, ev.team, ev.data)
    frame = kpi.draw_overlay(frame)

Event semantics (attacker-centric, as HandballEngine emits them):
    "goal"        team = SCORING team      → shot ON TARGET conceded by the
                                             OTHER team's goalkeeper
    "save"        team = SAVING team       → save FOR that team's goalkeeper
    "missed_out"  team = shooting team     → off target, keeper untouched
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class _GKStats:
    saves:      int = 0
    goals:      int = 0   # conceded
    off_target: int = 0   # shots at this GK's goal that missed the frame

    @property
    def on_target(self) -> int:
        return self.saves + self.goals

    @property
    def save_pct(self) -> float:
        return 100.0 * self.saves / self.on_target if self.on_target else 0.0


class GoalkeeperKPI:
    """Per-team goalkeeper KPI with a broadcast-style persistent panel."""

    FLASH_S = 2.5          # highlight the panel briefly after each update

    def __init__(self, team_names: dict | None = None):
        self.stats = {"team_a": _GKStats(), "team_b": _GKStats()}
        self.names = team_names or {"team_a": "HOME", "team_b": "AWAY"}
        self._last_update = {"team_a": 0.0, "team_b": 0.0}
        self._last_event  = {"team_a": "",  "team_b": ""}

    @staticmethod
    def _other(team: str) -> str:
        return "team_b" if team == "team_a" else "team_a"

    # ── event ingestion ──────────────────────────────────────────────────────
    def register_event(self, event_type: str, team: str, data: dict | None = None) -> None:
        """`team` follows HandballEngine semantics (see module docstring)."""
        if team not in self.stats:
            return
        now = time.perf_counter()
        if event_type == "save":
            self.stats[team].saves += 1
            self._last_update[team] = now
            self._last_event[team]  = "SAVE"
        elif event_type == "goal":
            gk_team = self._other(team)           # conceding keeper
            self.stats[gk_team].goals += 1
            self._last_update[gk_team] = now
            self._last_event[gk_team]  = "GOAL CONCEDED"
        elif event_type == "missed_out":
            gk_team = self._other(team)           # shot was AT this keeper's goal
            self.stats[gk_team].off_target += 1
            self._last_update[gk_team] = now
            self._last_event[gk_team]  = "OFF TARGET"

    def snapshot(self) -> dict:
        """JSON-safe dict for the API / dashboard."""
        return {t: {"saves": s.saves, "goals_conceded": s.goals,
                    "off_target": s.off_target,
                    "shots_on_target": s.on_target,
                    "save_pct": round(s.save_pct, 1)}
                for t, s in self.stats.items()}

    # ── rendering ────────────────────────────────────────────────────────────
    def draw_overlay(self, frame: np.ndarray,
                     anchor: str = "tr", margin: int = 12) -> np.ndarray:
        """Persistent two-row GK panel.  anchor: tl/tr/bl/br corner."""
        h, w = frame.shape[:2]
        pw, row_h = 268, 34
        ph = row_h * 2 + 26
        x0 = margin if "l" in anchor else w - pw - margin
        y0 = margin if "t" in anchor else h - ph - margin

        panel = frame[y0:y0 + ph, x0:x0 + pw].copy()
        overlay = np.zeros_like(panel)
        overlay[:] = (18, 16, 12)
        cv2.addWeighted(overlay, 0.72, panel, 0.28, 0, dst=panel)
        frame[y0:y0 + ph, x0:x0 + pw] = panel

        cv2.putText(frame, "GOALKEEPER  SAVE %", (x0 + 10, y0 + 17),
                    _FONT, 0.45, (160, 200, 230), 1, cv2.LINE_AA)

        now = time.perf_counter()
        for i, team in enumerate(("team_a", "team_b")):
            s = self.stats[team]
            ry = y0 + 26 + i * row_h
            flash = (now - self._last_update[team]) < self.FLASH_S
            name_col = (255, 255, 255)
            team_col = (80, 190, 255) if team == "team_a" else (90, 90, 255)

            cv2.rectangle(frame, (x0 + 8, ry + 4), (x0 + 12, ry + row_h - 6),
                          team_col, -1)
            cv2.putText(frame, str(self.names.get(team, team))[:10],
                        (x0 + 20, ry + 22), _FONT, 0.55, name_col, 1, cv2.LINE_AA)

            pct_txt = f"{s.save_pct:4.0f}%"
            pct_col = (80, 255, 140) if s.save_pct >= 30 else (90, 200, 255)
            if flash:
                pct_col = (0, 255, 255)
            cv2.putText(frame, pct_txt, (x0 + 118, ry + 23),
                        _FONT, 0.72, pct_col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{s.saves}/{s.on_target}",
                        (x0 + 196, ry + 22), _FONT, 0.5,
                        (190, 190, 190), 1, cv2.LINE_AA)
            if flash and self._last_event[team]:
                cv2.putText(frame, self._last_event[team],
                            (x0 + 20, ry + row_h - 1), _FONT, 0.36,
                            (0, 255, 255), 1, cv2.LINE_AA)
        return frame
