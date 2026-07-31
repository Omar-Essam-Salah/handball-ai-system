"""
transition_analysis.py — Attack→Defense transition (recovery) analysis.

On every POSSESSION_CHANGE the team that LOST the ball must sprint back and
rebuild its defensive block.  This module measures that recovery:

  • per-player recovery time  — frames until each defender crosses back into
    the defensive half (own-goal side of the halfway line)
  • laggard                   — the LAST player to recover (tagged on video)
  • vulnerable space          — the uncovered area inside the defensive half
    while the block is still forming, computed per frame as the region
    farther than R_cover from every recovered defender (grid method), plus
    Voronoi cells of the defenders for structure inspection (scipy).

Geometry runs in COURT METRES when a homography (pixel→metre, the
config/court_calibration.json one) is available, else in pixels with the
halfway line approximated by the frame's vertical centre line.

Usage (inside the pipeline loop):

    trans = TransitionAnalyzer(frame_w, frame_h, fps=25.0)
    trans.set_homography(H_px_to_m)          # optional, hot-reloadable

    for ev in new_events:
        if ev.event_type == EventType.POSSESSION_CHANGE:
            trans.on_turnover(new_attacking_team=ev.team, frame_n=frame_n)

    trans.update(frame_n, tracks)             # every frame
    frame = trans.draw(frame)                 # overlay while episode active
    if (ep := trans.pop_completed()):         # finished episode → log/report
        print(ep.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    from scipy.spatial import Voronoi
    _HAS_SCIPY = True
except ImportError:                                    # pragma: no cover
    _HAS_SCIPY = False

from ihf_notation import COL_VULN, draw_zone, draw_defensive_block

COURT_W_M, COURT_H_M = 40.0, 20.0
R_COVER_M      = 3.5        # a defender "covers" a 3.5 m radius disc
EPISODE_MAX_S  = 8.0        # give up analysing after this long
GRID_STEP_M    = 1.0        # coverage grid resolution
MIN_DEFENDERS  = 3          # need at least this many tracked defenders

_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class TransitionEpisode:
    defending_team: str
    start_frame:    int
    fps:            float
    recovery_frame: dict[int, int] = field(default_factory=dict)  # tid → fn
    laggard_tid:    Optional[int]  = None
    laggard_time_s: float = 0.0
    max_open_m2:    float = 0.0          # worst uncovered area seen
    end_frame:      Optional[int] = None
    done:           bool = False

    def summary(self) -> dict:
        return {
            "team": self.defending_team,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "recovered": len(self.recovery_frame),
            "laggard_tid": self.laggard_tid,
            "laggard_time_s": round(self.laggard_time_s, 2),
            "max_open_area_m2": round(self.max_open_m2, 1),
        }


class TransitionAnalyzer:
    def __init__(self, frame_w: int, frame_h: int, fps: float = 25.0,
                 team_a_defends_left: bool = True):
        self.fw, self.fh, self.fps = frame_w, frame_h, fps
        self.team_a_defends_left = team_a_defends_left
        self._H: Optional[np.ndarray] = None       # pixel → metre
        self._iH: Optional[np.ndarray] = None      # metre → pixel
        self.episode: Optional[TransitionEpisode] = None
        self._completed: list[TransitionEpisode] = []
        self._open_poly_px: list[np.ndarray] = []  # cached vulnerable polygons
        self._open_mask: Optional[np.ndarray] = None   # metre-grid open cells
        self._mask_x_off: float = 0.0
        self._defender_px: list[tuple] = []
        self._laggard_px: Optional[tuple] = None

    # ── configuration ────────────────────────────────────────────────────────
    def set_homography(self, H_px_to_m: Optional[np.ndarray]) -> None:
        self._H = None if H_px_to_m is None else np.asarray(H_px_to_m, np.float64)
        self._iH = None if self._H is None else np.linalg.inv(self._H)

    def flip_sides(self) -> None:
        self.team_a_defends_left = not self.team_a_defends_left

    # ── coordinate helpers ───────────────────────────────────────────────────
    def _px_to_m(self, pts: np.ndarray) -> np.ndarray:
        if self._H is None:
            # pixel fallback: pretend the frame IS the court
            return np.column_stack([pts[:, 0] * COURT_W_M / self.fw,
                                    pts[:, 1] * COURT_H_M / self.fh])
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64),
                                     self._H)
        return p.reshape(-1, 2)

    def _m_to_px(self, pts: np.ndarray) -> np.ndarray:
        if self._iH is None:
            return np.column_stack([pts[:, 0] * self.fw / COURT_W_M,
                                    pts[:, 1] * self.fh / COURT_H_M])
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64),
                                     self._iH)
        return p.reshape(-1, 2)

    def _defends_left(self, team: str) -> bool:
        return self.team_a_defends_left == (team == "team_a")

    # ── episode lifecycle ────────────────────────────────────────────────────
    def on_turnover(self, new_attacking_team: str, frame_n: int) -> None:
        """POSSESSION_CHANGE: `new_attacking_team` just WON the ball, so the
        other team is now transitioning back to defense."""
        defending = "team_b" if new_attacking_team == "team_a" else "team_a"
        if self.episode and not self.episode.done:
            self._finish(frame_n)                    # close any stale episode
        self.episode = TransitionEpisode(defending, frame_n, self.fps)

    def pop_completed(self) -> Optional[TransitionEpisode]:
        return self._completed.pop(0) if self._completed else None

    def _finish(self, frame_n: int) -> None:
        ep = self.episode
        if ep is None:
            return
        ep.end_frame, ep.done = frame_n, True
        self._completed.append(ep)
        self.episode = None
        self._open_poly_px, self._defender_px, self._laggard_px = [], [], None
        self._open_mask = None

    # ── per-frame update ─────────────────────────────────────────────────────
    def update(self, frame_n: int, tracks: list[dict]) -> None:
        ep = self.episode
        if ep is None or ep.done:
            return
        if (frame_n - ep.start_frame) / self.fps > EPISODE_MAX_S:
            self._finish(frame_n)
            return

        defenders = [t for t in tracks if t.get("team") == ep.defending_team]
        if len(defenders) < MIN_DEFENDERS:
            self._open_poly_px, self._defender_px = [], []
            self._open_mask = None
            return

        px = np.array([[t["cx"], t["cy"]] for t in defenders], np.float64)
        m  = self._px_to_m(px)
        left = self._defends_left(ep.defending_team)
        half_x = COURT_W_M / 2.0

        # recovery bookkeeping — a defender has "tracked back" once inside
        # their own half
        recovered_mask = (m[:, 0] < half_x) if left else (m[:, 0] > half_x)
        for t, rec in zip(defenders, recovered_mask):
            if rec and t["tid"] not in ep.recovery_frame:
                ep.recovery_frame[t["tid"]] = frame_n

        # laggard = slowest to recover so far (or still not back)
        not_back = [t for t, rec in zip(defenders, recovered_mask) if not rec]
        if not_back:
            # farthest from own half is the current laggard
            if left:
                lag = max(not_back, key=lambda t: self._px_to_m(
                    np.array([[t["cx"], t["cy"]]]))[0, 0])
            else:
                lag = min(not_back, key=lambda t: self._px_to_m(
                    np.array([[t["cx"], t["cy"]]]))[0, 0])
            ep.laggard_tid = lag["tid"]
            ep.laggard_time_s = (frame_n - ep.start_frame) / self.fps
            self._laggard_px = (lag["cx"], lag["cy"])
        else:
            self._laggard_px = None
            self._finish(frame_n)          # everyone back — episode complete
            return

        # vulnerable space: defensive-half grid cells > R_cover from every
        # RECOVERED defender
        self._compute_open_space(ep, m[recovered_mask], left)
        self._defender_px = [(t["cx"], t["cy"]) for t in defenders]

    def _compute_open_space(self, ep: TransitionEpisode,
                            back_m: np.ndarray, left: bool) -> None:
        xs = (np.arange(0, COURT_W_M / 2, GRID_STEP_M) if left
              else np.arange(COURT_W_M / 2, COURT_W_M, GRID_STEP_M))
        ys = np.arange(0, COURT_H_M, GRID_STEP_M)
        gx, gy = np.meshgrid(xs + GRID_STEP_M / 2, ys + GRID_STEP_M / 2)
        cells = np.column_stack([gx.ravel(), gy.ravel()])

        if len(back_m) == 0:
            open_mask = np.ones(len(cells), bool)
        else:
            d = np.linalg.norm(cells[:, None, :] - back_m[None, :, :], axis=2)
            open_mask = d.min(axis=1) > R_COVER_M

        open_area = float(open_mask.sum()) * GRID_STEP_M ** 2
        ep.max_open_m2 = max(ep.max_open_m2, open_area)

        # keep the raw grid mask for hole-accurate rendering in draw()
        self._open_mask   = open_mask.reshape(len(ys), len(xs))
        self._mask_x_off  = 0.0 if left else COURT_W_M / 2
        self._open_poly_px = []          # legacy path no longer used

    # ── Voronoi structure (for reports / playbook figures) ──────────────────
    def voronoi_cells_m(self, tracks: list[dict]) -> list[np.ndarray]:
        """Voronoi decomposition of the defending block (court metres).
        Useful for offline playbook figures; returns finite cells only."""
        ep = self.episode
        if not _HAS_SCIPY or ep is None:
            return []
        pts = np.array([[t["cx"], t["cy"]] for t in tracks
                        if t.get("team") == ep.defending_team], np.float64)
        if len(pts) < 4:
            return []
        m = self._px_to_m(pts)
        # mirror points across the court borders so border cells close
        mirrored = [m]
        for axis, bound in ((0, 0), (0, COURT_W_M), (1, 0), (1, COURT_H_M)):
            mm = m.copy()
            mm[:, axis] = 2 * bound - mm[:, axis]
            mirrored.append(mm)
        vor = Voronoi(np.vstack(mirrored))
        cells = []
        for i in range(len(m)):
            region = vor.regions[vor.point_region[i]]
            if -1 in region or not region:
                continue
            cells.append(vor.vertices[region])
        return cells

    # ── rendering ────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray) -> np.ndarray:
        ep = self.episode
        if ep is None or ep.done:
            return frame
        # vulnerable space: warp the metre-grid mask into frame space so
        # covered discs stay unshaded (hole-accurate, perspective-correct)
        if self._open_mask is not None and self._open_mask.any():
            ph, pw = self._open_mask.shape
            cell_px = 10                       # canvas resolution: 10 px per m
            canvas = cv2.resize((self._open_mask * 255).astype(np.uint8),
                                (pw * cell_px, ph * cell_px),
                                interpolation=cv2.INTER_NEAREST)
            canvas = cv2.GaussianBlur(canvas, (9, 9), 0)      # soften edges
            # canvas px → metres → frame px
            S = np.array([[GRID_STEP_M / cell_px, 0, self._mask_x_off],
                          [0, GRID_STEP_M / cell_px, 0],
                          [0, 0, 1]], np.float64)
            if self._iH is not None:
                M = self._iH @ S
            else:
                M = np.array([[self.fw / COURT_W_M, 0, 0],
                              [0, self.fh / COURT_H_M, 0],
                              [0, 0, 1]], np.float64) @ S
            warped = cv2.warpPerspective(canvas, M,
                                         (frame.shape[1], frame.shape[0]))
            sel = warped > 100
            if sel.any():
                col = np.array(COL_VULN, np.float64)
                frame[sel] = (frame[sel] * 0.66 + col * 0.34).astype(np.uint8)
                cts, _ = cv2.findContours((sel * 255).astype(np.uint8),
                                          cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, cts, -1, COL_VULN, 2, cv2.LINE_AA)
        # defensive block links between defenders already visible
        if len(self._defender_px) >= 2:
            draw_defensive_block(frame, self._defender_px, thickness=1,
                                 show_spacing=False)
        # laggard tag
        if self._laggard_px is not None:
            x, y = int(self._laggard_px[0]), int(self._laggard_px[1])
            cv2.circle(frame, (x, y), 26, (0, 0, 255), 2, cv2.LINE_AA)
            tag = f"LAGGARD  {ep.laggard_time_s:.1f}s"
            cv2.putText(frame, tag, (x - 52, y - 32), _FONT, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, tag, (x - 52, y - 32), _FONT, 0.55,
                        (0, 80, 255), 1, cv2.LINE_AA)
        # episode banner
        t = f"TRANSITION  {ep.defending_team.upper()}  " \
            f"open {ep.max_open_m2:.0f} m2"
        cv2.putText(frame, t, (14, frame.shape[0] - 16), _FONT, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (14, frame.shape[0] - 16), _FONT, 0.6,
                    COL_VULN, 1, cv2.LINE_AA)
        return frame
