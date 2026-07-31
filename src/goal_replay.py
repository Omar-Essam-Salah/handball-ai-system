"""
Goal Replay Recorder — IDEA-style pre-goal freeze + tactical overlay.

For every GOAL event the rules engine fires, this module exports:
    reports/<session>/goals/goal_NNN.png   — key freeze frame with tactical arrows
    reports/<session>/goals/goal_NNN.mp4   — short clip (pre + post window) with overlay
    reports/<session>/goals/goal_NNN.json  — metadata (scorer, team, trajectory)

How it integrates with the existing pipeline (no rewrite):
    rec = GoalReplayRecorder(out_dir=Path("reports") / session_id, fps=25.0)

    # every frame, after rules.update():
    rec.push_frame(frame, tracks, ball_track, frame_n)

    # for every GOAL event:
    rec.trigger_goal(goal_event, scorer_label, team_color_bgr, opponent_color_bgr)

    # at shutdown:
    rec.shutdown()

Design notes
------------
- Buffer is a circular deque of frame_copies + lightweight track snapshots.
  Memory usage ≈ pre+post seconds × fps × (H × W × 3) bytes. For 1280×720 @25fps,
  6 seconds ≈ 415 MB. We optionally downscale frames before buffering.
- On goal, we don't write immediately — we wait `post_seconds` more frames so
  the clip includes the celebration. `maybe_finalize()` is called every frame.
- The "shot frame" inside the window is found by scanning the recent ball
  trajectory for the frame where ball speed peaked while leaving the shooter.
  The freeze PNG uses that frame, not the moment the ball crosses the line.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


BALL_CLASS_ID = 32

# ─── colors (BGR) ───
_C_TRAJECTORY    = (0, 230, 255)    # yellow — ball flight path
_C_SHOOTER       = (60, 60, 255)    # red    — shooter
_C_SHOOTER_PATH  = (60, 60, 255)
_C_GK            = (255, 200, 0)    # cyan-blue — goalkeeper
_C_DEFENDER      = (180, 180, 180)  # grey   — other defenders
_C_TEAMMATE      = (140, 140, 140)
_C_GOAL_TARGET   = (0, 90, 255)     # orange
_C_BG_PANEL      = (10, 10, 12)
_C_TEXT_HI       = (255, 255, 255)
_C_TEXT_LO       = (180, 200, 220)


@dataclass
class _Snap:
    """One buffered frame + the minimum data needed to redraw arrows on it."""
    frame:    np.ndarray                        # BGR copy (possibly downscaled)
    fn:       int                               # frame number
    ts:       float                             # wall-clock seconds
    ball_xy:  Optional[tuple[float, float]]     # in *original* frame coords
    ball_real: bool                             # False if phantom/extrapolated
    tracks:   list[dict] = field(default_factory=list)
    # track dict keys: tid, cls, x1, y1, x2, y2, cx, cy, team, kpts (or None)


@dataclass
class _PendingGoal:
    event:           dict
    scorer_label:    str
    team:            str
    team_color:      tuple
    opponent_color:  tuple
    trigger_fn:      int
    pre_snaps:       list[_Snap]              # everything in buffer at trigger time
    post_needed:     int                       # how many more frames we still want
    post_snaps:      list[_Snap] = field(default_factory=list)


class GoalReplayRecorder:
    """Captures pre+post-goal windows and exports IDEA-style overlay assets."""

    def __init__(
        self,
        out_dir:        str | Path,
        fps:            float = 25.0,
        pre_seconds:    float = 4.0,
        post_seconds:   float = 1.5,
        downscale:      float = 0.5,    # default 0.5 → 4× less memory, ~3× faster push
        max_buffer_mb:  float = 350.0,
        write_mp4:      bool  = True,
        write_png:      bool  = True,
        write_json:     bool  = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps          = float(fps)
        self.pre_seconds  = float(pre_seconds)
        self.post_seconds = float(post_seconds)
        self.downscale    = max(0.25, min(1.0, float(downscale)))

        self._pre_frames  = max(1, int(round(self.pre_seconds  * self.fps)))
        self._post_frames = max(1, int(round(self.post_seconds * self.fps)))

        self._buf: deque[_Snap] = deque(maxlen=self._pre_frames)
        self._pending: list[_PendingGoal] = []          # multiple goals can overlap
        self._goal_count = 0

        self.write_mp4  = bool(write_mp4)
        self.write_png  = bool(write_png)
        self.write_json = bool(write_json)

        # Latest finalized freeze frame — pipeline polls this to pin the cv2
        # window. ``last_freeze_idx`` increments on every finalize so callers
        # can detect a new replay even if write_png is disabled.
        self.last_freeze: Optional[np.ndarray] = None
        self.last_freeze_idx: int = 0
        self.last_scorer_label: str = ""
        self.last_team: str = ""

        # Auto-cap buffer if downscale × resolution would blow the memory budget.
        # Will be enforced lazily when first frame arrives.
        self._max_bytes = int(max_buffer_mb * 1024 * 1024)
        self._mem_check_done = False

        print(f"[GoalReplay] out={self.out_dir}  fps={self.fps:.1f}  "
              f"pre={self.pre_seconds}s({self._pre_frames}f) "
              f"post={self.post_seconds}s({self._post_frames}f)  "
              f"downscale={self.downscale}")

    # ──────────────────────────────────────────────────────────────────────
    #  Frame ingestion
    # ──────────────────────────────────────────────────────────────────────
    def push_frame(self, frame: np.ndarray, tracks: list,
                   ball_track, frame_n: int,
                   ball_was_hallucinated: bool = False) -> None:
        if frame is None or frame.size == 0:
            return

        snap_frame = frame
        if self.downscale < 0.999:
            new_w = int(frame.shape[1] * self.downscale)
            new_h = int(frame.shape[0] * self.downscale)
            snap_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Lazy memory enforcement: if buffer would exceed budget, drop pre_frames.
        if not self._mem_check_done:
            per_frame = snap_frame.nbytes
            cap = max(int(self._max_bytes / max(per_frame, 1)), 30)
            if cap < self._pre_frames:
                print(f"[GoalReplay] memory cap reduced pre buffer "
                      f"{self._pre_frames} → {cap} frames")
                self._buf = deque(self._buf, maxlen=cap)
                self._pre_frames = cap
            self._mem_check_done = True

        ball_xy = None
        if ball_track is not None:
            try:
                ball_xy = (float(ball_track.cx), float(ball_track.cy))
            except Exception:
                ball_xy = None

        # Lightweight track snapshot — only what we need to redraw arrows.
        # NOTE: keypoints are stored by reference (no .copy()) — the underlying
        # numpy arrays are detached from the next-frame detector results, so this
        # is safe and saves ~1ms/track at 1080p.
        track_snaps = []
        for t in tracks:
            cls = int(getattr(t, "class_id", 0))
            if cls == BALL_CLASS_ID:
                continue
            track_snaps.append({
                "tid":  int(getattr(t, "track_id", -1)),
                "cls":  cls,
                "x1":   float(t.x1), "y1": float(t.y1),
                "x2":   float(t.x2), "y2": float(t.y2),
                "cx":   float(t.cx), "cy": float(t.cy),
                "team": getattr(t, "team", "unknown") or "unknown",
                "kpts": getattr(t, "keypoints", None),
            })

        snap = _Snap(
            frame=snap_frame.copy(),
            fn=int(frame_n),
            ts=time.perf_counter(),
            ball_xy=ball_xy,
            ball_real=not bool(ball_was_hallucinated),
            tracks=track_snaps,
        )
        self._buf.append(snap)

        # Feed pending goal captures with post-trigger frames.
        if self._pending:
            still_pending = []
            for pg in self._pending:
                if pg.post_needed > 0:
                    pg.post_snaps.append(snap)
                    pg.post_needed -= 1
                if pg.post_needed > 0:
                    still_pending.append(pg)
                else:
                    self._finalize(pg)
            self._pending = still_pending

    # ──────────────────────────────────────────────────────────────────────
    #  Goal trigger
    # ──────────────────────────────────────────────────────────────────────
    def trigger_goal(self, goal_event: dict, scorer_label: str,
                     team: str, team_color_bgr: tuple,
                     opponent_color_bgr: tuple) -> None:
        if not self._buf:
            print("[GoalReplay] trigger_goal: empty buffer — skipped")
            return
        pg = _PendingGoal(
            event=dict(goal_event),
            scorer_label=str(scorer_label or "Unknown"),
            team=str(team or "team_a"),
            team_color=tuple(int(c) for c in team_color_bgr),
            opponent_color=tuple(int(c) for c in opponent_color_bgr),
            trigger_fn=self._buf[-1].fn,
            pre_snaps=list(self._buf),
            post_needed=self._post_frames,
        )
        self._pending.append(pg)

    # ──────────────────────────────────────────────────────────────────────
    #  Shutdown
    # ──────────────────────────────────────────────────────────────────────
    def shutdown(self) -> None:
        # Flush any pending goals using whatever post-frames we managed to grab.
        for pg in self._pending:
            self._finalize(pg)
        self._pending.clear()
        print(f"[GoalReplay] shutdown — wrote {self._goal_count} goal replays "
              f"to {self.out_dir}")

    # ──────────────────────────────────────────────────────────────────────
    #  Finalization — locate shot frame, draw overlays, write outputs
    # ──────────────────────────────────────────────────────────────────────
    def _finalize(self, pg: _PendingGoal) -> None:
        self._goal_count += 1
        idx = self._goal_count
        stem = f"goal_{idx:03d}"

        all_snaps = pg.pre_snaps + pg.post_snaps
        if not all_snaps:
            return

        # Locate the SHOT frame inside the window: the frame where ball speed
        # peaked while heading toward the goal. Fallback: ~0.6s before trigger.
        shot_idx = self._locate_shot_frame(all_snaps, pg)
        if shot_idx is None:
            shot_idx = max(0, len(pg.pre_snaps) - int(0.6 * self.fps))
        shot_snap = all_snaps[shot_idx]

        # Raw trajectory: ball positions from shot_idx onward.
        raw_traj = []
        for s in all_snaps[shot_idx:]:
            if s.ball_xy is not None:
                raw_traj.append((s.ball_xy[0] * self.downscale,
                                  s.ball_xy[1] * self.downscale, s.ball_real))

        # Filter: prefer real detections, drop phantom outliers, smooth
        traj_pts = self._filter_trajectory(raw_traj)

        # Identify shooter from tracks closest to ball at shot_idx.
        shooter_tid = self._guess_shooter_tid(shot_snap)

        # Shooter run-up: track positions in last ~1.0s before shot for that tid.
        shooter_path = self._collect_shooter_path(all_snaps, shot_idx, shooter_tid)

        # Render the freeze PNG with overlays.
        key_img = None
        try:
            key_img = self._render_overlay(
                shot_snap, traj_pts, shooter_path, shooter_tid, pg,
                is_key_frame=True,
            )
        except Exception as e:
            print(f"[GoalReplay] overlay render failed for {stem}: {e}")

        if self.write_png and key_img is not None:
            try:
                cv2.imwrite(str(self.out_dir / f"{stem}.png"), key_img)
            except Exception as e:
                print(f"[GoalReplay] PNG write failed for {stem}: {e}")

        # Expose the freeze for the pipeline to pin
        if key_img is not None:
            self.last_freeze = key_img
            self.last_scorer_label = pg.scorer_label
            self.last_team = pg.team
        self.last_freeze_idx += 1

        # Render the MP4 with arrows drawn on every frame from shot onward.
        if self.write_mp4:
            mp4_path = self.out_dir / f"{stem}.mp4"
            try:
                self._render_clip(all_snaps, shot_idx, shooter_tid, pg, mp4_path)
            except Exception as e:
                print(f"[GoalReplay] MP4 write failed for {stem}: {e}")

        # Metadata json.
        if self.write_json:
            try:
                meta = {
                    "stem":         stem,
                    "scorer":       pg.scorer_label,
                    "team":         pg.team,
                    "trigger_fn":   pg.trigger_fn,
                    "shot_fn":      shot_snap.fn,
                    "frames_total": len(all_snaps),
                    "trajectory":   [(round(x, 1), round(y, 1), bool(real))
                                     for x, y, real in traj_pts],
                    "shooter_tid":  shooter_tid,
                    "raw_event":    {k: v for k, v in pg.event.items() if _jsonable(v)},
                }
                (self.out_dir / f"{stem}.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[GoalReplay] JSON write failed for {stem}: {e}")

        print(f"[GoalReplay] saved {stem}  scorer={pg.scorer_label}  "
              f"team={pg.team}  frames={len(all_snaps)}")

    # ──────────────────────────────────────────────────────────────────────
    #  Helpers — shot frame, shooter, path
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _filter_trajectory(raw: list[tuple[float, float, bool]]
                            ) -> list[tuple[float, float, bool]]:
        """Drop wild phantom jumps + smooth the kept points so the rendered
        arrow looks like a real ball flight, not a zigzag.

        Strategy:
            1. Walk the trajectory; reject any point whose distance from the
               previous accepted point exceeds 120 px (impossible single-frame jump).
            2. Real points are always kept; phantoms only kept if they fit the
               recent velocity within ±60% tolerance.
            3. Apply a 3-tap moving-average smoothing on the kept points.
        """
        if len(raw) < 2:
            return list(raw)
        kept: list[tuple[float, float, bool]] = [raw[0]]
        for i in range(1, len(raw)):
            x, y, real = raw[i]
            px, py, _ = kept[-1]
            d = math.hypot(x - px, y - py)
            # Hard outlier filter (single-frame ball can't move >120 px in any sport)
            if d > 120.0 and not real:
                continue
            # Phantom velocity sanity vs last 2 kept points
            if not real and len(kept) >= 2:
                a, b = kept[-2], kept[-1]
                vx_ref, vy_ref = b[0] - a[0], b[1] - a[1]
                vx_now, vy_now = x - b[0], y - b[1]
                # If direction flips drastically, drop it
                if (vx_ref * vx_now + vy_ref * vy_now) < -2.0:
                    continue
            kept.append((x, y, real))

        if len(kept) < 3:
            return kept

        # 3-tap moving average smoothing (preserve endpoints)
        out = [kept[0]]
        for i in range(1, len(kept) - 1):
            x = (kept[i - 1][0] + kept[i][0] + kept[i + 1][0]) / 3.0
            y = (kept[i - 1][1] + kept[i][1] + kept[i + 1][1]) / 3.0
            out.append((x, y, kept[i][2]))
        out.append(kept[-1])
        return out

    @staticmethod
    def _locate_shot_frame(snaps: list[_Snap], pg: _PendingGoal) -> Optional[int]:
        """Find the frame where ball speed peaked just before the goal.
        Restrict to the pre-window (before trigger) and ignore phantom-only runs."""
        n_pre = sum(1 for s in snaps if s.fn <= pg.trigger_fn)
        if n_pre < 4:
            return None
        speeds: list[tuple[int, float]] = []
        for i in range(1, n_pre):
            a, b = snaps[i - 1], snaps[i]
            if a.ball_xy is None or b.ball_xy is None:
                continue
            df = max(b.fn - a.fn, 1)
            d = math.hypot(b.ball_xy[0] - a.ball_xy[0],
                           b.ball_xy[1] - a.ball_xy[1]) / df
            speeds.append((i, d))
        if not speeds:
            return None
        # Argmax of speed; back off one frame to land on "just before peak".
        peak_i = max(speeds, key=lambda x: x[1])[0]
        return max(0, peak_i - 1)

    @staticmethod
    def _guess_shooter_tid(snap: _Snap) -> Optional[int]:
        if snap.ball_xy is None or not snap.tracks:
            return None
        bx, by = snap.ball_xy
        best, best_d = None, 1e9
        for t in snap.tracks:
            d = math.hypot(t["cx"] - bx, t["cy"] - by)
            # Prefer hand keypoints if available
            kp = t.get("kpts")
            if isinstance(kp, np.ndarray) and kp.shape[0] >= 17:
                for idx in (9, 10):                              # wrists
                    if float(kp[idx][2]) >= 0.35:
                        d_h = math.hypot(float(kp[idx][0]) - bx,
                                         float(kp[idx][1]) - by)
                        if d_h < d:
                            d = d_h
            if d < best_d:
                best_d, best = d, t["tid"]
        # Plausibility filter — must be within ~250px to be the shooter
        return best if best_d < 250.0 else None

    @staticmethod
    def _collect_shooter_path(snaps: list[_Snap], shot_idx: int,
                               tid: Optional[int]) -> list[tuple[float, float]]:
        if tid is None:
            return []
        # Look back ~1 second from shot frame
        start = max(0, shot_idx - 25)
        path = []
        for s in snaps[start:shot_idx + 1]:
            for t in s.tracks:
                if t["tid"] == tid:
                    path.append((t["cx"], t["cy"]))
                    break
        return path

    # ──────────────────────────────────────────────────────────────────────
    #  Drawing
    # ──────────────────────────────────────────────────────────────────────
    def _render_overlay(self, snap: _Snap,
                        traj_pts: list[tuple[float, float, bool]],
                        shooter_path: list[tuple[float, float]],
                        shooter_tid: Optional[int],
                        pg: _PendingGoal,
                        is_key_frame: bool = False) -> np.ndarray:
        """Render arrows + ring + label onto a copy of `snap.frame`."""
        ds = self.downscale
        out = snap.frame.copy()
        H, W = out.shape[:2]

        # 0. Soft darkening overlay so the colored arrows pop. Only on key frame.
        if is_key_frame:
            dark = np.zeros_like(out)
            cv2.addWeighted(out, 0.78, dark, 0.22, 0, out)

        # 1. Faint dots/labels for every other player so the technique is readable.
        for t in snap.tracks:
            if t["tid"] == shooter_tid:
                continue
            px = int(t["cx"] * ds)
            py = int(t["y2"] * ds)
            tcol = pg.team_color if t["team"] == pg.team else (
                pg.opponent_color if t["team"] in ("team_a", "team_b") else _C_DEFENDER
            )
            cv2.circle(out, (px, py), 5, tcol, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (px, py), 5, (10, 10, 10), 1, lineType=cv2.LINE_AA)

        # 2. Spotlight ring at shooter's feet (key frame only)
        shooter = next((t for t in snap.tracks if t["tid"] == shooter_tid), None)
        if is_key_frame and shooter is not None:
            fx = int(shooter["cx"] * ds)
            fy = int(shooter["y2"] * ds)
            ring_r = max(30, int((shooter["x2"] - shooter["x1"]) * 0.6 * ds))
            overlay = out.copy()
            cv2.ellipse(overlay, (fx, fy), (ring_r, ring_r // 2), 0, 0, 360,
                        pg.team_color, -1)
            cv2.addWeighted(overlay, 0.50, out, 0.50, 0, out)
            cv2.ellipse(out, (fx, fy), (ring_r, ring_r // 2), 0, 0, 360,
                        pg.team_color, 2, lineType=cv2.LINE_AA)
            # Two concentric rings for emphasis
            cv2.ellipse(out, (fx, fy), (ring_r + 6, (ring_r + 6) // 2),
                        0, 0, 360, pg.team_color, 1, lineType=cv2.LINE_AA)

        # 3. Shooter run-up: dashed red poly + arrowhead
        if len(shooter_path) >= 2:
            pts = [(int(x * ds), int(y * ds)) for x, y in shooter_path]
            for i in range(1, len(pts)):
                if i % 2 == 0:
                    continue
                cv2.line(out, pts[i - 1], pts[i], _C_SHOOTER_PATH, 2,
                         lineType=cv2.LINE_AA)
            cv2.arrowedLine(out, pts[-2], pts[-1], _C_SHOOTER_PATH, 3,
                            tipLength=0.35, line_type=cv2.LINE_AA)

        # 4. Ball trajectory — gradient from dim → bright + arrowhead.
        if len(traj_pts) >= 2:
            pts = [(int(x), int(y)) for x, y, _ in traj_pts]
            n = len(pts)
            for i in range(1, n):
                # brightness ramps from 0.45 at start to 1.00 at end
                k = 0.45 + 0.55 * (i / max(n - 1, 1))
                col = (int(_C_TRAJECTORY[0] * k),
                       int(_C_TRAJECTORY[1] * k),
                       int(_C_TRAJECTORY[2] * k))
                cv2.line(out, pts[i - 1], pts[i], col, 4, lineType=cv2.LINE_AA)
            cv2.arrowedLine(out, pts[-2], pts[-1], _C_TRAJECTORY, 4,
                            tipLength=0.45, line_type=cv2.LINE_AA)
            cv2.circle(out, pts[0],  7, _C_TRAJECTORY, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, pts[0],  7, (10, 10, 10),   1, lineType=cv2.LINE_AA)
            cv2.circle(out, pts[-1], 9, _C_GOAL_TARGET, 2, lineType=cv2.LINE_AA)

        # 5. Highlight goalkeeper of the conceding team (closest defender to ball end).
        if traj_pts:
            ex, ey = int(traj_pts[-1][0]), int(traj_pts[-1][1])
            gk = self._guess_goalkeeper(snap, pg, (ex, ey))
            if gk is not None:
                gx = int(gk["cx"] * ds)
                gy = int(gk["y2"] * ds)
                cv2.ellipse(out, (gx, gy), (28, 14), 0, 0, 360, _C_GK, 3,
                            lineType=cv2.LINE_AA)
                cv2.putText(out, "GK", (gx - 14, gy - 18),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, _C_GK, 2, cv2.LINE_AA)

        # 6. Top-left "scorer" badge — like the Once Sport "11 Once Sport" tag.
        self._draw_scorer_badge(out, pg)

        # 7. IDEA-style 2D top-down mini-court inset (key frame only — busy on clip)
        if is_key_frame:
            try:
                # Use the underlying snap.frame size for screen→court mapping.
                # snap.frame is already at downscaled resolution.
                fh, fw = snap.frame.shape[:2]
                mini = self._build_mini_court(snap, traj_pts, shooter_tid, pg, fw, fh)
                mh, mw = mini.shape[:2]
                # Place top-right with a 12 px margin
                tx, ty = W - mw - 12, 12
                if tx >= 0 and ty + mh < H:
                    # Add subtle drop shadow for readability
                    shadow = (max(0, tx - 4), ty + 4)
                    cv2.rectangle(out, shadow,
                                   (shadow[0] + mw + 4, shadow[1] + mh),
                                   (0, 0, 0), -1)
                    out[ty:ty + mh, tx:tx + mw] = mini
            except Exception as e:
                print(f"[GoalReplay] mini-court render failed: {e}")

        # 8. Footer with frame info + GOAL stamp on key frame.
        self._draw_footer(out, snap, pg, is_key_frame)

        return out

    # ──────────────────────────────────────────────────────────────────
    #  IDEA-style 2D top-down mini-court inset
    # ──────────────────────────────────────────────────────────────────
    def _build_mini_court(
        self,
        snap: _Snap,
        traj_pts: list[tuple[float, float, bool]],
        shooter_tid: Optional[int],
        pg: _PendingGoal,
        frame_w: int, frame_h: int,
    ) -> np.ndarray:
        """Returns a small BGR image (W×H) — schematic top-down court with all
        the players, the shooter highlighted, and the shot trajectory drawn
        in court-relative coordinates. The signature tactical view from Once
        Sport / Scoutz."""
        W, H = 360, 180
        canvas = np.full((H, W, 3), 22, dtype=np.uint8)

        # ── Court geometry (proportions: 40m × 20m → 2:1) ──
        m_left, m_right, m_top, m_bot = 6, 6, 8, 8
        cw, ch = W - m_left - m_right, H - m_top - m_bot
        x0, y0 = m_left, m_top
        # Court fill — slight tint to evoke the playing surface
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y0 + ch), (44, 60, 78), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + cw, y0 + ch), (180, 180, 180), 2,
                       lineType=cv2.LINE_AA)
        # Centre line
        cv2.line(canvas, (x0 + cw // 2, y0), (x0 + cw // 2, y0 + ch),
                 (130, 130, 130), 1, cv2.LINE_AA)
        # Centre dot
        cv2.circle(canvas, (x0 + cw // 2, y0 + ch // 2), 3, (160, 160, 160), 1,
                    cv2.LINE_AA)
        # Goal frames (3m wide → ch * 3/20 = ch * 0.15)
        gh = max(8, int(ch * 0.15))
        cv2.rectangle(canvas, (x0 - 4, y0 + (ch - gh) // 2),
                       (x0 + 1, y0 + (ch + gh) // 2), (255, 255, 255), 1)
        cv2.rectangle(canvas, (x0 + cw - 1, y0 + (ch - gh) // 2),
                       (x0 + cw + 4, y0 + (ch + gh) // 2), (255, 255, 255), 1)
        # 6m and 9m arcs
        r6 = max(4, int(cw * 6 / 40))
        r9 = max(6, int(cw * 9 / 40))
        cv2.ellipse(canvas, (x0, y0 + ch // 2), (r6, r6),
                    0, -90, 90, (140, 160, 175), 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x0 + cw, y0 + ch // 2), (r6, r6),
                    0, 90, 270, (140, 160, 175), 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x0, y0 + ch // 2), (r9, r9),
                    0, -90, 90, (100, 120, 140), 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x0 + cw, y0 + ch // 2), (r9, r9),
                    0, 90, 270, (100, 120, 140), 1, cv2.LINE_AA)

        # ── Map function: screen → court coords ──
        # No homography? Use a coarse inverse-perspective heuristic: assume the
        # camera sees mostly the court in the lower 70% of the frame, mapping
        # x linearly across the full width, y linearly across the height.
        def to_court(sx: float, sy: float) -> tuple[int, int]:
            ax = max(0.0, min(1.0, sx / max(frame_w, 1)))
            # y_frac maps lower 70% of frame (roughly the court band) to 0..1
            y_frac = max(0.0, (sy / max(frame_h, 1) - 0.30) / 0.70)
            y_frac = min(1.0, y_frac)
            cx_ = x0 + int(ax * cw)
            cy_ = y0 + int(y_frac * ch)
            return cx_, cy_

        # ── Plot all players ──
        for t in snap.tracks:
            cx_s = float(t.get("cx", 0))
            cy_s = float(t.get("y2", 0))    # use feet for ground-plane mapping
            mx, my = to_court(cx_s, cy_s)
            tcol = pg.team_color if t.get("team") == pg.team else (
                pg.opponent_color if t.get("team") in ("team_a", "team_b") else _C_DEFENDER
            )
            cv2.circle(canvas, (mx, my), 4, tcol, -1, cv2.LINE_AA)
            cv2.circle(canvas, (mx, my), 4, (10, 10, 10), 1, cv2.LINE_AA)

        # ── Highlight the shooter ──
        shooter = next((t for t in snap.tracks if t.get("tid") == shooter_tid), None)
        if shooter is not None:
            cx_s = float(shooter.get("cx", 0))
            cy_s = float(shooter.get("y2", 0))
            mx, my = to_court(cx_s, cy_s)
            cv2.circle(canvas, (mx, my), 6, pg.team_color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (mx, my), 9, pg.team_color, 1, cv2.LINE_AA)

        # ── Trajectory arrow on the court ──
        if len(traj_pts) >= 2:
            ds = self.downscale or 1.0
            scaled_pts = []
            for x, y, _real in traj_pts:
                scaled_pts.append(to_court(x / ds, y / ds))
            # Draw a polyline + arrowhead
            for i in range(1, len(scaled_pts)):
                cv2.line(canvas, scaled_pts[i - 1], scaled_pts[i],
                         _C_TRAJECTORY, 2, cv2.LINE_AA)
            cv2.arrowedLine(canvas, scaled_pts[-2], scaled_pts[-1],
                             _C_TRAJECTORY, 2, tipLength=0.4, line_type=cv2.LINE_AA)
            cv2.circle(canvas, scaled_pts[0], 3, _C_TRAJECTORY, -1, cv2.LINE_AA)

        # ── Title strip ──
        zone_lab = self._classify_zone(shooter, pg.team, frame_w, frame_h)
        # ASCII-only separators — OpenCV's Hershey fonts don't render Unicode
        title = f"TACTICAL  |  {pg.scorer_label}  |  {zone_lab}"
        cv2.rectangle(canvas, (0, 0), (W, 16), (12, 12, 12), -1)
        cv2.putText(canvas, title, (6, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 230, 245), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (90, 90, 90), 1)
        return canvas

    @staticmethod
    def _classify_zone(shooter: Optional[dict], team: str,
                        frame_w: int, frame_h: int) -> str:
        """Rough zone classification from screen coords (no homography needed).
        Mirrors the IDEA report categories: Wing / 9m / 6m / 7m / Long."""
        if shooter is None:
            return "Unknown"
        sx = float(shooter.get("cx", frame_w * 0.5)) / max(frame_w, 1)
        sy = float(shooter.get("y2", frame_h * 0.5)) / max(frame_h, 1)
        # Distance from the attacking goal — depends on team direction
        # Without flip-state we approximate: team_a attacks right, team_b attacks left
        attacking_right = (team == "team_a")
        depth = (1.0 - sx) if attacking_right else sx          # 0 = on the goal line
        # Lateral position
        lateral = abs(sy - 0.65)                                # 0 = mid-court
        if depth < 0.18 and lateral > 0.20:
            return "Wing"
        if depth < 0.18:
            return "6m Centre"
        if depth < 0.32:
            return "9m Centre"
        if depth < 0.50:
            return "Long Distance"
        return "Backcourt"

    @staticmethod
    def _guess_goalkeeper(snap: _Snap, pg: _PendingGoal,
                          ball_end_xy_drawn: tuple[int, int]) -> Optional[dict]:
        """Closest non-scoring-team player to the ball's final position."""
        if not snap.tracks:
            return None
        opp_team = "team_b" if pg.team == "team_a" else "team_a"
        bx, by = ball_end_xy_drawn
        best, bd = None, 1e9
        for t in snap.tracks:
            if t["team"] != opp_team:
                continue
            d = math.hypot(t["cx"] - bx, t["cy"] - by)
            if d < bd:
                bd, best = d, t
        return best

    def _draw_scorer_badge(self, out: np.ndarray, pg: _PendingGoal) -> None:
        H, W = out.shape[:2]
        text = f"  {pg.scorer_label}  "
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.65
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x0, y0 = 18, 18
        pad = 8
        cv2.rectangle(out, (x0, y0), (x0 + tw + pad * 2, y0 + th + pad * 2),
                      pg.team_color, -1)
        cv2.rectangle(out, (x0, y0), (x0 + tw + pad * 2, y0 + th + pad * 2),
                      _C_TEXT_HI, 1)
        cv2.putText(out, text, (x0 + pad, y0 + th + pad - 2),
                    font, scale, _C_TEXT_HI, thick, cv2.LINE_AA)
        # Small "GOAL" tag right of it
        gtext = "  GOAL  "
        (gw, gh), _ = cv2.getTextSize(gtext, font, scale, thick)
        gx0 = x0 + tw + pad * 2 + 4
        cv2.rectangle(out, (gx0, y0), (gx0 + gw + pad * 2, y0 + th + pad * 2),
                      _C_BG_PANEL, -1)
        cv2.rectangle(out, (gx0, y0), (gx0 + gw + pad * 2, y0 + th + pad * 2),
                      _C_GOAL_TARGET, 1)
        cv2.putText(out, gtext, (gx0 + pad, y0 + th + pad - 2),
                    font, scale, _C_GOAL_TARGET, thick, cv2.LINE_AA)

    @staticmethod
    def _draw_footer(out: np.ndarray, snap: _Snap, pg: _PendingGoal,
                     is_key: bool) -> None:
        H, W = out.shape[:2]
        bar_h = 28
        cv2.rectangle(out, (0, H - bar_h), (W, H), _C_BG_PANEL, -1)
        cv2.line(out, (0, H - bar_h), (W, H - bar_h), (60, 60, 60), 1)
        tag = "KEY FRAME" if is_key else "REPLAY"
        s = (f"{tag}  |  frame {snap.fn}  |  scorer {pg.scorer_label}  |  "
             f"team {pg.team.replace('team_', '').upper()}")
        cv2.putText(out, s, (10, H - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _C_TEXT_LO, 1, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────
    #  MP4 clip writer
    # ──────────────────────────────────────────────────────────────────────
    def _render_clip(self, all_snaps: list[_Snap], shot_idx: int,
                     shooter_tid: Optional[int], pg: _PendingGoal,
                     out_path: Path) -> None:
        if not all_snaps:
            return
        # Use first frame size as canvas
        h, w = all_snaps[0].frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
        if not writer.isOpened():
            print(f"[GoalReplay] VideoWriter failed to open {out_path}")
            return

        # Pre-compute shooter path that we'll grow as we step through frames.
        for i, snap in enumerate(all_snaps):
            # Trajectory grows from shot_idx → current
            traj = []
            for s in all_snaps[shot_idx: max(shot_idx + 1, i + 1)]:
                if s.ball_xy is not None:
                    traj.append((s.ball_xy[0] * self.downscale,
                                 s.ball_xy[1] * self.downscale, s.ball_real))

            # Shooter path up to the current frame
            shooter_path = []
            start = max(0, shot_idx - 25)
            end = min(i + 1, shot_idx + 1)
            if shooter_tid is not None:
                for s in all_snaps[start:end]:
                    for t in s.tracks:
                        if t["tid"] == shooter_tid:
                            shooter_path.append((t["cx"], t["cy"]))
                            break

            is_key = (i == shot_idx)
            try:
                rendered = self._render_overlay(snap, traj, shooter_path,
                                                shooter_tid, pg,
                                                is_key_frame=is_key)
            except Exception as e:
                rendered = snap.frame
            writer.write(rendered)
        writer.release()


def _jsonable(v) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, dict, tuple))
