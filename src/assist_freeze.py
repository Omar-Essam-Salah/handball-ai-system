"""
assist_freeze.py — Assist-triggered freeze-frame & tactical space analysis.

For EVERY finishing attempt (GOAL, SAVE, MISSED_OUT — not only scored goals)
this module rewinds a rolling snapshot buffer to the exact frame where the
ASSIST pass was initiated, then exports a frame-by-frame tactical sequence:

    • defensive block structure  — lines linking the defending players
    • creation / assist zone     — shaded zone around the passer plus the
                                   tapered passing-lane corridor
    • finishing zone             — shaded zone where the receiver catches
                                   and releases the shot
    • pass arrow (IHF dashed) + shot arrow (IHF double)

Outputs land in  reports/assists/<frame_n>_<outcome>/  as numbered PNGs plus
an optional MP4, mirroring GoalReplayRecorder's export style.

Wiring (pipeline loop):

    afz = AssistFreezeAnalyzer("reports/assists", fps=25.0)
    afz.push(frame, tracks_dicts, ball_xy, frame_n)          # every frame

    for ev in new_events:
        afz.on_event(ev.event_type.value, ev.team, ev.frame_n, ev.data,
                     all_events=rules.events)

`tracks_dicts` uses the same dict shape as goal_replay._Snap:
    {tid, cls, x1, y1, x2, y2, cx, cy, team, kpts}
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ihf_notation import (COL_ZONE_CREA, COL_ZONE_FIN, draw_defensive_block,
                          draw_pass_arrow, draw_shot_arrow, draw_zone,
                          passing_lane_polygon, zone_around)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
FINISHING_EVENTS = {"goal", "save", "missed_out"}


@dataclass
class _Snap:
    frame:   np.ndarray
    fn:      int
    ball_xy: Optional[tuple]
    tracks:  list[dict] = field(default_factory=list)


class AssistFreezeAnalyzer:
    """Rolling buffer + assist-window export on every finishing attempt."""

    def __init__(self, out_dir: str | Path = "reports/assists",
                 fps: float = 25.0,
                 buffer_seconds: float = 6.0,
                 downscale: float = 1.0,
                 slice_every: int = 2,      # export every Nth frame of the window
                 write_mp4: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.downscale = downscale
        self.slice_every = max(1, slice_every)
        self.write_mp4 = write_mp4
        self._buf: deque[_Snap] = deque(maxlen=int(buffer_seconds * fps))
        self._cooldown_until = 0.0

    # ── per-frame ingestion ──────────────────────────────────────────────────
    def push(self, frame: np.ndarray, tracks: list[dict],
             ball_xy: Optional[tuple], frame_n: int) -> None:
        if frame is None:
            return
        f = frame
        s = self.downscale
        if s < 0.999:
            f = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)),
                           interpolation=cv2.INTER_AREA)
            tracks = [{**t,
                       "cx": t["cx"] * s, "cy": t["cy"] * s,
                       "x1": t["x1"] * s, "y1": t["y1"] * s,
                       "x2": t["x2"] * s, "y2": t["y2"] * s} for t in tracks]
            ball_xy = None if ball_xy is None else (ball_xy[0] * s, ball_xy[1] * s)
        else:
            f = frame.copy()
            tracks = [dict(t) for t in tracks]
        self._buf.append(_Snap(f, frame_n, ball_xy, tracks))

    # ── event hook ───────────────────────────────────────────────────────────
    def on_event(self, event_type: str, team: str, frame_n: int,
                 data: dict, all_events: list) -> Optional[Path]:
        """Call for every MatchEvent; exports when a finishing event lands."""
        if event_type not in FINISHING_EVENTS:
            return None
        now = time.perf_counter()
        if now < self._cooldown_until:
            return None
        self._cooldown_until = now + 2.0

        shooter_id = data.get("shooter_id") or data.get("scorer_track_id")
        assist_ev = self._find_assist(frame_n, team, shooter_id, all_events)
        return self._export(event_type, team, frame_n, shooter_id, assist_ev)

    # ── assist localisation ──────────────────────────────────────────────────
    @staticmethod
    def _find_assist(finish_fn: int, team: str, shooter_id,
                     all_events: list) -> Optional[dict]:
        """The assist is the last completed PASS by `team` that ended at the
        shooter (when known) within ~4 s before the finish."""
        best = None
        for ev in reversed(all_events[-80:]):
            et = getattr(ev, "event_type", None)
            et = getattr(et, "value", et)
            if et != "pass":
                continue
            if getattr(ev, "team", None) != team:
                continue
            if finish_fn - getattr(ev, "frame_n", -10**9) > 100:   # >4 s → stop
                break
            d = getattr(ev, "data", {}) or {}
            to_id = d.get("to_id") or d.get("receiver_id")
            if shooter_id is not None and to_id is not None and to_id != shooter_id:
                continue
            best = {"frame_n": ev.frame_n,
                    "from_id": d.get("from_id") or d.get("passer_id"),
                    "to_id": to_id}
            break
        return best

    # ── overlay rendering for one snap ───────────────────────────────────────
    def _annotate(self, snap: _Snap, team: str,
                  passer_id, receiver_id,
                  pass_from_xy, pass_to_xy, shot_to_xy,
                  outcome: str) -> np.ndarray:
        img = snap.frame.copy()
        defending = "team_b" if team == "team_a" else "team_a"
        defenders = [(t["cx"], t["cy"]) for t in snap.tracks
                     if t.get("team") == defending]
        attackers = {t["tid"]: (t["cx"], t["cy"]) for t in snap.tracks
                     if t.get("team") == team}

        # 1. defensive block structure + spacing
        if len(defenders) >= 2:
            draw_defensive_block(img, defenders)

        # 2. creation / assist zone + passing lane
        p_from = attackers.get(passer_id, pass_from_xy)
        p_to   = attackers.get(receiver_id, pass_to_xy)
        if p_from is not None:
            draw_zone(img, zone_around([p_from], pad=52),
                      COL_ZONE_CREA, label="CREATION ZONE")
        if p_from is not None and p_to is not None:
            draw_zone(img, passing_lane_polygon(p_from, p_to),
                      COL_ZONE_CREA, alpha=0.18)
            draw_pass_arrow(img, p_from, p_to)

        # 3. finishing zone + shot arrow
        if p_to is not None:
            draw_zone(img, zone_around([p_to], pad=56),
                      COL_ZONE_FIN, label="FINISHING ZONE")
            if shot_to_xy is not None:
                draw_shot_arrow(img, p_to, shot_to_xy)

        # 4. ball marker + header
        if snap.ball_xy is not None:
            cv2.circle(img, (int(snap.ball_xy[0]), int(snap.ball_xy[1])),
                       9, (0, 255, 255), 2, cv2.LINE_AA)
        hdr = f"ASSIST ANALYSIS  |  {outcome.upper()}  |  frame {snap.fn}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (12, 12, 12), -1)
        cv2.putText(img, hdr, (10, 20), _FONT, 0.55, (80, 220, 255), 1, cv2.LINE_AA)
        return img

    # ── export ───────────────────────────────────────────────────────────────
    def _export(self, outcome: str, team: str, finish_fn: int,
                shooter_id, assist_ev: Optional[dict]) -> Optional[Path]:
        if not self._buf:
            return None
        snaps = list(self._buf)

        # window: from just before the assist pass to the finish
        if assist_ev:
            start_fn = assist_ev["frame_n"] - int(0.4 * self.fps)
        else:
            start_fn = finish_fn - int(2.0 * self.fps)   # fallback: last 2 s
        window = [s for s in snaps if start_fn <= s.fn <= finish_fn]
        if len(window) < 2:
            return None

        passer_id   = (assist_ev or {}).get("from_id")
        receiver_id = (assist_ev or {}).get("to_id") or shooter_id

        # pass endpoints: ball position at window start / at first receiver hold
        pass_from_xy = window[0].ball_xy
        pass_to_xy   = None
        for s in window:
            r = next((t for t in s.tracks if t["tid"] == receiver_id), None)
            if r is not None and s.ball_xy is not None:
                if np.hypot(r["cx"] - s.ball_xy[0], r["cy"] - s.ball_xy[1]) < 90:
                    pass_to_xy = (r["cx"], r["cy"])
                    break
        shot_to_xy = window[-1].ball_xy       # ball at the finish = shot target

        out = self.out_dir / f"{finish_fn}_{outcome}"
        out.mkdir(parents=True, exist_ok=True)
        writer = None
        n_exported = 0
        for i, s in enumerate(window):
            if i % self.slice_every and s is not window[-1]:
                continue
            img = self._annotate(s, team, passer_id, receiver_id,
                                 pass_from_xy, pass_to_xy, shot_to_xy, outcome)
            cv2.imwrite(str(out / f"{n_exported:03d}_f{s.fn}.png"), img)
            if self.write_mp4:
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out / "sequence.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(4.0, self.fps / self.slice_every / 2), (w, h))
                writer.write(img)
            n_exported += 1
        if writer is not None:
            writer.release()
        print(f"[AssistFreeze] {outcome} @f{finish_fn}: exported "
              f"{n_exported} tactical frames -> {out}")
        return out
