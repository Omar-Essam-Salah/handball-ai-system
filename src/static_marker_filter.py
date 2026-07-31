"""
Static-Marker Filter — suppresses fixed painted spots that YOLO mistakes for a ball.

Why this exists
---------------
The YOLO COCO sports-ball detector (and even handball-trained variants) regularly
fires on:
    - the 7-metre penalty mark in front of the goal
    - the centre-point of the court
    - logos / round emblems painted on the floor or hanging behind the goal
    - the referee's hand-held marker

Because these objects never move, they look "ball-shaped" in every frame, so the
existing physics filter in handball_rules.py (which rejects a ball that stays
still for ~2 seconds) catches them only AFTER the damage — i.e. after they've
been confused with the real ball for 50 frames. This module fixes that by
remembering exact pixel cells where ball candidates persist statically and
adding them to a blacklist that lasts for several minutes.

Algorithm
---------
1. Quantize the frame into ``cell_size``-pixel grid cells.
2. Each frame, for every ball candidate, record the cell hit in a rolling
   ``hit_window``-frame deque.
3. A cell whose hit-rate exceeds ``hit_threshold`` is blacklisted for
   ``exclusion_ttl`` frames.
4. When candidates arrive, drop any whose centroid lies in a blacklisted cell
   or its 8-neighbour ring (catches the soft edges of a painted marker).

The class is stateless w.r.t. frame contents — it only needs candidate boxes.
Memory is O(unique cells seen × hit_window).

Wiring
------
    static_filter = StaticMarkerFilter(frame_w, frame_h)

    all_dets    = detector.detect(frame)
    ball_cands  = [d for d in all_dets if d.class_id == BALL_CLASS_ID]
    static_filter.observe(ball_cands, frame_n)
    ball_clean  = static_filter.filter(ball_cands)
    clean_dets  = [d for d in all_dets if d.class_id != BALL_CLASS_ID] + ball_clean
"""

from __future__ import annotations
from collections import deque
from typing import Iterable

BALL_CLASS_ID = 32


class StaticMarkerFilter:
    def __init__(
        self,
        frame_w:        int,
        frame_h:        int,
        cell_size:      int   = 10,    # smaller cells = tighter localisation
        hit_window:     int   = 75,    # ~3s @25fps — faster reaction time
        hit_threshold:  float = 0.70,  # ≥70% of recent frames had a hit → marker
        exclusion_ttl:  int   = 1500,  # blacklist sticks for 60s — court markings don't move
        warmup_frames:  int   = 40,    # ~1.6s of history before any blacklist
        neighbor_ring:  int   = 1,
        player_safe_radius_px: float = 55.0,   # tighter — only cells UNDER a player are protected
        player_safe_ratio:     float = 0.45,   # ≥45% of hits with player nearby = held ball
        force_blacklist_ratio: float = 0.90,   # frozen in ONE cell ≥90% of the window → marker
    ):
        self.fw, self.fh   = int(frame_w), int(frame_h)
        self.cs            = max(4, int(cell_size))
        self.hit_window    = max(10, int(hit_window))
        self.hit_threshold = float(hit_threshold)
        self.exclusion_ttl = max(60, int(exclusion_ttl))
        self.warmup_frames = max(1, int(warmup_frames))
        self.neighbor_ring = max(0, int(neighbor_ring))
        self.player_safe_r2 = float(player_safe_radius_px) ** 2
        self.player_safe_ratio = float(player_safe_ratio)
        self.force_blacklist_ratio = float(force_blacklist_ratio)

        # cell → deque[(obs_n, had_player_nearby:bool)]
        self._cells:    dict[tuple[int, int], deque] = {}
        # cell → frame_n at which it became (or was last refreshed as) blacklisted
        self._blacklist: dict[tuple[int, int], int] = {}
        self._frame_n   = 0
        self._hits_seen = 0
        # Observation counter — incremented once per observe() call. The hit
        # window is measured in OBSERVATIONS, not raw frame numbers, so the
        # filter behaves identically regardless of --infer-every (it used to
        # silently stop working at infer_every>1 because only every Nth frame
        # produced an observation but the window was counted in frames).
        self._obs_n     = 0

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────
    def observe(self, ball_candidates: Iterable, frame_n: int,
                player_positions: Iterable | None = None) -> None:
        """Record ball-candidate positions and update the blacklist.
        If ``player_positions`` is provided (iterable of (cx, cy)), cells where a
        player is nearby get tagged as "live" — even if a candidate keeps firing
        there, we won't blacklist (it's a player holding the ball)."""
        self._frame_n = int(frame_n)
        self._obs_n  += 1

        # Snapshot player centroids once for cheap lookup
        ppos: list[tuple[float, float]] = []
        if player_positions is not None:
            for p in player_positions:
                try:
                    ppos.append((float(p[0]), float(p[1])))
                except Exception:
                    continue

        # 1. record one hit per cell per frame, tagged with player-near flag
        cells_this_frame: dict[tuple[int, int], bool] = {}
        for d in ball_candidates:
            try:
                cx = (float(d.x1) + float(d.x2)) * 0.5
                cy = (float(d.y1) + float(d.y2)) * 0.5
            except (AttributeError, TypeError):
                continue
            c = self._qcell(cx, cy)
            near = self._has_player_near(cx, cy, ppos)
            # if multiple candidates fall in the same cell, ANY player-near tag wins
            cells_this_frame[c] = cells_this_frame.get(c, False) or near

        for c, near in cells_this_frame.items():
            self._hits_seen += 1
            dq = self._cells.setdefault(c, deque(maxlen=self.hit_window))
            dq.append((self._obs_n, bool(near)))

        # 2. prune ancient hits + decide who is now a marker
        if self._hits_seen >= self.warmup_frames:
            self._refresh_blacklist()

        # 3. drop expired blacklist entries
        if self._blacklist:
            stale = [c for c, fn in self._blacklist.items()
                     if self._frame_n - fn > self.exclusion_ttl]
            for c in stale:
                del self._blacklist[c]

    def _has_player_near(self, cx: float, cy: float,
                          ppos: list[tuple[float, float]]) -> bool:
        if not ppos:
            return False
        r2 = self.player_safe_r2
        for px, py in ppos:
            dx = px - cx
            dy = py - cy
            if dx * dx + dy * dy < r2:
                return True
        return False

    def filter(self, ball_candidates: Iterable) -> list:
        """Return only candidates outside the blacklisted cells."""
        if not self._blacklist:
            return list(ball_candidates)
        kept = []
        for d in ball_candidates:
            try:
                cx = (float(d.x1) + float(d.x2)) * 0.5
                cy = (float(d.y1) + float(d.y2)) * 0.5
            except (AttributeError, TypeError):
                kept.append(d)
                continue
            if not self._is_blocked(cx, cy):
                kept.append(d)
        return kept

    def is_blocked(self, x: float, y: float) -> bool:
        """Convenience: would a candidate at (x,y) be rejected?"""
        return self._is_blocked(x, y)

    def blacklist_boxes(self) -> list[tuple[int, int, int, int]]:
        """Pixel-space rects (x1,y1,x2,y2) of all current blacklist cells.
        Useful for debug overlays — draw red squares where markers got muted."""
        out = []
        for (cx, cy) in self._blacklist:
            x1 = cx * self.cs
            y1 = cy * self.cs
            x2 = x1 + self.cs
            y2 = y1 + self.cs
            out.append((x1, y1, x2, y2))
        return out

    def stats(self) -> dict:
        return {
            "frame_n":       self._frame_n,
            "tracked_cells": len(self._cells),
            "blacklist":     len(self._blacklist),
            "hits_seen":     self._hits_seen,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────────────────
    def _qcell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cs), int(y // self.cs))

    def _refresh_blacklist(self) -> None:
        """Move cells that exceeded the hit-rate threshold into the blacklist.
        Skip any cell where ≥ ``player_safe_ratio`` of recent hits had a player
        nearby — those are real balls being held, not painted markers.

        EXCEPTION: a candidate that fires in the SAME cell in ≥
        ``force_blacklist_ratio`` of the window is "frozen" — a painted sign /
        marker that never moves. A genuinely held ball drifts across cells (the
        player breathes, pivots, the ball rolls in the hands), so it can never
        stay frozen this long. Frozen cells are blacklisted even with a player
        standing next to them — this is what finally kills the white sign in
        the goal mouth, where a defender/keeper is always within range."""
        threshold = max(3, int(self.hit_threshold * self.hit_window))
        frozen_count = max(threshold + 1, int(self.force_blacklist_ratio * self.hit_window))
        cutoff = self._obs_n - self.hit_window
        empty_cells = []
        for c, dq in self._cells.items():
            # Drop hits older than the rolling window
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            if not dq:
                empty_cells.append(c)
                continue

            if len(dq) < threshold:
                continue

            frozen = len(dq) >= frozen_count

            # If ≥X% of hits had a player nearby it's normally a held ball — but
            # a frozen cell overrides that (a held ball is never frozen).
            n_with_player = sum(1 for _, near in dq if near)
            if (not frozen) and (n_with_player / len(dq) >= self.player_safe_ratio):
                # actively un-blacklist if previously marked
                self._blacklist.pop(c, None)
                continue

            # All checks passed — this is a real static marker
            self._blacklist[c] = self._frame_n
        for c in empty_cells:
            del self._cells[c]

    def _is_blocked(self, x: float, y: float) -> bool:
        cx, cy = self._qcell(x, y)
        if (cx, cy) in self._blacklist:
            return True
        if self.neighbor_ring <= 0:
            return False
        for dx in range(-self.neighbor_ring, self.neighbor_ring + 1):
            for dy in range(-self.neighbor_ring, self.neighbor_ring + 1):
                if (dx, dy) == (0, 0):
                    continue
                if (cx + dx, cy + dy) in self._blacklist:
                    return True
        return False
