"""
Temporal Ball Consensus — TrackNet-inspired multi-frame voting.

Why this exists
───────────────
The Kalman filter (ball_kalman.py) gates SINGLE candidates against a smoothed
state — it works once initialised, but on the first 1-2 frames after a reset
(ball lost) it accepts any candidate. That's when head FPs and floor markers
sneak in.

TrackNet's core insight (papers V3/V4/V5) is to use *N consecutive frames* as
joint input: a real ball forms a smooth trajectory across frames; head FPs
and markers don't.

This module is a lightweight version of that idea:
    For each candidate at frame t, count how many of the last K frames had a
    "neighbour candidate" within MAX_NEIGHBOUR_PX. A real ball usually has
    1-2 neighbours (its previous positions); an isolated noise candidate has
    zero. We boost the score accordingly.

The output is a ranked list of (candidate, boosted_score) tuples — the
existing tracker scoring still has the final say.

Integration
───────────
    voter = TemporalBallVoter()

    # in tracker.update(), before the per-candidate scoring loop:
    scored = voter.push_and_score(ball_candidates_xy_conf)
    for b, temporal_bonus in scored:
        # ... existing scoring, then:
        score += temporal_bonus
"""

from __future__ import annotations
import math
from collections import deque
from typing import Iterable, Optional


# ── Tunables ──
_HISTORY_LEN          = 3       # frames to look back
_MAX_NEIGHBOUR_PX     = 120     # max distance to count as "same ball, prev frame"
_BONUS_PER_NEIGHBOUR  = 0.18    # additive score per matched past frame
_MAX_TOTAL_BONUS      = 0.50    # cap so a noisy chain can't dominate


class TemporalBallVoter:
    """3-frame temporal consensus for ball candidates.

    Each call to ``push_and_score`` does TWO things:
      1. Compute a "temporal bonus" for each input candidate by counting
         neighbours in the last K frames.
      2. Add the current candidates to history so future calls can vote
         against them.

    The bonus is purely additive — the tracker still applies its own
    hand-magnet / anti-shoe / Kalman gating on top.
    """

    def __init__(
        self,
        history_len:      int   = _HISTORY_LEN,
        max_neighbour_px: float = _MAX_NEIGHBOUR_PX,
        bonus_per_match:  float = _BONUS_PER_NEIGHBOUR,
        max_total_bonus:  float = _MAX_TOTAL_BONUS,
    ):
        self._history: deque[list[tuple[float, float, float]]] = deque(maxlen=history_len)
        self._max_n  = float(max_neighbour_px)
        self._bonus  = float(bonus_per_match)
        self._cap    = float(max_total_bonus)

    def push_and_score(
        self,
        candidates: Iterable[tuple[float, float, float]],
    ) -> list[tuple[tuple[float, float, float], float]]:
        """``candidates`` = iterable of (cx, cy, conf). Returns list of
        ((cx, cy, conf), bonus) tuples in the same order."""
        cands = list(candidates)
        if not cands:
            self._history.append([])
            return []

        max_n2 = self._max_n * self._max_n
        out: list[tuple[tuple[float, float, float], float]] = []
        for cx, cy, conf in cands:
            bonus = 0.0
            for past_frame in self._history:
                # Find the *closest* past candidate to this one in this frame
                best_d2 = max_n2
                matched = False
                for px, py, pconf in past_frame:
                    d2 = (cx - px) ** 2 + (cy - py) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        matched = True
                if matched:
                    bonus += self._bonus
                    if bonus >= self._cap:
                        bonus = self._cap
                        break
            out.append(((cx, cy, conf), bonus))

        # Record this frame's candidates for future calls
        self._history.append([(c[0], c[1], c[2]) for c in cands])
        return out

    def reset(self) -> None:
        self._history.clear()

    def stats(self) -> dict:
        return {
            "history_len":      len(self._history),
            "history_size":     [len(f) for f in self._history],
        }
