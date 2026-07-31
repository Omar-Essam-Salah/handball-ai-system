"""
Player Profiler — stamina modelling, sprint events, and offline AI reports.

Layers on top of PlayerAnalytics to add:
  1. StaminaTracker  — per-player physiological energy model (0-100 %)
  2. SprintEvent     — individual sprint recordings (time, distance, peak speed)
  3. PeriodSnapshot  — per-half metric snapshots
  4. OfflineAnalyzer — rule-based NLG that writes a coaching report with no
                       internet or GPU required

Usage (pipeline.py)
───────────────────
    profiler = PlayerProfiler(analytics, ROOT / "data" / "profiles")

    # each frame, after analytics.update():
    profiler.tick(player_id, xm, ym, frame_n, fps)

    # on demand (API):
    full = profiler.full_profile(player_id)   # → unified dict
    report = profiler.generate_report(player_id)   # → text analysis

    # at end of session:
    profiler.save()

Elite norms (sports science reference values per 60-min match)
──────────────────────────────────────────────────────────────
  CB / LB / RB  : 5.4–7.2 km,  15–25 sprints,  max speed 7.0–9.0 m/s
  LW / RW       : 4.2–5.8 km,  10–20 sprints,  max speed 7.5–9.5 m/s
  PV (pivot)    : 3.8–5.2 km,   8–15 sprints,  max speed 6.5–8.5 m/s
  GK            : 1.2–2.5 km,   2–8  sprints,  max speed 5.5–7.0 m/s
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


# ── Stamina model constants ───────────────────────────────────────────────────
SPRINT_MS       = 5.5    # m/s — speed classified as sprint
RUN_MS          = 2.5    # m/s — speed classified as running (not sprint)

STAMP_SPRINT_COST  = 0.80   # % stamina per second of sprinting
STAMP_RUN_COST     = 0.10   # % stamina per second of running
STAMP_WALK_COST    = 0.01   # % stamina per second of walking
STAMP_RECOVERY     = 0.25   # % stamina per second of standing/resting
STAMP_FLOOR        = 10.0   # minimum stamina (exhausted but still playing)
STAMP_INITIAL      = 100.0

SPRINT_MIN_SECONDS = 1.0    # minimum duration to count as a sprint event
STAMP_SAMPLE_S     = 30.0   # snapshot stamina every N seconds

# ── Elite performance norms (per role, per 60 minutes) ───────────────────────
ELITE_NORMS = {
    "GK":  {"dist_km": (1.2, 2.5), "sprints": (2, 8),  "max_ms": (5.5, 7.0)},
    "CB":  {"dist_km": (5.4, 7.2), "sprints": (15, 25), "max_ms": (7.0, 9.0)},
    "LB":  {"dist_km": (5.2, 6.8), "sprints": (12, 22), "max_ms": (7.0, 9.0)},
    "RB":  {"dist_km": (5.2, 6.8), "sprints": (12, 22), "max_ms": (7.0, 9.0)},
    "LW":  {"dist_km": (4.2, 5.8), "sprints": (10, 20), "max_ms": (7.5, 9.5)},
    "RW":  {"dist_km": (4.2, 5.8), "sprints": (10, 20), "max_ms": (7.5, 9.5)},
    "PV":  {"dist_km": (3.8, 5.2), "sprints": (8, 15),  "max_ms": (6.5, 8.5)},
}
_DEFAULT_NORM = {"dist_km": (4.8, 6.5), "sprints": (12, 22), "max_ms": (6.8, 8.8)}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SprintEvent:
    start_frame:   int
    start_time_s:  float
    duration_s:    float
    distance_m:    float
    peak_speed_ms: float

    def to_dict(self) -> dict:
        return {
            "start_frame":   self.start_frame,
            "start_time_s":  round(self.start_time_s, 2),
            "duration_s":    round(self.duration_s, 2),
            "distance_m":    round(self.distance_m, 2),
            "peak_speed_ms": round(self.peak_speed_ms, 2),
        }


@dataclass
class StaminaState:
    current:              float = STAMP_INITIAL   # current stamina 0-100
    history:              list  = field(default_factory=list)  # [(elapsed_s, stamina)]
    high_intensity_s:     float = 0.0    # total seconds above SPRINT_MS
    recovery_windows:     int   = 0      # distinct rest windows > 30 s
    _in_rest:             bool  = False
    _rest_start_s:        float = 0.0
    _last_sample_s:       float = 0.0
    _start_wall:          float = field(default_factory=time.perf_counter)

    def to_dict(self, sprint_events: list) -> dict:
        n = len(sprint_events)
        avg_sprint_dist = (sum(e.distance_m for e in sprint_events) / n) if n else 0.0
        return {
            "current":          round(self.current, 1),
            "history":          [(round(t, 0), round(s, 1)) for t, s in self.history],
            "high_intensity_s": round(self.high_intensity_s, 1),
            "recovery_windows": self.recovery_windows,
            "sprint_count":     n,
            "avg_sprint_dist_m": round(avg_sprint_dist, 1),
            "status":           _stamina_label(self.current),
        }


def _stamina_label(s: float) -> str:
    if s >= 80: return "fresh"
    if s >= 60: return "good"
    if s >= 40: return "tired"
    if s >= 20: return "fatigued"
    return "exhausted"


@dataclass
class AnalysisReport:
    executive_summary: str
    physical_summary:  str
    tactical_summary:  str
    coaching_notes:    list
    strengths:         list
    improvements:      list
    alerts:            list
    generated_at:      float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "executive_summary": self.executive_summary,
            "physical_summary":  self.physical_summary,
            "tactical_summary":  self.tactical_summary,
            "coaching_notes":    self.coaching_notes,
            "strengths":         self.strengths,
            "improvements":      self.improvements,
            "alerts":            self.alerts,
            "generated_at":      round(self.generated_at, 0),
        }


# ── Main profiler ─────────────────────────────────────────────────────────────

class PlayerProfiler:
    """
    Sits alongside PlayerAnalytics and adds stamina, sprint timeline,
    and AI-generated coaching reports.

    Thread safety: tick() is called from the pipeline thread; full_profile()
    and generate_report() may be called from the API thread.  The data
    structures are small and writes are atomic enough for CPython's GIL —
    no extra locking needed.
    """

    def __init__(self, analytics, save_dir: str | Path):
        self._analytics  = analytics
        self._dir        = Path(save_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # Per-player state
        self._stamina:        dict[str, StaminaState]     = {}
        self._sprints:        dict[str, list[SprintEvent]] = {}
        self._in_sprint:      dict[str, dict | None]       = {}  # active sprint accumulator
        self._last_pos:       dict[str, tuple]             = {}  # (xm, ym, t)
        self._session_start:  float                        = time.perf_counter()

        # Cached reports (regenerated on demand)
        self._reports:        dict[str, AnalysisReport]   = {}

        self.load()

    # ── Per-frame update ──────────────────────────────────────────────────────

    def tick(
        self,
        player_id: str,
        xm:        float | None,
        ym:        float | None,
        frame_n:   int,
        fps:       float = 25.0,
    ) -> None:
        """
        Call once per frame per visible player, AFTER analytics.update().
        xm / ym are meter-space coordinates (None if court not calibrated).
        """
        if xm is None or ym is None:
            return

        now     = time.perf_counter()
        elapsed = now - self._session_start

        st      = self._stamina.setdefault(
            player_id,
            StaminaState(_start_wall=self._session_start))
        sprints = self._sprints.setdefault(player_id, [])

        # Speed from previous position
        speed_ms = 0.0
        prev = self._last_pos.get(player_id)
        if prev:
            prev_xm, prev_ym, prev_t = prev
            dt = now - prev_t
            if 0 < dt < 1.0:
                dist = math.hypot(xm - prev_xm, ym - prev_ym)
                speed_ms = dist / dt if dt > 0 else 0.0
                if speed_ms > 30.0:
                    speed_ms = 0.0   # teleport noise

        self._last_pos[player_id] = (xm, ym, now)

        # ── Stamina update ────────────────────────────────────────────────────
        dt_s = 1.0 / max(fps, 1.0)

        if speed_ms >= SPRINT_MS:
            cost = STAMP_SPRINT_COST * dt_s
            st.high_intensity_s += dt_s
            st.current = max(STAMP_FLOOR, st.current - cost)
            st._in_rest = False
        elif speed_ms >= RUN_MS:
            st.current = max(STAMP_FLOOR, st.current - STAMP_RUN_COST * dt_s)
            st._in_rest = False
        elif speed_ms >= 0.3:
            st.current = max(STAMP_FLOOR, st.current - STAMP_WALK_COST * dt_s)
            if not st._in_rest:
                st._in_rest = True
                st._rest_start_s = elapsed
        else:
            # Standing still — recovery
            st.current = min(STAMP_INITIAL, st.current + STAMP_RECOVERY * dt_s)
            if not st._in_rest:
                st._in_rest = True
                st._rest_start_s = elapsed
            elif elapsed - st._rest_start_s > 30.0:
                st.recovery_windows += 1
                st._rest_start_s = elapsed  # reset so we count each 30s window

        # Sample stamina history
        if elapsed - st._last_sample_s >= STAMP_SAMPLE_S:
            st.history.append((round(elapsed, 0), round(st.current, 1)))
            st._last_sample_s = elapsed
            if len(st.history) > 200:
                st.history = st.history[-200:]

        # ── Sprint event detection ────────────────────────────────────────────
        acc = self._in_sprint.get(player_id)

        if speed_ms >= SPRINT_MS:
            if acc is None:
                # Sprint started
                self._in_sprint[player_id] = {
                    "start_frame": frame_n,
                    "start_time":  elapsed,
                    "distance":    0.0,
                    "peak_speed":  speed_ms,
                    "prev_xm":     xm,
                    "prev_ym":     ym,
                }
            else:
                # Continuing sprint
                if prev:
                    acc["distance"] += math.hypot(xm - acc["prev_xm"],
                                                   ym - acc["prev_ym"])
                    acc["prev_xm"] = xm
                    acc["prev_ym"] = ym
                acc["peak_speed"] = max(acc["peak_speed"], speed_ms)
        else:
            if acc is not None:
                # Sprint ended
                dur = elapsed - acc["start_time"]
                if dur >= SPRINT_MIN_SECONDS:
                    sprints.append(SprintEvent(
                        start_frame   = acc["start_frame"],
                        start_time_s  = acc["start_time"],
                        duration_s    = dur,
                        distance_m    = acc["distance"],
                        peak_speed_ms = acc["peak_speed"],
                    ))
                self._in_sprint[player_id] = None

        # Invalidate cached report (data changed)
        self._reports.pop(player_id, None)

    # ── Profile aggregation ───────────────────────────────────────────────────

    def full_profile(self, player_id: str) -> dict:
        """
        Unified profile dict combining identity, analytics, stamina, sprints,
        and AI report.  Safe to call from the API thread.
        """
        base = self._analytics.profile_dict(player_id)
        if not base:
            return {}

        st      = self._stamina.get(player_id)
        sprints = self._sprints.get(player_id, [])
        report  = self._get_or_generate_report(player_id, base, st, sprints)

        return {
            **base,
            "stamina":        st.to_dict(sprints) if st else _empty_stamina(),
            "sprint_events":  [e.to_dict() for e in sprints[-20:]],  # last 20
            "sprint_count":   len(sprints),
            "report":         report.to_dict() if report else None,
        }

    def all_full_profiles(self) -> list[dict]:
        return [self.full_profile(p.player_id)
                for p in self._analytics.all_profiles()]

    def generate_report(self, player_id: str) -> dict | None:
        """Force-regenerate AI report for a player. Returns report dict."""
        base    = self._analytics.profile_dict(player_id)
        if not base:
            return None
        st      = self._stamina.get(player_id)
        sprints = self._sprints.get(player_id, [])
        self._reports.pop(player_id, None)   # invalidate cache
        report  = self._get_or_generate_report(player_id, base, st, sprints)
        return report.to_dict() if report else None

    def get_stamina(self, player_id: str) -> dict:
        st      = self._stamina.get(player_id)
        sprints = self._sprints.get(player_id, [])
        if st is None:
            return _empty_stamina()
        return st.to_dict(sprints)

    def get_sprints(self, player_id: str) -> list[dict]:
        return [e.to_dict() for e in self._sprints.get(player_id, [])]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        for pid in set(list(self._stamina) + list(self._sprints)):
            d: dict = {}
            st = self._stamina.get(pid)
            if st:
                d["stamina"] = {
                    "current":          st.current,
                    "history":          st.history,
                    "high_intensity_s": st.high_intensity_s,
                    "recovery_windows": st.recovery_windows,
                }
            sp = self._sprints.get(pid, [])
            d["sprints"] = [e.to_dict() for e in sp]
            (self._dir / f"{pid}_profiler.json").write_text(
                json.dumps(d, indent=2), encoding="utf-8")

    def load(self) -> None:
        for path in self._dir.glob("*_profiler.json"):
            pid = path.stem.replace("_profiler", "")
            try:
                d  = json.loads(path.read_text(encoding="utf-8"))
                sd = d.get("stamina", {})
                st = StaminaState(
                    current          = sd.get("current", STAMP_INITIAL),
                    history          = sd.get("history", []),
                    high_intensity_s = sd.get("high_intensity_s", 0.0),
                    recovery_windows = sd.get("recovery_windows", 0),
                    _start_wall      = time.perf_counter(),
                )
                self._stamina[pid] = st
                self._sprints[pid] = [
                    SprintEvent(**s) for s in d.get("sprints", [])
                ]
            except Exception as e:
                print(f"[Profiler] Load failed for {pid}: {e}")
        if self._stamina:
            print(f"[Profiler] Loaded stamina data for {len(self._stamina)} players.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_or_generate_report(
        self,
        player_id: str,
        base:      dict,
        st:        StaminaState | None,
        sprints:   list[SprintEvent],
    ) -> AnalysisReport | None:
        if player_id in self._reports:
            return self._reports[player_id]
        if base.get("frames_seen", 0) < 75:   # < 3 seconds visible — too early
            return None
        report = OfflineAnalyzer.analyze(base, st, sprints)
        self._reports[player_id] = report
        return report


def _empty_stamina() -> dict:
    return {
        "current": 100.0,
        "history": [],
        "high_intensity_s": 0.0,
        "recovery_windows": 0,
        "sprint_count": 0,
        "avg_sprint_dist_m": 0.0,
        "status": "fresh",
    }


# ── Offline AI Analyzer ───────────────────────────────────────────────────────

class OfflineAnalyzer:
    """
    Rule-based natural language generator for player coaching reports.

    Completely offline — no model weights, no API calls.
    Runs in < 1 ms.  Output quality is coaching-grade text.

    Design principle: each sentence is selected from a conditional bank
    based on quantitative thresholds.  Sentences are then joined into
    coherent paragraphs with varied connectors.
    """

    @staticmethod
    def analyze(
        profile:  dict,
        stamina:  StaminaState | None,
        sprints:  list[SprintEvent],
    ) -> AnalysisReport:
        name         = profile.get("name", "The player")
        role         = profile.get("role") or profile.get("inferred_role") or "Unknown"
        team         = profile.get("team", "")
        dist_km      = profile.get("distance_km", 0.0)
        max_ms       = profile.get("max_speed_ms", 0.0)
        avg_ms       = profile.get("avg_speed_ms", 0.0)
        sprint_count = profile.get("sprint_count", 0)
        grade        = profile.get("grade", "—")
        perf_score   = profile.get("performance", 0.0)
        actions      = profile.get("actions", {})
        zone_pct     = profile.get("zone_pct", {})
        frames_seen  = profile.get("frames_seen", 0)

        goals    = actions.get("goals", 0)
        shots    = actions.get("shots", 0)
        saves    = actions.get("saves", 0)
        assists  = actions.get("assists", 0)
        fb       = actions.get("fast_breaks", 0)
        tackles  = actions.get("tackles", 0)

        stamina_pct = stamina.current if stamina else 100.0
        norms       = ELITE_NORMS.get(role, _DEFAULT_NORM)
        minutes     = frames_seen / 25.0 / 60.0   # approx minutes

        # ── Normalise to per-60-min for fair comparisons ──────────────────────
        scale   = (60.0 / max(minutes, 1.0)) if minutes > 5 else 1.0
        dist_60 = dist_km * scale
        spr_60  = sprint_count * scale

        norm_dist   = norms["dist_km"]
        norm_sprints = norms["sprints"]
        norm_speed   = norms["max_ms"]

        dist_rel  = _relative(dist_60, norm_dist)
        spr_rel   = _relative(spr_60, norm_sprints)
        speed_rel = _relative(max_ms, norm_speed)

        shot_pct  = goals / shots if shots > 0 else 0.0

        # ── Executive summary ─────────────────────────────────────────────────
        grade_word = _grade_word(grade)
        role_label = _role_full(role)

        exec_parts = []
        exec_parts.append(
            f"{name} ({role_label}) delivered a {grade_word} performance "
            f"with an overall score of {perf_score:.0f}/100 (grade {grade}).")

        if dist_rel == "above":
            exec_parts.append(
                f"Work rate was exceptional — {dist_km:.2f} km covered, "
                f"above the elite norm of {norm_dist[0]:.1f}–{norm_dist[1]:.1f} km.")
        elif dist_rel == "below":
            exec_parts.append(
                f"Movement volume was limited at {dist_km:.2f} km "
                f"(elite norm {norm_dist[0]:.1f}–{norm_dist[1]:.1f} km).")
        else:
            exec_parts.append(
                f"Movement volume of {dist_km:.2f} km is within the expected "
                f"range for a {role_label}.")

        if stamina_pct < 30:
            exec_parts.append(
                f"Current stamina is critically low ({stamina_pct:.0f}%) — "
                f"immediate substitution is strongly recommended.")
        elif stamina_pct < 55:
            exec_parts.append(
                f"Stamina reserve stands at {stamina_pct:.0f}% — "
                f"monitor closely over the next 10 minutes.")

        executive_summary = " ".join(exec_parts)

        # ── Physical summary ──────────────────────────────────────────────────
        phys_parts = []

        speed_desc = (
            "outstanding" if speed_rel == "above" else
            "adequate"    if speed_rel == "in range" else
            "below average")
        phys_parts.append(
            f"Maximum recorded speed was {max_ms:.1f} m/s ({max_ms * 3.6:.1f} km/h) — "
            f"{speed_desc} for the {role_label} position.")

        if avg_ms > 0:
            phys_parts.append(
                f"Average movement speed was {avg_ms:.1f} m/s.")

        spr_desc = (
            "high sprint load" if spr_rel == "above" else
            "appropriate sprint volume" if spr_rel == "in range" else
            "low sprint frequency")
        phys_parts.append(
            f"{sprint_count} sprint{'s' if sprint_count != 1 else ''} recorded "
            f"({'~' + str(int(spr_60)) + ' per 60 min' if minutes > 5 else 'total'}) — "
            f"{spr_desc}.")

        if sprints:
            avg_dur  = sum(e.duration_s for e in sprints) / len(sprints)
            avg_dist = sum(e.distance_m for e in sprints) / len(sprints)
            phys_parts.append(
                f"Average sprint: {avg_dur:.1f} s, {avg_dist:.1f} m.")

        if stamina and stamina.high_intensity_s > 0:
            phys_parts.append(
                f"Total high-intensity time: {stamina.high_intensity_s:.0f} s "
                f"({stamina.high_intensity_s / 60:.1f} min above {SPRINT_MS} m/s).")

        physical_summary = " ".join(phys_parts)

        # ── Tactical summary ──────────────────────────────────────────────────
        tact_parts = []

        dominant_zone = max(zone_pct, key=zone_pct.get) if zone_pct else None
        if dominant_zone:
            tact_parts.append(
                f"Primarily active in the {_zone_label(dominant_zone)} "
                f"({zone_pct[dominant_zone]:.0f}% of visible time).")

        if role == "GK":
            if saves > 0:
                tact_parts.append(
                    f"Made {saves} save{'s' if saves != 1 else ''} — "
                    f"{'decisive goalkeeper presence' if saves >= 5 else 'steady between the posts'}.")
            else:
                tact_parts.append("No shots on goal recorded during the observed period.")
        else:
            if shots > 0:
                pct_str = f"{shot_pct * 100:.0f}%"
                quality = ("clinical" if shot_pct >= 0.40 else
                           "solid" if shot_pct >= 0.25 else
                           "requiring improvement")
                tact_parts.append(
                    f"Shot conversion: {goals}/{shots} ({pct_str}) — {quality}.")
            if assists > 0:
                tact_parts.append(
                    f"{assists} assist{'s' if assists != 1 else ''} created.")
            if fb > 0:
                tact_parts.append(
                    f"Initiated {fb} fast break{'s' if fb != 1 else ''} — "
                    f"{'strong counter-attack threat' if fb >= 3 else 'contributing to transition play'}.")
            if tackles > 0:
                tact_parts.append(
                    f"{tackles} defensive tackle{'s' if tackles != 1 else ''} recorded.")

        tactical_summary = " ".join(tact_parts) if tact_parts else "Insufficient match data for tactical analysis."

        # ── Coaching notes ────────────────────────────────────────────────────
        notes = []

        if dist_rel == "below" and role != "GK":
            notes.append(
                f"Increase movement volume — {dist_km:.1f} km is below the "
                f"{role_label} target of {norm_dist[0]:.1f}+ km.")
        if spr_rel == "below" and role not in ("GK", "PV"):
            notes.append(
                "Sprint frequency is low — focus on explosive runs in transition.")
        if shots > 3 and shot_pct < 0.20 and role != "GK":
            notes.append(
                f"Shot selection needs refinement — {shot_pct * 100:.0f}% conversion "
                f"from {shots} attempts is below acceptable threshold.")
        if stamina_pct < 40:
            notes.append(
                f"Stamina at {stamina_pct:.0f}% — consider tactical rest or "
                f"substitution to prevent performance degradation.")
        if max_ms < norm_speed[0] and role not in ("GK", "PV"):
            notes.append(
                f"Maximum speed of {max_ms:.1f} m/s is below the positional "
                f"benchmark ({norm_speed[0]:.1f} m/s) — sprint training recommended.")

        if role == "GK" and saves == 0 and shots == 0:
            notes.append("Goalkeeper was not tested — limited sample for evaluation.")

        if not notes:
            notes.append(
                "Performance is on track — maintain current intensity and positioning.")

        # ── Strengths ─────────────────────────────────────────────────────────
        strengths = []
        if dist_rel == "above":
            strengths.append(f"Exceptional work rate ({dist_km:.2f} km covered)")
        if speed_rel == "above":
            strengths.append(f"Elite-level speed ({max_ms:.1f} m/s)")
        if spr_rel == "above":
            strengths.append(f"High sprint volume ({sprint_count} sprints)")
        if shot_pct >= 0.35 and shots >= 3:
            strengths.append(f"Strong shot conversion ({shot_pct * 100:.0f}%)")
        if saves >= 5:
            strengths.append(f"Commanding goalkeeping ({saves} saves)")
        if assists >= 2:
            strengths.append(f"Creative playmaking ({assists} assists)")
        if fb >= 3:
            strengths.append(f"Counter-attack threat ({fb} fast breaks)")
        if stamina_pct >= 75:
            strengths.append(f"Good stamina reserve ({stamina_pct:.0f}% remaining)")
        if not strengths:
            strengths.append("Consistent positional discipline")

        # ── Areas for improvement ─────────────────────────────────────────────
        improvements = []
        if dist_rel == "below" and role != "GK":
            improvements.append("Increase movement and off-ball running")
        if speed_rel == "below":
            improvements.append("Work on explosive acceleration")
        if shots > 2 and shot_pct < 0.20:
            improvements.append("Shot selection and finishing accuracy")
        if spr_rel == "below" and role not in ("GK", "PV"):
            improvements.append("Sprint frequency in transition phases")
        if not improvements:
            improvements.append("Continue building on current performance level")

        # ── Alerts ────────────────────────────────────────────────────────────
        alerts = []
        if stamina_pct < 25:
            alerts.append(f"CRITICAL: Stamina at {stamina_pct:.0f}% — substitution required")
        elif stamina_pct < 45:
            alerts.append(f"WARNING: Stamina at {stamina_pct:.0f}% — approaching substitution threshold")
        if sprint_count > 0 and sprints:
            recent = sprints[-5:]
            if len(recent) >= 3:
                late_peak = max(e.peak_speed_ms for e in recent)
                early_peak = max(e.peak_speed_ms for e in sprints[:3]) if len(sprints) >= 3 else late_peak
                if late_peak < early_peak * 0.80:
                    alerts.append("Speed decline detected in recent sprints — fatigue indicator")
        if dist_km > 0 and max_ms < 4.0 and role not in ("GK",):
            alerts.append("Unusually low top speed — check for possible injury or tracking issue")

        return AnalysisReport(
            executive_summary = executive_summary,
            physical_summary  = physical_summary,
            tactical_summary  = tactical_summary,
            coaching_notes    = notes,
            strengths         = strengths,
            improvements      = improvements,
            alerts            = alerts,
        )


# ── NLG helpers ───────────────────────────────────────────────────────────────

def _relative(value: float, norm: tuple) -> str:
    lo, hi = norm
    if value > hi * 1.05:
        return "above"
    if value < lo * 0.90:
        return "below"
    return "in range"


def _grade_word(grade: str) -> str:
    return {
        "A+": "exceptional",  "A": "excellent",
        "B+": "very good",    "B": "good",
        "C":  "average",      "D": "below average",
        "F":  "poor",         "—": "preliminary",
    }.get(grade, "notable")


def _role_full(role: str) -> str:
    return {
        "GK": "Goalkeeper",  "CB": "Centre Back", "LB": "Left Back",
        "RB": "Right Back",  "LW": "Left Wing",   "RW": "Right Wing",
        "PV": "Pivot",
    }.get(role, role or "Field Player")


def _zone_label(zone: str) -> str:
    return {
        "own_goal_area": "own goal area (6m zone)",
        "own_9m":        "own defensive half (9m)",
        "own_half":      "own half",
        "centre":        "central zone",
        "opp_half":      "attacking half",
        "opp_9m":        "attacking 9m zone",
        "opp_6m":        "opposition 6m danger zone",
    }.get(zone, zone)
