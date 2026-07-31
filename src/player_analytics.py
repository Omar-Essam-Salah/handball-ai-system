"""
Player Analytics Engine — deep per-player profiling.
"""

import json
import math
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Court zones (meter space, left→right = attack direction) ──
ZONES = {
    "own_goal_area":   (0,   0,   6,   20),   
    "own_9m":          (0,   0,   9,   20),   
    "own_half":        (0,   0,   20,  20),   
    "centre":          (16,  0,   24,  20),   
    "opp_half":        (20,  0,   40,  20),   
    "opp_9m":          (31,  0,   40,  20),   
    "opp_6m":          (34,  0,   40,  20),   
}

FPS_ASSUMED   = 25.0     
SPRINT_SPEED  = 5.5      
WALK_SPEED    = 1.2      
HEATMAP_W     = 80       
HEATMAP_H     = 40       
HISTORY_LEN   = 3000     
SAVE_EVERY    = 300      


@dataclass
class ActionCounts:
    shots:       int = 0
    goals:       int = 0
    saves:       int = 0
    assists:     int = 0   
    fast_breaks: int = 0
    tackles:     int = 0   

    def to_dict(self) -> dict:
        return {
            "shots":       self.shots,
            "goals":       self.goals,
            "saves":       self.saves,
            "assists":     self.assists,
            "fast_breaks": self.fast_breaks,
            "tackles":     self.tackles,
        }

@dataclass
class PeriodStats:
    period:          int
    distance_m:      float = 0.0
    max_speed_ms:    float = 0.0
    avg_speed_ms:    float = 0.0
    sprint_count:    int   = 0
    minutes_played:  float = 0.0
    actions:         ActionCounts = field(default_factory=ActionCounts)
    zone_time:       dict  = field(default_factory=dict)

@dataclass
class PlayerProfile:
    player_id:      str
    name:           str
    team:           str
    shirt_number:   int   = 0
    role:           str   = ""
    inferred_role:  str   = ""

    total_distance_m:  float = 0.0
    max_speed_ms:      float = 0.0
    avg_speed_ms:      float = 0.0
    total_sprint_count: int  = 0

    actions: ActionCounts = field(default_factory=ActionCounts)
    periods: list[PeriodStats] = field(default_factory=list)
    zone_fractions: dict = field(default_factory=dict)

    heatmap: np.ndarray = field(default_factory=lambda: np.zeros((HEATMAP_H, HEATMAP_W), np.float32))

    performance_score: float = 0.0   
    grade:             str   = "—"   

class PlayerAnalytics:
    def __init__(self, save_dir: str | Path):
        self._dir = Path(save_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, PlayerProfile] = {}
        self._pos_hist: dict[str, deque] = {}
        self._speed_buf: dict[str, deque] = {}   
        self._last_frame: dict[str, int]   = {}
        self._frames_seen: dict[str, int]  = {}
        self._current_period: int   = 1
        self._period_start_ts: float = time.perf_counter()
        self._fps: float             = FPS_ASSUMED
        self.load()

    def update(
        self,
        player_id:    str,
        name:         str,
        team:         str,
        shirt_number: int,
        px: float, py: float,
        frame_n: int,
        fps: float = 25.0,
        xm: float | None = None,
        ym: float | None = None,
        frame_w: int = 1280,
        frame_h: int = 720,
    ) -> None:
        self._fps = fps or FPS_ASSUMED
        prof = self._get_or_create(player_id, name, team, shirt_number)
        self._frames_seen[player_id] = self._frames_seen.get(player_id, 0) + 1

        if xm is None: xm = px / frame_w * 40.0
        if ym is None: ym = py / frame_h * 20.0

        now = time.perf_counter()
        hist = self._pos_hist.setdefault(player_id, deque(maxlen=HISTORY_LEN))
        speed_ms = 0.0
        
        if hist:
            prev_xm, prev_ym, prev_t = hist[-1]
            dt = now - prev_t
            if 0 < dt < 0.5:   
                dx, dy = xm - prev_xm, ym - prev_ym
                dist_m = math.hypot(dx, dy)
                speed_ms = dist_m / dt

                if speed_ms < 30.0:   
                    prof.total_distance_m += dist_m
                    if speed_ms > prof.max_speed_ms: prof.max_speed_ms = round(speed_ms, 2)
                    if speed_ms > SPRINT_SPEED: prof.total_sprint_count += 1

        hist.append((xm, ym, now))

        sbuf = self._speed_buf.setdefault(player_id, deque(maxlen=90))
        sbuf.append(speed_ms)
        valid = [s for s in sbuf if s > 0]
        if valid: prof.avg_speed_ms = round(float(np.mean(valid)), 2)

        col = int(np.clip(xm / 40.0 * HEATMAP_W, 0, HEATMAP_W - 1))
        row = int(np.clip(ym / 20.0 * HEATMAP_H, 0, HEATMAP_H - 1))
        prof.heatmap[row, col] += 1.0

        for zone, (zx1, zy1, zx2, zy2) in ZONES.items():
            if zx1 <= xm <= zx2 and zy1 <= ym <= zy2:
                cnt = prof.zone_fractions.get(zone, 0.0)
                prof.zone_fractions[zone] = cnt + 1.0

        self._infer_role(prof, xm, ym)

        prof.performance_score = self._calc_performance(prof)
        prof.grade = _score_to_grade(prof.performance_score)
        self._last_frame[player_id] = frame_n

        if frame_n % SAVE_EVERY == 0: self.save()

    def record_action(self, player_id: str, action: str) -> None:
        prof = self._profiles.get(player_id)
        if prof is None: return
        ac = prof.actions
        if action == "goal":      ac.goals += 1; ac.shots += 1
        elif action == "shot":    ac.shots += 1
        elif action == "save":    ac.saves += 1
        elif action == "assist":  ac.assists += 1
        elif action == "fast_break": ac.fast_breaks += 1
        elif action == "tackle":  ac.tackles += 1
        prof.performance_score = self._calc_performance(prof)
        prof.grade = _score_to_grade(prof.performance_score)

    def set_period(self, period: int) -> None:
        self._current_period = period
        self._period_start_ts = time.perf_counter()

    def get_profile(self, player_id: str) -> PlayerProfile | None:
        return self._profiles.get(player_id)

    def all_profiles(self) -> list[PlayerProfile]:
        return list(self._profiles.values())

    def heatmap_png(self, player_id: str) -> bytes | None:
        prof = self._profiles.get(player_id)
        if prof is None: return None
        try:
            import cv2
            hm = prof.heatmap.copy()
            if hm.max() == 0: return None
            hm = (hm / hm.max() * 255).astype(np.uint8)
            hm = cv2.resize(hm, (400, 200), interpolation=cv2.INTER_LINEAR)
            colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            _, buf = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes()
        except Exception:
            return None

    def profile_dict(self, player_id: str) -> dict:
        prof = self._profiles.get(player_id)
        if prof is None: return {}
        total_cells = max(sum(prof.zone_fractions.values()), 1)
        zone_pct = {z: round(v / total_cells * 100, 1) for z, v in prof.zone_fractions.items()}
        return {
            "player_id":       prof.player_id,
            "name":            prof.name,
            "team":            prof.team,
            "shirt_number":    prof.shirt_number,
            "role":            prof.role or prof.inferred_role,
            "inferred_role":   prof.inferred_role,
            "distance_km":     round(prof.total_distance_m / 1000, 2),
            "max_speed_ms":    prof.max_speed_ms,
            "avg_speed_ms":    prof.avg_speed_ms,
            "sprint_count":    prof.total_sprint_count,
            "actions":         prof.actions.to_dict(),
            "zone_pct":        zone_pct,
            "performance":     round(prof.performance_score, 1),
            "grade":           prof.grade,
            "frames_seen":     self._frames_seen.get(player_id, 0),
        }

    def save(self) -> None:
        for pid, prof in self._profiles.items():
            d = {
                "name":            prof.name, "team": prof.team, "shirt_number": prof.shirt_number,
                "role":            prof.role, "inferred_role": prof.inferred_role,
                "total_distance_m": round(prof.total_distance_m, 2), "max_speed_ms": prof.max_speed_ms,
                "avg_speed_ms":    prof.avg_speed_ms, "total_sprint_count": prof.total_sprint_count,
                "actions":         prof.actions.to_dict(), "zone_fractions": prof.zone_fractions,
                "performance_score": round(prof.performance_score, 2), "grade": prof.grade,
                "frames_seen":     self._frames_seen.get(pid, 0),
            }
            (self._dir / f"{pid}.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
            np.save(str(self._dir / f"{pid}_heatmap.npy"), prof.heatmap)

    def load(self) -> None:
        for path in self._dir.glob("*.json"):
            pid = path.stem
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                prof = PlayerProfile(
                    player_id=pid, name=d.get("name", ""), team=d.get("team", "unknown"),
                    shirt_number=d.get("shirt_number", 0), role=d.get("role", ""),
                    inferred_role=d.get("inferred_role", ""), total_distance_m=d.get("total_distance_m", 0.0),
                    max_speed_ms=d.get("max_speed_ms", 0.0), avg_speed_ms=d.get("avg_speed_ms", 0.0),
                    total_sprint_count=d.get("total_sprint_count", 0), zone_fractions=d.get("zone_fractions", {}),
                    performance_score=d.get("performance_score", 0.0), grade=d.get("grade", "—"),
                )
                ac_d = d.get("actions", {})
                prof.actions = ActionCounts(
                    shots=ac_d.get("shots", 0), goals=ac_d.get("goals", 0), saves=ac_d.get("saves", 0),
                    assists=ac_d.get("assists", 0), fast_breaks=ac_d.get("fast_breaks", 0), tackles=ac_d.get("tackles", 0),
                )
                hm_path = self._dir / f"{pid}_heatmap.npy"
                prof.heatmap = np.load(str(hm_path)) if hm_path.exists() else np.zeros((HEATMAP_H, HEATMAP_W), np.float32)
                self._profiles[pid] = prof
                self._frames_seen[pid] = d.get("frames_seen", 0)
            except Exception: pass

    def _get_or_create(self, pid: str, name: str, team: str, shirt_number: int) -> PlayerProfile:
        if pid not in self._profiles:
            self._profiles[pid] = PlayerProfile(player_id=pid, name=name, team=team, shirt_number=shirt_number)
        else:
            p = self._profiles[pid]
            if name and not name.startswith("Player "): p.name = name
            if team and team != "unknown": p.team = team
            if shirt_number: p.shirt_number = shirt_number
        return self._profiles[pid]

    # ── التعلم الذاتي التكتيكي للمراكز ──
    def _infer_role(self, prof: PlayerProfile, xm: float, ym: float) -> None:
        zf = prof.zone_fractions
        total = max(sum(zf.values()), 1)
        if total < 50: return # لا تستعجل الحكم قبل رؤية اللاعب كفاية
        
        own_ga  = zf.get("own_goal_area", 0) / total
        opp_6m  = zf.get("opp_6m", 0) / total
        opp_9m  = zf.get("opp_9m", 0) / total
        centre  = zf.get("centre", 0) / total
        own_half = zf.get("own_half", 0) / total

        # قانون كرة اليد المترجم مكانياً
        if own_ga > 0.40 or (own_half > 0.85 and prof.total_distance_m < 1500):
            prof.inferred_role = "GK"
        elif opp_6m > 0.25:
            if ym < 5.0: prof.inferred_role = "RW"      
            elif ym > 15.0: prof.inferred_role = "LW"   
            else: prof.inferred_role = "PV"             
        elif opp_9m > 0.20 or centre > 0.30:
            if ym < 7.5: prof.inferred_role = "RB"      
            elif ym > 12.5: prof.inferred_role = "LB"   
            else: prof.inferred_role = "CB"             
        else:
            prof.inferred_role = "CB"

    def _calc_performance(self, prof: PlayerProfile) -> float:
        score = min(prof.actions.goals * 8.0, 25.0)
        if prof.actions.shots > 0: score += (prof.actions.goals / prof.actions.shots) * 20.0
        if prof.total_distance_m > 0: score += min(prof.total_distance_m / 6000.0, 1.0) * 15.0
        score += min(prof.max_speed_ms / 9.0, 1.0) * 10.0
        score += min(prof.actions.fast_breaks * 5.0, 10.0)
        if (prof.role or prof.inferred_role) == "GK": score += min(prof.actions.saves * 5.0, 20.0)
        score += min(prof.actions.assists * 4.0, 10.0)
        return round(min(score, 100.0), 1)

def _score_to_grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B+"
    if score >= 60: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    if score > 0:   return "F"
    return "—"