"""
Shot / Save report model — the professional handball analysis output (matches
the IDEA/ reference PDFs exactly).

Every shot is one record:
    player, number, team, zone, cell, outcome

  • zone   — where on the court the shot was taken (1 of 12 ZONES)
  • cell   — where in the GOAL MOUTH it went, a 3x3 grid:
                LU CU RU      (Left/Center/Right  Up)
                LH CH RH      (                   Half)
                LD CD RD      (                   Down)
             plus the off-target cells: MISS, POST, BLOCK
  • outcome — GOAL | SAVE | MISS | POST | BLOCK

From these records it builds:
  • Shot Report (per shooter): per-zone 3x3 grid of  goals/total
  • Save Report (per GK)     : per-zone 3x3 grid of  saves/total  + the 9-cell
                               summary with save % (LU 33%, CH 75%, …)
  • Distribution pies        : by distance (6m/9m), by outcome
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

ZONES = [
    "Left Wing", "Right Wing", "Left Back", "Right Back", "Center Back",
    "Left 6m", "Center 6m", "Right 6m", "Fast Break", "Break Through",
    "7m Penalty", "Long Distance",
]
# 3x3 goal-mouth cells, row-major (top→bottom, left→right)
CELLS = ["LU", "CU", "RU", "LH", "CH", "RH", "LD", "CD", "RD"]
OFF_TARGET = ["MISS", "POST", "BLOCK"]
OUTCOMES = ["GOAL", "SAVE", "MISS", "POST", "BLOCK"]

# distance bucket per zone (for the distribution pie)
_ZONE_DISTANCE = {
    "Left Wing": "6m", "Right Wing": "6m", "Left 6m": "6m", "Center 6m": "6m",
    "Right 6m": "6m", "Break Through": "6m", "Fast Break": "6m", "7m Penalty": "7m",
    "Left Back": "9m", "Right Back": "9m", "Center Back": "9m", "Long Distance": "9m",
}


class ShotReport:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.shots: list[dict] = []
        if self.path and self.path.exists():
            try:
                self.shots = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.shots = []

    # ── data entry ──────────────────────────────────────────────────────
    def add(self, *, player: str = "?", number: int | None = None,
            team: str = "team_a", zone: str = "Center Back", cell: str = "CH",
            outcome: str = "GOAL", gk: str | None = None, frame_n: int = 0) -> dict:
        zone = zone if zone in ZONES else "Center Back"
        cell = cell if cell in CELLS or cell in OFF_TARGET else "CH"
        outcome = outcome.upper() if outcome.upper() in OUTCOMES else "GOAL"
        rec = {"player": player, "number": number, "team": team, "zone": zone,
               "cell": cell, "outcome": outcome, "gk": gk, "frame_n": frame_n,
               "id": len(self.shots) + 1, "ts": time.time()}
        self.shots.append(rec)
        self.save()
        return rec

    def update(self, shot_id: int, **fields) -> bool:
        for s in self.shots:
            if s.get("id") == shot_id:
                s.update({k: v for k, v in fields.items()
                          if k in ("player", "number", "team", "zone", "cell",
                                   "outcome", "gk")})
                self.save(); return True
        return False

    def delete(self, shot_id: int) -> bool:
        n = len(self.shots)
        self.shots = [s for s in self.shots if s.get("id") != shot_id]
        if len(self.shots) != n:
            self.save(); return True
        return False

    def save(self) -> None:
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.shots, indent=1), encoding="utf-8")
            except Exception:
                pass

    # ── aggregation ─────────────────────────────────────────────────────
    def players(self) -> list[dict]:
        seen = {}
        for s in self.shots:
            key = (s["player"], s["team"])
            seen.setdefault(key, {"player": s["player"], "number": s.get("number"),
                                  "team": s["team"], "shots": 0, "goals": 0})
            seen[key]["shots"] += 1
            if s["outcome"] == "GOAL":
                seen[key]["goals"] += 1
        return sorted(seen.values(), key=lambda p: -p["shots"])

    def goalkeepers(self) -> list[dict]:
        seen = {}
        for s in self.shots:
            gk = s.get("gk")
            if not gk:
                continue
            seen.setdefault(gk, {"gk": gk, "faced": 0, "saves": 0})
            seen[gk]["faced"] += 1
            if s["outcome"] == "SAVE":
                seen[gk]["saves"] += 1
        return sorted(seen.values(), key=lambda g: -g["faced"])

    def _grid(self, records: list[dict], hit_outcome: str) -> dict:
        """Per-zone {cell: [hits, total]} + 9-cell summary with %."""
        zones = {z: {c: [0, 0] for c in CELLS} for z in ZONES}
        summary = {c: [0, 0] for c in CELLS}
        off = {o: 0 for o in OFF_TARGET}
        for s in records:
            cell, zone, out = s["cell"], s["zone"], s["outcome"]
            hit = 1 if out == hit_outcome else 0
            if cell in CELLS:
                zones[zone][cell][1] += 1
                zones[zone][cell][0] += hit
                summary[cell][1] += 1
                summary[cell][0] += hit
            elif cell in OFF_TARGET:
                off[cell] += 1
        return {"zones": zones, "summary": summary, "off_target": off}

    def shot_report(self, player: str, team: str) -> dict:
        recs = [s for s in self.shots if s["player"] == player and s["team"] == team]
        g = self._grid(recs, "GOAL")
        total = len(recs)
        goals = sum(1 for s in recs if s["outcome"] == "GOAL")
        # distribution pies
        by_dist = defaultdict(lambda: {"goals": 0, "total": 0})
        outcome_ct = {o: 0 for o in OUTCOMES}
        for s in recs:
            d = _ZONE_DISTANCE.get(s["zone"], "9m")
            by_dist[d]["total"] += 1
            if s["outcome"] == "GOAL":
                by_dist[d]["goals"] += 1
            outcome_ct[s["outcome"]] += 1
        return {
            "title": "Shot Report", "player": player, "team": team,
            "number": next((s.get("number") for s in recs if s.get("number")), None),
            "grid": g, "total_shots": total, "goals": goals,
            "efficiency": round(100 * goals / total, 1) if total else 0.0,
            "by_distance": dict(by_dist), "by_outcome": outcome_ct,
        }

    def save_report(self, gk: str) -> dict:
        recs = [s for s in self.shots if s.get("gk") == gk]
        g = self._grid(recs, "SAVE")
        faced = len(recs)
        saves = sum(1 for s in recs if s["outcome"] == "SAVE")
        # 9-cell save % summary
        summ = {}
        for c in CELLS:
            h, t = g["summary"][c]
            summ[c] = {"saves": h, "total": t,
                       "pct": round(100 * h / t, 1) if t else 0.0}
        return {
            "title": "Goalkeeper Save Report", "gk": gk,
            "grid": g, "summary_pct": summ, "faced": faced, "saves": saves,
            "save_pct": round(100 * saves / faced, 1) if faced else 0.0,
        }

    def team_summary(self) -> dict:
        out = {o: 0 for o in OUTCOMES}
        per_zone = {z: {"goals": 0, "total": 0} for z in ZONES}
        for s in self.shots:
            out[s["outcome"]] += 1
            per_zone[s["zone"]]["total"] += 1
            if s["outcome"] == "GOAL":
                per_zone[s["zone"]]["goals"] += 1
        return {"by_outcome": out, "by_zone": per_zone, "total": len(self.shots)}
