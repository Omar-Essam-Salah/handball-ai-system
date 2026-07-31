"""
MatchStateManager — the "Dynamic Brain" of the Human-in-the-Loop system.

Thread-safe, file-persisted entity mapping:
  track_id  →  {name, team, role, shirt_number, locked}

Also manages:
  - Team display names and custom colors
  - Court zone definitions (manual override for auto-calibration)
  - Editable event log
  - Match metadata

Persistence: atomic JSON write to  data/match_state.json
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
STATE_PATH = DATA_DIR / "match_state.json"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EntityInfo:
    track_id:     int
    name:         str   = ""           # e.g. "Nikola Karabatic"
    team:         str   = "unknown"    # "team_a" | "team_b" | "goalkeeper"
    role:         str   = ""           # "GK" | "LW" | "RW" | "CB" | "PV" …
    shirt_number: int   = 0
    locked:       bool  = False        # True → color-classifier cannot override
    updated_at:   float = field(default_factory=time.perf_counter)


@dataclass
class TeamConfig:
    key:        str
    name:       str  = "Team"
    short_name: str  = ""
    color_bgr:  list = field(default_factory=lambda: [130, 130, 130])


@dataclass
class ZoneDefinition:
    name:  str
    x1:    float
    y1:    float
    x2:    float
    y2:    float
    label: str = ""


# ── Manager ───────────────────────────────────────────────────────────────────

class MatchStateManager:
    """
    Central shared-state object.  One instance per process, passed to both
    the pipeline loop and the FastAPI server.

    All mutating methods are protected by RLock — safe to call simultaneously
    from the 30-FPS vision thread, API handler threads, and Streamlit threads.
    """

    def __init__(self, state_path: Path = STATE_PATH):
        self._lock  = threading.RLock()
        self._path  = state_path

        # ── Entity map: track_id → EntityInfo ────────────────────────────────
        self._entities: dict[int, EntityInfo] = {}

        # ── Team config ───────────────────────────────────────────────────────
        self._teams: dict[str, TeamConfig] = {
            "team_a":     TeamConfig("team_a",     "Home",   "HM", [60, 60, 220]),
            "team_b":     TeamConfig("team_b",     "Away",   "AW", [220, 80, 0]),
            "goalkeeper": TeamConfig("goalkeeper", "GK",     "GK", [0, 200, 200]),
            "unknown":    TeamConfig("unknown",    "Unknown","?",  [130, 130, 130]),
        }

        # ── Court zones ───────────────────────────────────────────────────────
        self._zones: dict[str, ZoneDefinition] = {}

        # ── Match metadata ────────────────────────────────────────────────────
        self._meta: dict = {
            "match_id":   "",
            "date":       "",
            "venue":      "",
            "competition": "",
        }

        # ── Editable event log ────────────────────────────────────────────────
        self._corrected_events: list[dict] = []

        # Load persisted state if present
        self._load()

    # ── Entity API ────────────────────────────────────────────────────────────

    def get_entity(self, track_id: int) -> Optional[EntityInfo]:
        with self._lock:
            return self._entities.get(track_id)

    def set_entity(
        self,
        track_id:     int,
        name:         str  = "",
        team:         str  = "",
        role:         str  = "",
        shirt_number: int  = 0,
        locked:       bool = True,
    ) -> EntityInfo:
        """
        Map a track_id to a player identity.
        locked=True prevents the auto-color-classifier from changing the team.
        Any blank field is left unchanged on an existing entity.
        """
        with self._lock:
            existing = self._entities.get(track_id)
            if existing:
                if name:                   existing.name         = name
                if team:                   existing.team         = team
                if role:                   existing.role         = role
                if shirt_number:           existing.shirt_number = shirt_number
                existing.locked     = locked
                existing.updated_at = time.perf_counter()
                entity = existing
            else:
                entity = EntityInfo(
                    track_id=track_id, name=name, team=team or "unknown",
                    role=role, shirt_number=shirt_number, locked=locked,
                )
                self._entities[track_id] = entity
            self._save()
            return entity

    def delete_entity(self, track_id: int) -> bool:
        with self._lock:
            if track_id in self._entities:
                del self._entities[track_id]
                self._save()
                return True
            return False

    def resolve_team(self, track_id: int, ai_team: str) -> str:
        """
        Returns effective team for a track.
        If entity is locked (manually assigned), AI classification is ignored.
        Called by pipeline before rendering — no I/O, must be fast.
        """
        with self._lock:
            e = self._entities.get(track_id)
            if e and e.locked and e.team:
                return e.team
            return ai_team

    def get_label(self, track_id: int, fallback: str = "") -> str:
        """Display label for a track — player name if known, else fallback."""
        with self._lock:
            e = self._entities.get(track_id)
            if e and e.name:
                num = f"#{e.shirt_number} " if e.shirt_number else ""
                return f"{num}{e.name}"
            return fallback

    def all_entities(self) -> list[EntityInfo]:
        with self._lock:
            return list(self._entities.values())

    def clear_all(self) -> None:
        """Wipe entities, zones, corrected events, and metadata.
        Team config is preserved (names/colors set up before the test session)."""
        with self._lock:
            self._entities.clear()
            self._zones.clear()
            self._corrected_events.clear()
            self._meta = {"match_id": "", "date": "", "venue": "", "competition": ""}
            self._save()

    # ── Team config API ───────────────────────────────────────────────────────

    def get_team(self, key: str) -> TeamConfig:
        with self._lock:
            return self._teams.get(key, TeamConfig(key, key, key[:2].upper()))

    def set_team(
        self,
        key:        str,
        name:       str  = "",
        short_name: str  = "",
        color_bgr:  list = None,
    ) -> None:
        with self._lock:
            if key not in self._teams:
                self._teams[key] = TeamConfig(key)
            t = self._teams[key]
            if name:        t.name       = name
            if short_name:  t.short_name = short_name
            if color_bgr:   t.color_bgr  = color_bgr
            self._save()

    def team_names(self) -> dict[str, str]:
        with self._lock:
            return {k: v.name for k, v in self._teams.items()}

    def team_bgr_colors(self) -> dict[str, tuple]:
        """BGR tuples for pipeline rendering (matches AutoColorDetector interface)."""
        with self._lock:
            return {k: tuple(v.color_bgr) for k, v in self._teams.items()}

    # ── Zone API ──────────────────────────────────────────────────────────────

    def set_zone(self, name: str, x1: float, y1: float,
                 x2: float, y2: float, label: str = "") -> None:
        with self._lock:
            self._zones[name] = ZoneDefinition(name, x1, y1, x2, y2, label)
            self._save()

    def get_zone(self, name: str) -> Optional[ZoneDefinition]:
        with self._lock:
            return self._zones.get(name)

    def delete_zone(self, name: str) -> bool:
        with self._lock:
            if name in self._zones:
                del self._zones[name]
                self._save()
                return True
            return False

    def all_zones(self) -> list[ZoneDefinition]:
        with self._lock:
            return list(self._zones.values())

    # ── Match metadata API ────────────────────────────────────────────────────

    def get_meta(self) -> dict:
        with self._lock:
            return self._meta.copy()

    def set_meta(self, **kwargs) -> None:
        with self._lock:
            self._meta.update(kwargs)
            self._save()

    # ── Corrected / manual events ─────────────────────────────────────────────

    def add_corrected_event(self, event: dict) -> None:
        with self._lock:
            event.setdefault("corrected_at", time.perf_counter())
            self._corrected_events.append(event)
            self._save()

    def update_event(self, index: int, patch: dict) -> bool:
        with self._lock:
            if 0 <= index < len(self._corrected_events):
                self._corrected_events[index].update(patch)
                self._save()
                return True
            return False

    def remove_event(self, index: int) -> bool:
        with self._lock:
            if 0 <= index < len(self._corrected_events):
                self._corrected_events.pop(index)
                self._save()
                return True
            return False

    def all_corrected_events(self) -> list[dict]:
        with self._lock:
            return list(self._corrected_events)

    # ── Full snapshot (for API /state) ────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "entities": {str(k): asdict(v) for k, v in self._entities.items()},
                "teams":    {k: asdict(v) for k, v in self._teams.items()},
                "zones":    {k: asdict(v) for k, v in self._zones.items()},
                "meta":     self._meta.copy(),
                "corrected_events": list(self._corrected_events),
            }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Atomic write — called inside the lock."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.snapshot(), indent=2, default=str),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as exc:
            print(f"[StateManager] Save error: {exc}")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))

            for tid_s, d in raw.get("entities", {}).items():
                d.pop("updated_at", None)   # perf_counter values don't survive restart
                self._entities[int(tid_s)] = EntityInfo(**d,
                    updated_at=time.perf_counter())

            for key, d in raw.get("teams", {}).items():
                self._teams[key] = TeamConfig(**d)

            for name, d in raw.get("zones", {}).items():
                self._zones[name] = ZoneDefinition(**d)

            self._meta.update(raw.get("meta", {}))
            self._corrected_events = raw.get("corrected_events", [])

            print(f"[StateManager] Loaded {len(self._entities)} entities, "
                  f"{len(self._zones)} zones from {self._path}")
        except Exception as exc:
            print(f"[StateManager] Load failed (starting fresh): {exc}")
