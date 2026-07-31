"""
ConfigManager — hot-reloadable system configuration.

Loads all YAML config files from config/ and merges them into a single
in-memory dict.  A background thread polls file modification times every
two seconds; when a file changes it reloads that section and sets a
`changed` flag that the pipeline polls once per frame.

Config files
────────────
  config/camera.yaml   — camera connection (ip, user, password, port)
  config/system.yaml   — AI thresholds, detection, display settings
  config/teams.yaml    — team names, colors
  config/rules.yaml    — handball rules tuning

Hot-reloadable parameters (polled by pipeline each frame)
──────────────────────────────────────────────────────────
  detection.confidence          → Detector.conf
  detection.ball_confidence     → Detector.ball_conf
  detection.imgsz               → re-used on next Detector init
  player_db.min_confidence      → PlayerDatabase._recognizer._min_conf
  player_db.lock_frames         → PlayerDatabase._recognizer (cache TTL)
  player_db.hysteresis          → unlock threshold
  player_db.auto_enroll_frames  → when to auto-create a player slot
  coach.passive_play_frames     → HandballCoach passive-play detector
  coach.fatigue_sprint_threshold→ substitution hint threshold
  display.scale                 → frame rescale factor
  display.fps_target            → waitKey timing

Usage
─────
    cfg = ConfigManager(ROOT / "config")

    # read a value (dot-path)
    conf = cfg.get("detection.confidence", default=0.25)
    team_color = cfg.get("teams.team_a.color_bgr", default=[60,60,220])

    # update a value and write back to disk
    cfg.set("detection.confidence", 0.30, save=True)

    # bulk-update a section
    cfg.update_section("teams", {"team_a": {"name": "Lions"}})

    # pipeline: poll once per frame
    if cfg.consume_change():
        _apply_hot_reload(cfg, detector, player_db, coach)

    # snapshot for the API
    safe = cfg.snapshot(redact_passwords=True)
"""

import copy
import threading
import time
from pathlib import Path
from typing import Any

import yaml


# ── Default values written to disk on first run ───────────────────────────────

_SYSTEM_DEFAULTS = {
    "detection": {
        "confidence":       0.25,
        "ball_confidence":  0.07,
        "imgsz":            640,
        "min_crops":        30,
    },
    "player_db": {
        "min_confidence":       0.50,
        "lock_frames":          150,
        "hysteresis":           0.12,
        "auto_enroll_frames":   45,
        "cross_session_thresh": 0.62,
    },
    "coach": {
        "passive_play_frames":          125,
        "fatigue_sprint_threshold":     22,
        "fatigue_distance_threshold_m": 5500.0,
        "fast_break_dist_m":            8.0,
        "period_duration_s":            1800,
    },
    "display": {
        "scale":              1.0,
        "fps_target":         30,
        "show_ghost_tracks":  True,
        "sidebar_enabled":    True,
        "hud_enabled":        True,
    },
    "stamina": {
        "sprint_speed_ms":        5.5,
        "sprint_depletion_per_s": 0.80,
        "run_depletion_per_s":    0.10,
        "recovery_per_s":         0.25,
        "floor_pct":              10.0,
    },
}

_TEAMS_DEFAULTS = {
    "team_a": {
        "name":           "Home",
        "short_name":     "HME",
        "color_bgr":      [60, 60, 220],
        "attacks_right":  True,
    },
    "team_b": {
        "name":           "Away",
        "short_name":     "AWY",
        "color_bgr":      [220, 80, 0],
        "attacks_right":  False,
    },
}

_RULES_DEFAULTS = {
    "period_duration_s":   1800,
    "passive_play_frames": 125,
    "fast_break_dist_m":   8.0,
    "seven_meter_radius":  7.0,
    "six_meter_radius":    6.0,
}

# Maps config file stems to their default content
_DEFAULTS: dict[str, dict] = {
    "system": _SYSTEM_DEFAULTS,
    "teams":  _TEAMS_DEFAULTS,
    "rules":  _RULES_DEFAULTS,
}

POLL_INTERVAL = 2.0   # seconds between mtime checks


class ConfigManager:
    """
    Thread-safe, hot-reloading configuration store.
    """

    def __init__(self, config_dir: str | Path):
        self._dir   = Path(config_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._data:  dict[str, Any]   = {}   # merged flat config by section
        self._mtimes: dict[str, float] = {}  # last-seen mtime per file
        self._lock   = threading.RLock()
        self._changed = threading.Event()

        self._ensure_defaults()
        self._load_all()
        self._start_watcher()

    # ── Public read interface ─────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dot-separated key lookup: "detection.confidence", "teams.team_a.name".
        Returns `default` if any part of the path is missing.
        """
        with self._lock:
            parts = key.split(".")
            node  = self._data
            for p in parts:
                if not isinstance(node, dict) or p not in node:
                    return default
                node = node[p]
            return copy.deepcopy(node)

    def section(self, name: str) -> dict:
        """Return a deep copy of an entire config section."""
        with self._lock:
            return copy.deepcopy(self._data.get(name, {}))

    def snapshot(self, redact_passwords: bool = True) -> dict:
        """Full config snapshot safe for API serialisation."""
        with self._lock:
            snap = copy.deepcopy(self._data)
        if redact_passwords and "camera" in snap:
            pwd = snap["camera"].get("password", "")
            snap["camera"]["password"] = "*" * len(pwd) if pwd else ""
        return snap

    def consume_change(self) -> bool:
        """Returns True (and clears the flag) if config changed since last call."""
        if self._changed.is_set():
            self._changed.clear()
            return True
        return False

    # ── Public write interface ────────────────────────────────────────────────

    def set(self, key: str, value: Any, save: bool = True) -> None:
        """
        Set a dot-path key.  If save=True, flush the owning file to disk.
        e.g. cfg.set("detection.confidence", 0.30, save=True)
        """
        parts = key.split(".")
        if not parts:
            return
        section = parts[0]
        with self._lock:
            node = self._data.setdefault(section, {})
            for p in parts[1:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        if save:
            self._save_section(section)
        self._changed.set()

    def update_section(
        self,
        section:  str,
        updates:  dict,
        save:     bool = True,
    ) -> None:
        """
        Deep-merge `updates` into a config section.
        e.g. cfg.update_section("teams", {"team_a": {"name": "Lions"}})
        """
        with self._lock:
            target = self._data.setdefault(section, {})
            _deep_merge(target, updates)
        if save:
            self._save_section(section)
        self._changed.set()

    def reload(self) -> None:
        """Force re-read of all config files from disk."""
        self._load_all()
        self._changed.set()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _ensure_defaults(self) -> None:
        """Write default YAML files for any section that doesn't exist yet."""
        for stem, defaults in _DEFAULTS.items():
            path = self._dir / f"{stem}.yaml"
            if not path.exists():
                path.write_text(
                    yaml.dump(defaults, default_flow_style=False, allow_unicode=True),
                    encoding="utf-8")
                print(f"[Config] Created default {path.name}")

    def _load_all(self) -> None:
        for path in sorted(self._dir.glob("*.yaml")):
            self._load_file(path)

    def _load_file(self, path: Path) -> None:
        try:
            mtime  = path.stat().st_mtime
            raw    = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            section = path.stem          # "camera", "system", "teams", "rules"
            with self._lock:
                self._data[section] = raw
                self._mtimes[str(path)] = mtime
        except Exception as e:
            print(f"[Config] Failed to load {path.name}: {e}")

    def _save_section(self, section: str) -> None:
        path = self._dir / f"{section}.yaml"
        try:
            with self._lock:
                content = copy.deepcopy(self._data.get(section, {}))
            path.write_text(
                yaml.dump(content, default_flow_style=False, allow_unicode=True),
                encoding="utf-8")
            with self._lock:
                self._mtimes[str(path)] = path.stat().st_mtime
        except Exception as e:
            print(f"[Config] Failed to save {section}.yaml: {e}")

    # ── File watcher ──────────────────────────────────────────────────────────

    def _start_watcher(self) -> None:
        t = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="ConfigWatcher",
        )
        t.start()

    def _watch_loop(self) -> None:
        while True:
            time.sleep(POLL_INTERVAL)
            try:
                for path in self._dir.glob("*.yaml"):
                    key = str(path)
                    try:
                        mtime = path.stat().st_mtime
                    except OSError:
                        continue
                    if mtime != self._mtimes.get(key, 0):
                        self._load_file(path)
                        self._changed.set()
                        print(f"[Config] Hot-reloaded {path.name}")
            except Exception:
                pass


# ── Pipeline integration helper ───────────────────────────────────────────────

def apply_hot_reload(cfg: ConfigManager, **components) -> None:
    """
    Apply changed config values to live pipeline components.
    Call this from the pipeline when cfg.consume_change() returns True.

    Keyword components:
      detector    — Detector instance
      player_db   — PlayerDatabase instance
      coach       — HandballCoach instance
      profiler    — PlayerProfiler instance (updates stamina constants)

    Example:
        if cfg.consume_change():
            apply_hot_reload(cfg,
                             detector=detector,
                             player_db=player_db,
                             coach=coach)
    """
    detector  = components.get("detector")
    player_db = components.get("player_db")
    coach     = components.get("coach")
    profiler  = components.get("profiler")

    # ── Detection thresholds ─────────────────────────────────────────────────
    if detector is not None:
        conf           = cfg.get("detection.confidence")
        ball_conf      = cfg.get("detection.ball_confidence")
        small_conf     = cfg.get("detection.small_conf")
        large_conf     = cfg.get("detection.large_conf")
        small_area_px  = cfg.get("detection.small_area_px")
        if conf          is not None: detector.conf          = float(conf)
        if ball_conf     is not None: detector.ball_conf     = float(ball_conf)
        if small_conf    is not None: detector.small_conf    = float(small_conf)
        if large_conf    is not None: detector.large_conf    = float(large_conf)
        if small_area_px is not None: detector.small_area_px = int(small_area_px)

    # ── Player recognition thresholds ────────────────────────────────────────
    if player_db is not None:
        min_conf  = cfg.get("player_db.min_confidence")
        lock_f    = cfg.get("player_db.lock_frames")
        hysteresis = cfg.get("player_db.hysteresis")
        auto_enroll = cfg.get("player_db.auto_enroll_frames")

        if min_conf   is not None and hasattr(player_db, "_recognizer"):
            player_db._recognizer._min_conf  = float(min_conf)
        if lock_f     is not None:
            # Update the module-level constant proxy
            import player_db as _pdb_mod
            _pdb_mod.LOCK_FRAMES    = int(lock_f)
        if hysteresis is not None:
            import player_db as _pdb_mod
            _pdb_mod.HYSTERESIS     = float(hysteresis)
        if auto_enroll is not None:
            import player_db as _pdb_mod
            _pdb_mod.AUTO_ENROLL_FRAMES = int(auto_enroll)

    # ── Coach AI thresholds ──────────────────────────────────────────────────
    if coach is not None:
        passive  = cfg.get("coach.passive_play_frames")
        fat_spr  = cfg.get("coach.fatigue_sprint_threshold")
        fat_dist = cfg.get("coach.fatigue_distance_threshold_m")
        if passive   is not None and hasattr(coach, "_passive_frames"):
            coach._passive_frames           = int(passive)
        if fat_spr   is not None and hasattr(coach, "_fatigue_sprint_thresh"):
            coach._fatigue_sprint_thresh    = int(fat_spr)
        if fat_dist  is not None and hasattr(coach, "_fatigue_dist_thresh"):
            coach._fatigue_dist_thresh      = float(fat_dist)

    print("[Config] Hot-reload applied.")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge `override` into `base` (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
