"""
MatchLogger — thread-safe event recorder.

Sources of truth
????????????????
  • Vision pipeline (auto)  : goals, shots, possession changes, position
                              snapshots, technique events, etc.
  • Dashboard      (manual) : injuries, fouls, player substitutions,
                              time-outs, manual goals.

Hot-path guarantee
??????????????????
log_event() is O(1) — it just locks, appends to an in-memory list, and sets
a "dirty" flag.  A background daemon thread flushes the JSON file at most
every FLUSH_INTERVAL_S seconds, so the 30 FPS vision loop never pays disk
I/O cost.

Atomic write: tmp + os.replace, so a process kill mid-flush never corrupts
match_log.json.

Restart resilience: events are restored from disk on construction, so a
crash mid-match doesn't lose history.

Usage
?????
    logger = MatchLogger()                    # autosave on
    logger.log_goal("team_a", scorer="Player 7")
    logger.log_manual("injury", team="team_b", player="Player 4")
    recent = logger.get_recent(window_s=300)  # last 5 min
    logger.stop()                             # final flush
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional


PROJECT_ROOT     = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "match_log.json"

# Disk flushes are debounced to this cadence — keeps log_event() free from I/O.
FLUSH_INTERVAL_S = 2.0

# FIFO cap: drop oldest events past this count so the list and the JSON file
# can't grow unbounded over a long match.
DEFAULT_MAX_EVENTS = 50_000


class MatchLogger:
    """Thread-safe append-only event log with debounced JSON persistence."""

    def __init__(
        self,
        path:        Path = DEFAULT_LOG_PATH,
        max_events:  int  = DEFAULT_MAX_EVENTS,
        autosave:    bool = True,
    ):
        self._path        = Path(path)
        self._max_events  = int(max_events)
        self._lock        = threading.RLock()
        self._events:     list[dict] = []
        self._dirty       = False
        self._stop        = threading.Event()
        self._writer:     Optional[threading.Thread] = None

        # Per-track throttle for high-frequency position events.  Stored in
        # memory only; not persisted.  Format: {track_id: last_ts}
        self._pos_last_ts: dict[int, float] = {}

        self._load()

        if autosave:
            self._writer = threading.Thread(
                target=self._flush_loop,
                name="MatchLogger-writer",
                daemon=True,
            )
            self._writer.start()

    # ?? public logging API ??????????????????????????????????????????????????

    def log_event(self, event_type: str, source: str = "auto", **data) -> dict:
        """Generic logger — used by everything else under the hood."""
        ev = {
            "ts":     round(time.time(), 3),
            "type":   event_type,
            "source": source,
            **data,
        }
        with self._lock:
            self._events.append(ev)
            # FIFO eviction so memory + file stay bounded
            if len(self._events) > self._max_events:
                drop = len(self._events) - self._max_events
                del self._events[:drop]
            self._dirty = True
        return ev

    # ?? auto events (called by the vision pipeline) ?????????????????????????

    def log_goal(self, team: str, scorer: Optional[str] = None,
                 **extras) -> dict:
        return self.log_event("goal", source="auto",
                              team=team, scorer=scorer, **extras)

    def log_shot(self, team: str, on_target: Optional[bool] = None,
                 **extras) -> dict:
        return self.log_event("shot", source="auto",
                              team=team, on_target=on_target, **extras)

    def log_possession(self, team: str, duration_s: float = 0.0,
                       **extras) -> dict:
        return self.log_event("possession_change", source="auto",
                              team=team, duration_s=round(duration_s, 1),
                              **extras)

    def log_position(self, track_id: int, x: float, y: float,
                     team: str = "", min_interval_s: float = 1.0,
                     **extras) -> Optional[dict]:
        """
        Logs a player position.  Throttled per-track to one event every
        `min_interval_s` seconds — a 30 FPS pipeline that calls this every
        frame for every player would otherwise generate ~14×30 = 420 events
        per second and balloon the JSON file.

        Returns the event dict if logged, None if throttled.
        """
        now = time.time()
        with self._lock:
            last = self._pos_last_ts.get(track_id, 0.0)
            if (now - last) < float(min_interval_s):
                return None
            self._pos_last_ts[track_id] = now
        return self.log_event("position", source="auto",
                              track_id=track_id, team=team,
                              x=round(float(x), 2), y=round(float(y), 2),
                              **extras)

    # ?? manual events (called by the dashboard) ?????????????????????????????

    def log_manual(self, event_type: str, **data) -> dict:
        """Dashboard-pushed events: injury, foul, substitution, time-out…"""
        return self.log_event(event_type, source="manual", **data)

    # ?? reads ???????????????????????????????????????????????????????????????

    def get_recent(self, window_s: float = 300.0) -> list[dict]:
        """Events that landed in the last `window_s` seconds (wall clock)."""
        cutoff = time.time() - float(window_s)
        with self._lock:
            return [e for e in self._events if e.get("ts", 0) >= cutoff]

    def get_all(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # ?? lifecycle ???????????????????????????????????????????????????????????

    def clear(self) -> None:
        """Wipe the log — call at the start of a fresh match."""
        with self._lock:
            self._events.clear()
            self._pos_last_ts.clear()
            self._dirty = True

    def save(self) -> None:
        """Atomic write to disk.  Safe to call from any thread."""
        with self._lock:
            data = list(self._events)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as exc:
            print(f"[MatchLogger] Save failed: {exc}")

    def stop(self) -> None:
        """Stop the background flusher and write a final snapshot."""
        self._stop.set()
        if self._writer is not None and self._writer.is_alive():
            self._writer.join(timeout=2.0)
        self.save()

    # ?? internals ???????????????????????????????????????????????????????????

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                with self._lock:
                    self._events = data[-self._max_events:]
                print(f"[MatchLogger] Loaded {len(self._events)} events "
                      f"from {self._path.name}")
        except Exception as exc:
            print(f"[MatchLogger] Load failed (starting fresh): {exc}")

    def _flush_loop(self) -> None:
        while not self._stop.wait(FLUSH_INTERVAL_S):
            with self._lock:
                dirty = self._dirty
                self._dirty = False
            if dirty:
                self.save()


# ?? ad-hoc smoke test ????????????????????????????????????????????????????????

if __name__ == "__main__":
    log = MatchLogger()
    log.log_goal("team_a", scorer="Player 7")
    log.log_shot("team_b", on_target=False, distance_m=9.5)
    log.log_manual("injury", team="team_a", player="Player 4",
                   note="left ankle, off the court")
    log.log_position(7, 12.3, 8.7, team="team_a")
    log.log_position(7, 12.4, 8.8, team="team_a")  # throttled
    print(f"len={len(log)}  recent={len(log.get_recent(60))}")
    log.stop()
    print(f"Wrote {log._path}")