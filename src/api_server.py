"""
FastAPI server — bridges the vision pipeline with the dashboard.

Threading model
───────────────
  • Vision loop (pipeline.py) runs in its own thread at ~30 FPS.
    It calls push_frame() each processed frame — pure in-memory write, no I/O.

  • Uvicorn runs in a separate daemon thread (started by start_server()).
    All WebSocket / HTTP handlers are async coroutines inside that thread.

  • MatchStateManager is shared between both threads via RLock-protected methods.

  • Two threading.Event flags let the dashboard signal back to the pipeline:
      _recalibrate_flag  →  pipeline calls color_det.recalibrate()
      _side_flip_flag    →  pipeline calls rules.flip_sides()

REST endpoints
──────────────
  GET  /state              full MatchStateManager snapshot
  POST /entity             map track_id → player info
  POST /team               rename / recolor a team
  POST /zone               define a court zone
  DELETE /zone/{name}      remove a zone
  POST /event/add          insert a manual event
  POST /event/remove       delete an event by index
  POST /goal               record a manual goal
  POST /recalibrate        schedule color recalibration
  POST /flip-sides         schedule side flip

WebSocket
─────────
  WS /ws  — bidirectional
    Server → Client:  {type:"frame", frame_b64, tracks, hud, ts}
    Client → Server:  {type:"click", x, y}
                      {type:"ping"}
    Server → Client (click reply):
                      {type:"click_result", track_id, entity}
"""

import asyncio
import base64
import json
import math
import threading
import time
from collections import deque
from typing import Any, Optional

import aiohttp

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import shutil
import yaml
from pathlib import Path
from state_manager import MatchStateManager

ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "camera.yaml"
RULES_FILE  = ROOT / "coach_rules.txt" # ← Added for God Mode

# ── Ollama LLM tactical consumer ──────────────────────────────────────────────
_OLLAMA_URL          = "http://localhost:11434/api/generate"
_OLLAMA_MODEL        = "llama3.2"   # must match: ollama pull llama3.2
_OLLAMA_TIMEOUT_S    = 60      # generous — llama3.2 on CPU can take ~40 s
_OLLAMA_INTERVAL_S   = 30      # consumer wake-up cadence (seconds)
_OLLAMA_BACKOFF_S    = 90      # retry delay when Ollama is unreachable
_TACTICAL_WINDOW_S   = 45      # event horizon fed to the LLM (30–60 s range)
_EVENT_RING_MAXLEN   = 2_000   # hard cap — deque auto-evicts oldest entries
_MAX_EVENTS_PER_CALL = 50      # events forwarded per LLM call
_OLLAMA_MAX_TOKENS   = 120     # hard token cap — we only want 1-2 short commands
_MAX_RESPONSE_CHARS  = 400     # character ceiling on stored/broadcast response

# System prompt lives in handball_coach.py (coaching domain knowledge).
# Imported here so api_server has no direct coaching logic.
from handball_coach import SYSTEM_PROMPT as _SYSTEM_PROMPT  # noqa: E402

# ── Shared references (injected by pipeline before server starts) ─────────────
_state:     Optional[MatchStateManager] = None
_rules:     Any = None      # HandballEngine
_color_det: Any = None      # AutoColorDetector
_player_db: Any = None      # PlayerDatabase
_court_det: Any = None      # CourtDetector
_analytics: Any = None      # PlayerAnalytics
_profiler:  Any = None      # PlayerProfiler
_coach:     Any = None      # HandballCoach
_bridge:    Any = None      # TrackBridge
_camera:    Any = None      # HikvisionCamera
_cfg_mgr:   Any = None      # ConfigManager
_logger:    Any = None      # MatchLogger ← Added for God Mode
_analyst:   Any = None      # OllamaAnalyst ← Added for God Mode

# Latest processed frame (deque(maxlen=1) = lock-free mailbox)
_frame_buffer:     deque = deque(maxlen=1)
# Raw camera frame (before HUD/overlays) — used by the calibration widget
_raw_frame_buffer: deque = deque(maxlen=1)

# Pipeline status string — updated by pipeline each phase
_pipeline_status: str = "Starting…"

# Outbound notification queue — pipeline writes, WebSocket loop drains
_notif_queue: deque = deque(maxlen=64)

# ── Ollama consumer state ─────────────────────────────────────────────────────
# Ring buffer of match-event dicts pushed by pipeline / HandballCoach.
# Each entry: {"ts": float, "type": str, "level": str, "team": str, ...}
# deque is GIL-protected for append/popleft — no explicit lock needed for writes.
_event_ring: deque = deque(maxlen=_EVENT_RING_MAXLEN)

# Latest LLM insight (maxlen=1 = lock-free single-slot mailbox).
_tactical_insight: deque = deque(maxlen=1)

# asyncio.Lock — created in _startup() inside the event loop, not at import time.
_ollama_lock: Optional[asyncio.Lock] = None

# Persistent aiohttp session — created in _startup(), closed in _shutdown().
# One session for the process lifetime avoids per-call TCP handshake overhead.
_aiohttp_session: Optional[aiohttp.ClientSession] = None

# Latest court-coordinate positions from CourtMapper (updated by pipeline)
_court_positions_buffer: deque = deque(maxlen=1)

# Signal flags — pipeline polls these each frame
recalibrate_flag    = threading.Event()
side_flip_flag      = threading.Event()
court_calib_flag    = threading.Event()   # pipeline re-runs court auto_calibrate

# ── Technique event ring ─────────────────────────────────────────────────────
_technique_ring: deque = deque(maxlen=200)   # last 200 detected techniques

# ── Auto-zoom tracking state ──────────────────────────────────────────────────
# mode: "off" | "court" | "players" | "ball" | "goal_a" | "goal_b" | "scan"
_autozoom_mode: str             = "off"
_autozoom_task: Optional[Any]   = None   # asyncio.Task

# ── Software (digital) zoom + framing ─────────────────────────────────────────
# Used when the camera has no optical zoom (fixed-lens bullet cams) OR when
# pointing the camera at a TV screen and you want to crop to just the screen.
# Pipeline reads these values via get_digital_zoom() and crops the frame BEFORE
# YOLO inference, so detection only sees the cropped region.
#   zoom : 1.0 = full frame, 2.0 = 2× crop, 4.0 = 4× crop (max)
#   cx,cy: crop centre in 0..1 normalised frame coordinates
_dz_zoom: float = 1.0
_dz_cx:   float = 0.5
_dz_cy:   float = 0.5


def get_digital_zoom() -> tuple[float, float, float]:
    """Pipeline-side accessor — returns (zoom, cx, cy)."""
    return _dz_zoom, _dz_cx, _dz_cy

# ── Smart scan state ──────────────────────────────────────────────────────────
# Divides the court into 6 zones; cycles through them looking for undetected
# players.  After each move, autofocus is triggered and player count recorded.
_SCAN_ZONES: list[dict] = [
    {"name": "wide",   "preset": 1,    "pan":  0.0,  "tilt": 0.0, "zoom": -0.2},
    {"name": "left",   "preset": None, "pan": -0.45, "tilt": 0.0, "zoom":  0.15},
    {"name": "center", "preset": None, "pan":  0.0,  "tilt": 0.0, "zoom":  0.0},
    {"name": "right",  "preset": None, "pan":  0.45, "tilt": 0.0, "zoom":  0.15},
    {"name": "goal_a", "preset": 2,    "pan": -0.7,  "tilt":-0.1, "zoom":  0.3},
    {"name": "goal_b", "preset": 3,    "pan":  0.7,  "tilt":-0.1, "zoom":  0.3},
]
_scan_idx:      int   = 0
_scan_moved_at: float = 0.0   # monotonic timestamp of last zone move
_scan_focused:  bool  = False  # autofocus triggered for current zone?
_scan_dwell_s:  float = 5.0   # seconds to dwell per zone
_scan_results:  dict  = {}    # zone_name → {"players": int, "ts": float}


def init(state: MatchStateManager, rules=None, color_det=None) -> None:
    """Inject shared objects. Call this before start_server()."""
    global _state, _rules, _color_det
    _state     = state
    _rules     = rules
    _color_det = color_det


def _safe_json(obj):
    """JSON encoder that handles numpy scalars and non-finite floats."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0.0 if not math.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def push_match_event(event: dict) -> None:
    """
    Thread-safe.  Called by pipeline or HandballCoach to feed the Ollama ring.
    """
    _event_ring.append({"ts": time.time(), **event})


def push_raw_frame(frame_bgr: np.ndarray) -> None:
    """Store the latest pre-overlay camera frame for the calibration widget."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if ok:
        _raw_frame_buffer.append(base64.b64encode(buf.tobytes()).decode("ascii"))


def push_court_positions(positions: list[dict]) -> None:
    """Called by pipeline after CourtMapper.tracks_to_court_positions()."""
    _court_positions_buffer.append(positions)


def push_technique_event(event: dict) -> None:
    """
    Called by pipeline whenever TechniqueDetector fires a new event.
    Stores in the ring and also feeds the match event ring (for Ollama).
    """
    event.setdefault("ts", round(time.time(), 3))
    event["event"] = f"technique:{event.get('technique','unknown')}"
    _technique_ring.append(event)
    _event_ring.append(event)   # Ollama consumer sees it in its window


def push_frame(frame_bgr: np.ndarray, tracks: list, hud_data: dict) -> None:
    """
    Called by the pipeline once per processed frame.
    Encodes the annotated frame as JPEG and stores it in _frame_buffer.
    """
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
    if not ok:
        return
    try:
        hud_safe = json.loads(json.dumps(hud_data, default=_safe_json))
    except Exception:
        hud_safe = {}
    h, w = frame_bgr.shape[:2]
    _frame_buffer.append({
        "frame_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
        "tracks":    [_track_dict(t) for t in tracks],
        "hud":       hud_safe,
        "ts":        round(time.perf_counter(), 3),
        "fw":        w,
        "fh":        h,
    })


def _track_dict(t) -> dict:
    label = _state.get_label(t.track_id) if _state else ""
    return {
        "id":    t.track_id,
        "team":  t.team,
        "ball":  t.is_ball,
        "label": label,
        "x1":    round(t.x1), "y1": round(t.y1),
        "x2":    round(t.x2), "y2": round(t.y2),
        "conf":  round(t.confidence, 2),
    }


# ── Pydantic request models ───────────────────────────────────────────────────

class EntityUpdate(BaseModel):
    track_id:     int
    name:         str  = ""
    team:         str  = ""
    role:         str  = ""
    shirt_number: int  = 0
    locked:       bool = True

class TeamUpdate(BaseModel):
    key:        str
    name:       str  = ""
    short_name: str  = ""
    color_bgr:  list = []

class ZoneUpdate(BaseModel):
    name:  str
    x1:    float
    y1:    float
    x2:    float
    y2:    float
    label: str = ""

class EventAdd(BaseModel):
    event_type: str
    team:       str
    data:       dict = {}

class EventRemove(BaseModel):
    index: int

class EventPatch(BaseModel):
    index: int
    patch: dict

class ManualGoal(BaseModel):
    team: str

class CameraConfig(BaseModel):
    ip:       str  = ""
    user:     str  = "admin"
    password: str  = ""
    rtsp_port: int = 554
    channel:  int  = 102


# ── Pipeline status helper (called by pipeline.py) ────────────────────────────

def set_status(msg: str) -> None:
    global _pipeline_status
    _pipeline_status = msg


def notify_event(event_type: str, team: str = "", data: dict | None = None) -> None:
    """
    Thread-safe: called by the pipeline to push a real-time notification
    to all connected dashboard WebSocket clients.
    """
    _notif_queue.append({
        "type":  "notification",
        "event": event_type,
        "team":  team,
        "data":  data or {},
        "ts":    round(time.time(), 3),
    })


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Handball AI — Human-in-the-Loop API",
    version="2.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_HEATMAP_INTERVAL_S = 300   # generate heatmaps every 5 minutes


@app.on_event("startup")
async def _startup():
    global _ollama_lock, _aiohttp_session
    _ollama_lock = asyncio.Lock()
    import aiohttp
    _aiohttp_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=_OLLAMA_TIMEOUT_S),
        trust_env=False,
    )
    asyncio.create_task(_heatmap_scheduler())
    # asyncio.create_task(_ollama_consumer())  <--- قم بوضع علامة # هنا لإيقافه
    asyncio.create_task(_autozoom_loop())
    print(f"[API] Ollama consumer DISABLED for performance")


@app.on_event("shutdown")
async def _shutdown():
    if _aiohttp_session and not _aiohttp_session.closed:
        await _aiohttp_session.close()


async def _autozoom_loop():
    _INTERVAL      = 0.20   
    _TARGET_COVER  = 0.68   
    _DEAD_PAN      = 0.04   
    _DEAD_ZOOM     = 0.06   
    _PAN_GAIN      = 1.4    
    _ZOOM_GAIN     = 0.8

    while True:
        await asyncio.sleep(_INTERVAL)

        if _autozoom_mode == "off" or _camera is None:
            continue

        if _autozoom_mode in ("court", "goal_a", "goal_b"):
            preset_map = {"court": 1, "goal_a": 2, "goal_b": 3}
            pid = preset_map[_autozoom_mode]
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, _camera.goto_preset, pid)
            except Exception:
                pass
            continue

        if _autozoom_mode == "scan":
            global _scan_idx, _scan_moved_at, _scan_focused
            now_m = time.monotonic()
            zone  = _SCAN_ZONES[_scan_idx % len(_SCAN_ZONES)]

            if now_m - _scan_moved_at >= _scan_dwell_s:
                _scan_idx   = (_scan_idx + 1) % len(_SCAN_ZONES)
                zone        = _SCAN_ZONES[_scan_idx]
                _scan_moved_at = now_m
                _scan_focused  = False

                try:
                    if zone.get("preset"):
                        await asyncio.get_event_loop().run_in_executor(
                            None, _camera.goto_preset, zone["preset"])
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda z=zone: _camera.ptz_continuous(
                                pan=z["pan"], tilt=z["tilt"],
                                zoom=z["zoom"], speed=0.65,
                            ),
                        )
                        await asyncio.sleep(0.7)
                        await asyncio.get_event_loop().run_in_executor(
                            None, _camera.ptz_stop)
                except Exception:
                    pass

            elif not _scan_focused and now_m - _scan_moved_at >= 1.2:
                _scan_focused = True
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, _camera.focus_auto)
                except Exception:
                    pass

                if _frame_buffer:
                    snap     = _frame_buffer[-1]
                    fw_scan  = snap.get("fw", 1280) or 1280
                    tracks   = snap.get("tracks", [])
                    zone_name = zone["name"]

                    zone_x_ranges = {
                        "wide":   (0.0, 1.0),
                        "left":   (0.0, 0.38),
                        "center": (0.31, 0.69),
                        "right":  (0.62, 1.0),
                        "goal_a": (0.0, 0.22),
                        "goal_b": (0.78, 1.0),
                    }
                    x_lo, x_hi = zone_x_ranges.get(zone_name, (0.0, 1.0))
                    n_players = sum(
                        1 for t in tracks
                        if not t.get("ball")
                        and x_lo <= (t["x1"] + t["x2"]) / 2 / fw_scan < x_hi
                    )
                    _scan_results[zone_name] = {
                        "players": n_players,
                        "ts":      round(time.time(), 1),
                    }
            continue

        if not _frame_buffer:
            continue

        snap  = _frame_buffer[-1]
        fw    = snap.get("fw", 1280)
        fh    = snap.get("fh", 720)
        tracks = snap.get("tracks", [])

        if _autozoom_mode == "ball":
            targets = [t for t in tracks if t.get("ball")]
        else:  
            targets = [t for t in tracks if not t.get("ball")]

        if not targets:
            continue

        x1 = min(t["x1"] for t in targets)
        y1 = min(t["y1"] for t in targets)
        x2 = max(t["x2"] for t in targets)
        y2 = max(t["y2"] for t in targets)

        margin_px = max(fw, fh) * 0.10   
        x1 = max(0,  x1 - margin_px)
        y1 = max(0,  y1 - margin_px)
        x2 = min(fw, x2 + margin_px)
        y2 = min(fh, y2 + margin_px)

        cx_n = ((x1 + x2) / 2) / fw
        cy_n = ((y1 + y2) / 2) / fh

        diag_target = math.hypot(1.0, fh / fw)        
        diag_clust  = math.hypot((x2 - x1) / fw,
                                 (y2 - y1) / fh)
        coverage    = diag_clust / diag_target

        err_x    = cx_n - 0.5
        err_y    = cy_n - 0.5
        err_zoom = coverage - _TARGET_COVER

        pan_cmd  = err_x * _PAN_GAIN  if abs(err_x)    > _DEAD_PAN  else 0.0
        tilt_cmd = err_y * _PAN_GAIN  if abs(err_y)    > _DEAD_PAN  else 0.0
        zoom_cmd = err_zoom * _ZOOM_GAIN if abs(err_zoom) > _DEAD_ZOOM else 0.0

        zoom_cmd = -zoom_cmd

        if pan_cmd == 0.0 and tilt_cmd == 0.0 and zoom_cmd == 0.0:
            await asyncio.get_event_loop().run_in_executor(
                None, _camera.ptz_stop)
        else:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _camera.ptz_continuous(
                    pan=max(-1.0, min(1.0, pan_cmd)),
                    tilt=max(-1.0, min(1.0, -tilt_cmd)),  
                    zoom=max(-1.0, min(1.0, zoom_cmd)),
                    speed=0.55,
                ),
            )


async def _heatmap_scheduler():
    while True:
        await asyncio.sleep(_HEATMAP_INTERVAL_S)
        if _analytics:
            try:
                for p in _analytics.all_profiles():
                    _analytics.update_heatmap(p.player_id)
                await _broadcast_json({"type": "notification",
                                       "event": "heatmaps_updated",
                                       "ts": round(time.time(), 3)})
            except Exception:
                pass


async def _broadcast_json(payload: dict) -> None:
    txt = json.dumps(payload, default=_safe_json)
    dead = []
    async with _ws_mgr._lock:
        targets = list(_ws_mgr._active)
    for ws in targets:
        try:
            await ws.send_text(txt)
        except Exception:
            dead.append(ws)
    for ws in dead:
        await _ws_mgr.disconnect(ws)


# ── Ollama tactical consumer ──────────────────────────────────────────────────

def _gather_match_stats() -> dict:
    stats: dict = {}

    if _rules:
        try:
            now_pc = time.perf_counter()
            cutoff_pc = now_pc - _TACTICAL_WINDOW_S

            stats["period"]          = _rules.period
            stats["score"]           = _rules.score.copy()
            stats["shots"]           = _rules.shots.copy()
            stats["possession"]      = _rules.possession
            stats["poss_duration_s"] = round(_rules.possession_duration, 1)
            stats["shot_accuracy"]   = _rules.shot_accuracy

            rule_evs = []
            for ev in list(_rules.events)[-80:]:
                try:
                    if ev.timestamp >= cutoff_pc:
                        rule_evs.append({
                            "event": ev.event_type.value,
                            "team":  ev.team,
                            "data":  ev.data,
                        })
                except Exception:
                    pass
            if rule_evs:
                stats["rule_events"] = rule_evs
        except Exception:
            pass

    if _state:
        try:
            now_pc = time.perf_counter()
            cutoff_pc = now_pc - _TACTICAL_WINDOW_S
            state_evs = [
                e for e in list(getattr(_state, "corrected_events", []))[-50:]
                if e.get("ts", 0) >= cutoff_pc
            ]
            if state_evs:
                stats["corrected_events"] = state_evs
        except Exception:
            pass

    if _coach:
        try:
            stats["coach_alerts"] = _coach.recent_alerts(12)
        except Exception:
            pass

    return stats


def _build_ollama_prompt(ring_events: list[dict], stats: dict) -> str:
    _j = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False)

    period = stats.get("period", 1)
    score  = stats.get("score",  {"team_a": 0, "team_b": 0})
    shots  = stats.get("shots",  {"team_a": 0, "team_b": 0})
    poss   = stats.get("possession") or "unknown"
    poss_s = stats.get("poss_duration_s", 0)
    acc    = stats.get("shot_accuracy", {"team_a": 0.0, "team_b": 0.0})

    compact_ring = []
    for ev in ring_events[-_MAX_EVENTS_PER_CALL:]:
        row = {k: v for k, v in ev.items()
               if k not in ("ts",) and v not in ("", None, {})}
        compact_ring.append(row)

    sections: list[str] = [
        f"LIVE MATCH DATA — window={_TACTICAL_WINDOW_S}s",
        (f"Period {period} | Score: TeamA {score.get('team_a', 0)}–"
         f"{score.get('team_b', 0)} TeamB | "
         f"Shots A={shots.get('team_a', 0)} B={shots.get('team_b', 0)} | "
         f"Accuracy A={acc.get('team_a', 0):.0%} B={acc.get('team_b', 0):.0%} | "
         f"Possession: {poss} for {poss_s}s"),
    ]

    if stats.get("coach_alerts"):
        sections.append(f"Vision alerts: {_j(stats['coach_alerts'])}")

    if stats.get("rule_events"):
        sections.append(f"Match events: {_j(stats['rule_events'])}")

    if stats.get("corrected_events"):
        sections.append(f"Verified events: {_j(stats['corrected_events'])}")

    if compact_ring:
        sections.append(f"Tactical alerts ({len(compact_ring)}): {_j(compact_ring)}")

    cutoff = time.time() - _TACTICAL_WINDOW_S
    techs  = [e for e in _technique_ring if e.get("ts", 0) >= cutoff]
    if techs:
        compact_techs = [
            {"tech": e.get("technique"), "team": e.get("team"),
             "zone": e.get("zone"), "pid": e.get("player_id") or e.get("track_id"),
             "conf": e.get("confidence")}
            for e in techs[-20:]   
        ]
        sections.append(f"Detected techniques ({len(compact_techs)}): {_j(compact_techs)}")

    sections.append("Output your 1 or 2 commands now. No preamble.")

    return "\n".join(sections)


async def _call_ollama(user_msg: str) -> str:
    """
    Native async POST to Ollama.
    Uses asyncio.to_thread and requests to completely avoid aiohttp session errors.
    """
    import requests
    
    payload = {
        "model":  _OLLAMA_MODEL,
        "system": _SYSTEM_PROMPT,
        "prompt": user_msg,
        "stream": False,
        "options": {
            "num_gpu":        0,      # ← FORCE CPU
            "num_predict":    _OLLAMA_MAX_TOKENS,
            "temperature":    0.20,
            "top_p":          0.90,
            "repeat_penalty": 1.15,
            "stop":           ["\n\n", "Note:", "However", "Based on"],
        },
    }

    try:
        # Run the synchronous requests.post in a background thread
        # This completely eliminates aiohttp Timeout and NoneType session errors!
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.post(_OLLAMA_URL, json=payload, timeout=_OLLAMA_TIMEOUT_S)
        )
        resp.raise_for_status()
        data = resp.json()
        
        text = data.get("response", "").strip()
        return text[:_MAX_RESPONSE_CHARS]

    except Exception as exc:
        print(f"[Ollama] Inference Failed: {exc}")
        return ""


async def _ollama_consumer() -> None:
    _down_until: float = 0.0

    while True:
        await asyncio.sleep(_OLLAMA_INTERVAL_S)

        if time.monotonic() < _down_until:
            continue

        if _ollama_lock is None or _ollama_lock.locked():
            continue

        async with _ollama_lock:
            try:
                now    = time.time()
                cutoff = now - _TACTICAL_WINDOW_S

                ring_events = [e for e in _event_ring
                               if e.get("ts", 0.0) >= cutoff]

                if not ring_events and _rules is None and _coach is None:
                    continue

                stats    = _gather_match_stats()
                user_msg = _build_ollama_prompt(ring_events, stats)

                response = await _call_ollama(user_msg)

                if not response:
                    continue

                insight = {
                    "text":     response,
                    "ts":       round(now, 3),
                    "n_events": len(ring_events),
                    "model":    _OLLAMA_MODEL,
                }
                _tactical_insight.append(insight)

                await _broadcast_json({
                    "type":    "tactical_insight",
                    "insight": response,
                    "ts":      insight["ts"],
                })

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print(f"[Ollama] Unexpected error: {type(exc).__name__}: {exc}")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/state")
def get_state():
    return _state.snapshot() if _state else {}


@app.get("/health")
def health():
    return {
        "ok":      True,
        "clients": _ws_mgr.count,
        "frame":   bool(_frame_buffer),
        "status":  _pipeline_status,
    }


@app.get("/status")
def get_status():
    return {"status": _pipeline_status, "has_frame": bool(_frame_buffer)}


# ── Admin: full reset for testing mode ────────────────────────────────────────
class AdminClearRequest(BaseModel):
    players:      bool = True
    analytics:    bool = True
    profiler:     bool = True
    match_state:  bool = True
    rules:        bool = True
    detection_log: bool = True


def _wipe_dir_contents(d: Path) -> int:
    if not d.exists():
        return 0
    n = 0
    for p in d.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                n += 1
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                n += 1
        except Exception:
            pass
    return n


@app.post("/admin/clear-all")
def admin_clear_all(req: AdminClearRequest = AdminClearRequest()):
    cleared: dict[str, int | bool] = {}

    if req.match_state and _state is not None:
        _state.clear_all()
        cleared["match_state"] = True

    if req.rules and _rules is not None:
        _rules.reset()
        cleared["rules"] = True

    if req.players and _player_db is not None:
        try:
            _player_db._cache.clear()
            _player_db._obs.clear()
        except AttributeError:
            pass
        for attr in ("_profiles", "profiles", "_players"):
            store = getattr(_player_db, attr, None)
            if isinstance(store, dict):
                store.clear()
        cleared["players_files"] = _wipe_dir_contents(ROOT / "data" / "player_db")

    if req.analytics and _analytics is not None:
        for attr in ("_profiles", "profiles"):
            store = getattr(_analytics, attr, None)
            if isinstance(store, dict):
                store.clear()
        cleared["analytics_files"] = _wipe_dir_contents(ROOT / "data" / "analytics")

    if req.profiler and _profiler is not None:
        for attr in ("_profiles", "profiles", "_stamina"):
            store = getattr(_profiler, attr, None)
            if isinstance(store, dict):
                store.clear()
        cleared["profiler_files"] = _wipe_dir_contents(ROOT / "data" / "profiles")

    if req.detection_log:
        log_path = ROOT / "detection_log.jsonl"
        if log_path.exists():
            try:
                log_path.write_text("", encoding="utf-8")
                cleared["detection_log"] = True
            except Exception:
                cleared["detection_log"] = False

    try:
        _technique_ring.clear()
        _event_ring.clear()
    except Exception:
        pass

    return {"ok": True, "cleared": cleared}


@app.get("/camera")
def get_camera():
    if CONFIG_PATH.exists():
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    else:
        cfg = {}
    pwd = cfg.get("password", "")
    return {
        "ip":        cfg.get("ip", ""),
        "user":      cfg.get("user", "admin"),
        "password":  ("*" * len(pwd)) if pwd else "",
        "rtsp_port": cfg.get("rtsp_port", 554),
        "channel":   cfg.get("channel", 102),
    }


@app.post("/camera")
def update_camera(req: CameraConfig):
    if CONFIG_PATH.exists():
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    else:
        cfg = {}
    if req.ip:        cfg["ip"]        = req.ip
    if req.user:      cfg["user"]      = req.user
    if req.password and not req.password.startswith("*"):
        cfg["password"]  = req.password
    cfg["rtsp_port"] = req.rtsp_port
    cfg["channel"]   = req.channel
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.dump(cfg), encoding="utf-8")
    recalibrate_flag.set()   
    return {"ok": True, "ip": cfg.get("ip")}


@app.post("/entity")
def update_entity(req: EntityUpdate):
    if not _state:
        return {"ok": False, "error": "state not initialised"}
    e = _state.set_entity(
        track_id=req.track_id, name=req.name, team=req.team,
        role=req.role, shirt_number=req.shirt_number, locked=req.locked,
    )
    return {"ok": True, "entity": {
        "track_id": e.track_id, "name": e.name,
        "team": e.team, "locked": e.locked,
    }}


@app.delete("/entity/{track_id}")
def delete_entity(track_id: int):
    if not _state:
        return {"ok": False}
    ok = _state.delete_entity(track_id)
    return {"ok": ok}


@app.post("/team")
def update_team(req: TeamUpdate):
    if not _state:
        return {"ok": False}
    _state.set_team(
        req.key,
        name=req.name,
        short_name=req.short_name,
        color_bgr=req.color_bgr or None,
    )
    return {"ok": True}


@app.post("/zone")
def define_zone(req: ZoneUpdate):
    if not _state:
        return {"ok": False}
    _state.set_zone(req.name, req.x1, req.y1, req.x2, req.y2, req.label)
    return {"ok": True}


@app.delete("/zone/{name}")
def remove_zone(name: str):
    if not _state:
        return {"ok": False}
    ok = _state.delete_zone(name)
    return {"ok": ok}


@app.post("/event/add")
def add_event(req: EventAdd):
    if not _state:
        return {"ok": False}
    ev = {
        "event": req.event_type, "team": req.team,
        "ts": time.perf_counter(), "frame": -1,
        **req.data,
    }
    _state.add_corrected_event(ev)
    if _rules and req.event_type == "goal":
        _rules.manual_goal(req.team)
    return {"ok": True}


@app.post("/event/remove")
def remove_event(req: EventRemove):
    if not _state:
        return {"ok": False}
    return {"ok": _state.remove_event(req.index)}


@app.post("/event/patch")
def patch_event(req: EventPatch):
    if not _state:
        return {"ok": False}
    return {"ok": _state.update_event(req.index, req.patch)}


@app.post("/goal")
def manual_goal(req: ManualGoal):
    if not _rules:
        return {"ok": False, "error": "rules engine not ready yet"}
    _rules.manual_goal(req.team)
    if _state:
        _state.add_corrected_event({
            "event": "goal", "team": req.team,
            "ts": time.perf_counter(), "frame": -1, "manual": True,
        })
    return {"ok": True, "score": _rules.score.copy()}


@app.post("/recalibrate")
def recalibrate():
    recalibrate_flag.set()
    return {"ok": True, "message": "recalibration scheduled for next frame"}


@app.post("/flip-sides")
def flip_sides():
    side_flip_flag.set()
    return {"ok": True, "message": "side-flip scheduled for next frame"}


@app.get("/players")
def get_players():
    if not _player_db:
        return {"ok": False, "players": []}
    profiles = []
    for p in _player_db.all_profiles():
        profiles.append({
            "player_id":    p.player_id,
            "name":         p.name,
            "shirt_number": p.shirt_number,
            "team":         p.team,
            "role":         p.role,
            "sample_count": p.sample_count,
            "has_thumb":    bool(p.thumb_jpg),
        })
    return {"ok": True, "players": profiles, "stats": _player_db.stats()}


@app.get("/players/{player_id}/thumb")
def get_player_thumb(player_id: str):
    if not _player_db:
        return {"ok": False}
    p = _player_db.get(player_id)
    if not p or not p.thumb_jpg:
        return {"ok": False, "thumb_b64": ""}
    return {"ok": True,
            "thumb_b64": base64.b64encode(p.thumb_jpg).decode()}


class PlayerAdd(BaseModel):
    name:         str
    team:         str
    shirt_number: int  = 0
    role:         str  = ""

@app.post("/players")
def add_player(req: PlayerAdd):
    if not _player_db:
        return {"ok": False, "error": "player_db not ready"}
    pid = _player_db.get_or_create(req.name, req.team, req.shirt_number, req.role)
    return {"ok": True, "player_id": pid}


@app.delete("/players/{player_id}")
def delete_player(player_id: str):
    if not _player_db:
        return {"ok": False}
    return {"ok": _player_db.delete_player(player_id)}


@app.post("/players/{player_id}/rename")
def rename_player(player_id: str, body: dict = {}):
    if not _player_db:
        return {"ok": False}
    name         = body.get("name", "").strip()
    shirt_number = int(body.get("shirt_number", 0))
    ok = _player_db.rename_player(player_id, name, shirt_number)
    if ok and _state:
        for entity in _state.all_entities():
            cached = _player_db._cache.get(entity.track_id)
            if cached and cached[0] == player_id:
                _state.set_entity(
                    entity.track_id,
                    name=name,
                    shirt_number=shirt_number if shirt_number else entity.shirt_number,
                )
    return {"ok": ok}


@app.get("/match-stats")
def match_stats():
    if not _rules:
        return {"ok": False, "error": "rules engine not ready"}

    events_out = []
    for ev in _rules.events:
        events_out.append({
            "event":   ev.event_type.value,
            "team":    ev.team,
            "ts":      round(ev.timestamp, 3),
            "frame":   ev.frame_n,
            "data":    ev.data,
        })

    period_elapsed = 0.0
    if _rules._period_start_ts:
        import time as _time
        period_elapsed = _time.perf_counter() - _rules._period_start_ts

    return {
        "ok":           True,
        "period":       _rules.period,
        "period_elapsed_s": round(period_elapsed, 0),
        "score":        _rules.score.copy(),
        "shots":        _rules.shots.copy(),
        "saves":        _rules.saves.copy(),
        "fast_breaks":  _rules.fast_breaks.copy(),
        "shot_pct":     _rules.shot_accuracy,
        "events":       events_out,
        "possession":   _rules.possession,
        "possession_duration_s": round(_rules.possession_duration, 1),
    }


@app.get("/analytics/player/{player_id}")
def analytics_player(player_id: str):
    if not _analytics:
        return {"ok": False}
    d = _analytics.profile_dict(player_id)
    if not d:
        return {"ok": False, "error": "not found"}
    return {"ok": True, **d}


@app.get("/analytics/all")
def analytics_all():
    if not _analytics:
        return {"ok": False, "players": []}
    return {"ok": True,
            "players": [_analytics.profile_dict(p.player_id)
                        for p in _analytics.all_profiles()]}


@app.get("/analytics/player/{player_id}/heatmap")
def analytics_heatmap(player_id: str):
    if not _analytics:
        return {"ok": False}
    import base64
    jpg = _analytics.heatmap_png(player_id)
    if jpg is None:
        return {"ok": False}
    return {"ok": True, "heatmap_b64": base64.b64encode(jpg).decode()}


@app.get("/profiles")
def get_all_profiles():
    if not _profiler:
        if not _analytics:
            return {"ok": False, "players": []}
        return {
            "ok":      True,
            "players": [_analytics.profile_dict(p.player_id)
                        for p in _analytics.all_profiles()],
        }
    return {"ok": True, "players": _profiler.all_full_profiles()}


@app.get("/profiles/{player_id}")
def get_full_profile(player_id: str):
    if not _profiler:
        if not _analytics:
            return {"ok": False, "error": "profiler not ready"}
        d = _analytics.profile_dict(player_id)
        return {"ok": bool(d), **d} if d else {"ok": False, "error": "player not found"}

    d = _profiler.full_profile(player_id)
    if not d:
        return {"ok": False, "error": "player not found"}
    d["heatmap_url"] = f"/analytics/player/{player_id}/heatmap"
    return {"ok": True, **d}


@app.get("/profiles/{player_id}/report")
def get_player_report(player_id: str):
    if not _profiler:
        return {"ok": False, "error": "profiler not ready"}
    r = _profiler.generate_report(player_id)
    if r is None:
        return {"ok": False, "error": "insufficient data — player visible < 3 seconds"}
    return {"ok": True, "player_id": player_id, "report": r}


@app.post("/profiles/{player_id}/report/refresh")
def refresh_player_report(player_id: str):
    if not _profiler:
        return {"ok": False, "error": "profiler not ready"}
    r = _profiler.generate_report(player_id)
    if r is None:
        return {"ok": False, "error": "insufficient data"}
    return {"ok": True, "player_id": player_id, "report": r}


@app.get("/profiles/{player_id}/stamina")
def get_player_stamina(player_id: str):
    if not _profiler:
        return {"ok": False, "error": "profiler not ready"}
    d = _profiler.get_stamina(player_id)
    return {"ok": True, "player_id": player_id, **d}


@app.get("/profiles/{player_id}/sprints")
def get_player_sprints(player_id: str):
    if not _profiler:
        return {"ok": False, "error": "profiler not ready"}
    events = _profiler.get_sprints(player_id)
    return {
        "ok":      True,
        "player_id": player_id,
        "count":   len(events),
        "sprints": events,
    }


@app.get("/players/compare")
def compare_players(ids: str = ""):
    if not ids:
        return {"ok": False, "error": "provide ?ids=pid1,pid2,..."}

    player_ids = [p.strip() for p in ids.split(",") if p.strip()]
    source     = _profiler if _profiler else None
    rows       = []

    for pid in player_ids:
        if source:
            d = source.full_profile(pid)
        elif _analytics:
            d = _analytics.profile_dict(pid)
        else:
            d = {}
        if not d:
            continue
        rows.append({
            "player_id":    pid,
            "name":         d.get("name", pid),
            "team":         d.get("team", ""),
            "role":         d.get("role") or d.get("inferred_role", ""),
            "grade":        d.get("grade", "—"),
            "performance":  d.get("performance", 0),
            "distance_km":  d.get("distance_km", 0),
            "sprint_count": d.get("sprint_count", 0),
            "max_speed_ms": d.get("max_speed_ms", 0),
            "goals":        d.get("actions", {}).get("goals", 0),
            "stamina_pct":  d.get("stamina", {}).get("current", 100),
            "stamina_status": d.get("stamina", {}).get("status", "fresh"),
        })

    rows.sort(key=lambda r: r["performance"], reverse=True)
    return {"ok": True, "players": rows}


@app.get("/bridge/status")
def bridge_status():
    if not _bridge:
        return {"ok": False}
    active   = _bridge.all_active_slots()
    ghosts   = _bridge.invisible_ghosts()
    ghost_out = []
    for g in ghosts:
        entity = _state.get_entity(g.slot_id) if _state else None
        ghost_out.append({
            "slot_id":      g.slot_id,
            "name":         entity.name if entity else "",
            "shirt_number": entity.shirt_number if entity else 0,
            "team":         g.team,
            "invisible_frames": g.invisible,
            "last_seen":    g.last_seen,
            "predicted_x":  round(g.positions[-1][0] + g.vx * g.invisible, 1) if g.positions else 0,
            "predicted_y":  round(g.positions[-1][1] + g.vy * g.invisible, 1) if g.positions else 0,
        })
    return {
        "ok":           True,
        "active_count": len(active),
        "ghost_count":  len(ghosts),
        "ghosts":       ghost_out,
    }


@app.get("/coach/alerts")
def coach_alerts():
    if not _coach:
        return {"ok": False, "alerts": []}
    return {"ok": True, "alerts": _coach.recent_alerts(20)}


@app.get("/coach/tactical")
def coach_tactical():
    if not _tactical_insight:
        return {
            "ok":      False,
            "insight": None,
            "message": (
                "No tactical analysis available yet.  "
                f"Ensure Ollama is running at {_OLLAMA_URL} "
                f"with model '{_OLLAMA_MODEL}'.  "
                "The consumer retries every "
                f"{_OLLAMA_INTERVAL_S}s."
            ),
        }
    latest = _tactical_insight[-1]
    return {"ok": True, **latest}


@app.get("/llm/insight")
def llm_insight():
    return coach_tactical()


@app.get("/coach/tactical/events")
def coach_tactical_events():
    cutoff = time.time() - _TACTICAL_WINDOW_S
    events = [e for e in _event_ring if e.get("ts", 0.0) >= cutoff]
    return {
        "ok":         True,
        "window_s":   _TACTICAL_WINDOW_S,
        "total_ring": len(_event_ring),
        "in_window":  len(events),
        "events":     events[-50:],   
    }


@app.get("/techniques/recent")
def techniques_recent(n: int = 30, window_s: float = 60.0):
    cutoff = time.time() - window_s
    recent = [e for e in _technique_ring if e.get("ts", 0) >= cutoff]
    return {"ok": True, "total": len(recent), "events": recent[-n:]}


@app.get("/techniques/summary")
def techniques_summary(window_s: float = 120.0):
    cutoff = time.time() - window_s
    recent = [e for e in _technique_ring if e.get("ts", 0) >= cutoff]
    by_tech: dict[str, int] = {}
    by_team: dict[str, dict[str, int]] = {}
    for e in recent:
        t = e.get("technique", "unknown")
        team = e.get("team", "unknown")
        by_tech[t] = by_tech.get(t, 0) + 1
        by_team.setdefault(team, {})[t] = by_team[team].get(t, 0) + 1
    return {"ok": True, "window_s": window_s,
            "by_technique": by_tech, "by_team": by_team,
            "total": len(recent)}


@app.get("/coach/rules")
def coach_rules():
    from handball_coach import HandballCoach
    return {"ok": True, "rules": HandballCoach.all_rules()}


class CoachAsk(BaseModel):
    question: str
    context:  dict = {}

@app.post("/coach/ask")
def coach_ask(req: CoachAsk):
    if not req.question.strip():
        return {"ok": False, "error": "empty question"}

    if _coach and hasattr(_coach, "answer"):
        try:
            answer = _coach.answer(req.question, context=req.context)
            return {"ok": True, "answer": answer}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    ms_data = {}
    if _rules:
        ms_data = {
            "score":       _rules.score,
            "period":      _rules.period,
            "possession":  _rules.possession,
            "fast_breaks": _rules.fast_breaks,
        }
    fallback = (
        f"[Offline coach] Match context: {ms_data}. "
        "Tactical suggestion: maintain defensive shape and exploit fast-break opportunities."
    )
    return {"ok": True, "answer": fallback}

# ── GOD MODE API ──────────────────────────────────────────────────────

@app.post("/match/manual-event")
def handle_manual_event(req: dict):
    if _logger:
        _logger.log_manual(req.get("event_type", "manual"), team=req.get("team"), player=req.get("player"))
    return {"ok": True}

@app.get("/coach/rules_text")
def get_rules_text():
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            return {"rules": f.read()}
    except Exception:
        return {"rules": "You are a handball tactical analyst..."}

@app.post("/coach/rules_text")
def update_rules_text(req: dict):
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        f.write(req.get("rules", ""))
    return {"ok": True}

@app.post("/analyst/request")
def force_analyst():
    if _analyst:
        _analyst.request_now()
    return {"ok": True}

# ──────────────────────────────────────────────────────────────────────────────

@app.get("/heatmap/live")
def heatmap_live():
    if not _analytics:
        return {"ok": False, "error": "analytics not ready"}
    import base64
    jpg = _analytics.combined_heatmap_jpg()
    if jpg is None:
        return {"ok": False, "error": "not enough data yet"}
    return {"ok": True, "heatmap_b64": base64.b64encode(jpg).decode()}


@app.get("/heatmap/player/{player_id}")
def heatmap_player_fresh(player_id: str):
    if not _analytics:
        return {"ok": False}
    import base64
    if hasattr(_analytics, "update_heatmap"):
        _analytics.update_heatmap(player_id)
    jpg = _analytics.heatmap_png(player_id)
    if jpg is None:
        return {"ok": False, "error": "no data"}
    return {"ok": True, "heatmap_b64": base64.b64encode(jpg).decode()}


@app.get("/court/positions")
def court_positions():
    return {"ok": True, "positions": list(_court_positions_buffer)}


@app.get("/snapshot")
def get_snapshot():
    if _raw_frame_buffer:
        return {"ok": True, "frame_b64": _raw_frame_buffer[-1]}
    if _frame_buffer:
        return {"ok": True, "frame_b64": _frame_buffer[-1]["frame_b64"]}
    return {"ok": False, "error": "no frame yet — is the pipeline running?"}


@app.get("/court/status")
def court_status():
    if _court_det is None:
        return {"calibrated": False, "error": "court detector not ready"}
    return {
        "calibrated": _court_det.is_calibrated,
        "homography": _court_det._iH.tolist() if _court_det._iH is not None else None,
    }


@app.post("/court/calibrate")
def court_calibrate(body: dict = {}):
    if _court_det is None:
        return {"ok": False, "error": "court detector not ready"}

    if body.get("clear"):
        _court_det.clear_calibration()
        return {"ok": True, "method": "clear"}

    corners = body.get("corners")
    if corners and len(corners) == 4:
        ok = _court_det.set_manual_corners(corners)
        if ok and _rules is not None:
            try:
                _rules.court_transformer.load_calibration()
            except Exception:
                pass
        return {"ok": ok, "method": "manual",
                "error": None if ok else "homography computation failed"}

    court_calib_flag.set()
    return {"ok": True, "method": "auto_pending"}


# ── Camera control — Pydantic models ─────────────────────────────────────────

class PTZMove(BaseModel):
    pan:   float = 0.0    
    tilt:  float = 0.0    
    zoom:  float = 0.0    
    speed: float = 0.5    

class PTZMomentary(BaseModel):
    pan:         float = 0.0
    tilt:        float = 0.0
    zoom:        float = 0.0
    focus:       float = 0.0   
    duration_ms: int   = 500

class PTZPresetSave(BaseModel):
    preset_id: int
    name:      str = ""

class FocusManual(BaseModel):
    direction:   str   = "near"   
    speed:       float = 0.5
    duration_ms: int   = 600

class ImageParams(BaseModel):
    brightness:  int | None = None   
    contrast:    int | None = None
    saturation:  int | None = None
    sharpness:   int | None = None

class OSDText(BaseModel):
    text:    str  = ""
    enabled: bool = True


# ── Camera control endpoints ──────────────────────────────────────────────────

def _cam_guard():
    if _camera is None:
        return None, {"ok": False, "error": "camera not initialised"}
    return _camera, None


@app.get("/camera/info")
def camera_info():
    cam, err = _cam_guard()
    if err:
        return err
    return {"ok": True, **cam.device_info()}


@app.get("/camera/ping")
def camera_ping():
    cam, err = _cam_guard()
    if err:
        return err
    alive = cam.ping()
    return {"ok": True, "reachable": alive,
            "ip": cam._ip if alive else None}


@app.get("/camera/ptz/status")
def ptz_status():
    cam, err = _cam_guard()
    if err:
        return err
    s = cam.get_status()
    return {
        "ok":    not bool(s.error),
        "pan":   s.pan,
        "tilt":  s.tilt,
        "zoom":  s.zoom,
        "error": s.error or None,
    }


@app.post("/camera/ptz/move")
def ptz_move(req: PTZMove):
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.ptz_continuous(req.pan, req.tilt, req.zoom, req.speed)
    return {"ok": ok, "error": cam.last_error if not ok else None}


@app.post("/camera/ptz/momentary")
def ptz_momentary(req: PTZMomentary):
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.ptz_momentary(req.pan, req.tilt, req.zoom,
                           req.focus, req.duration_ms)
    return {"ok": ok, "error": cam.last_error if not ok else None}


@app.post("/camera/ptz/stop")
def ptz_stop():
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.ptz_stop()
    return {"ok": ok}


class ZoomReq(BaseModel):
    direction: str = "in"   
    step:      float = 0.25 


@app.post("/camera/zoom")
def camera_zoom(req: ZoomReq):
    global _dz_zoom, _dz_cx, _dz_cy

    if req.direction == "reset":
        _dz_zoom, _dz_cx, _dz_cy = 1.0, 0.5, 0.5
    elif req.direction == "in":
        _dz_zoom = min(4.0, _dz_zoom + req.step)
    elif req.direction == "out":
        _dz_zoom = max(1.0, _dz_zoom - req.step)

    if _camera is not None and req.direction != "reset":
        try:
            z = req.step if req.direction == "in" else -req.step
            if _camera.zoom_continuous(z, speed=1.0):
                threading.Timer(0.5, _camera.ptz_stop).start()
        except Exception:
            pass

    return {
        "ok":   True,
        "zoom": round(_dz_zoom, 2),
        "cx":   round(_dz_cx, 3),
        "cy":   round(_dz_cy, 3),
    }


class FrameOffsetReq(BaseModel):
    direction: str = "right"  
    step:      float = 0.05


@app.post("/camera/frame/offset")
def camera_frame_offset(req: FrameOffsetReq):
    global _dz_cx, _dz_cy
    if req.direction == "left":     _dz_cx = max(0.0, _dz_cx - req.step)
    elif req.direction == "right":  _dz_cx = min(1.0, _dz_cx + req.step)
    elif req.direction == "up":     _dz_cy = max(0.0, _dz_cy - req.step)
    elif req.direction == "down":   _dz_cy = min(1.0, _dz_cy + req.step)
    elif req.direction == "centre": _dz_cx, _dz_cy = 0.5, 0.5
    return {"ok": True, "zoom": round(_dz_zoom, 2),
            "cx": round(_dz_cx, 3), "cy": round(_dz_cy, 3)}


@app.get("/camera/frame/state")
def camera_frame_state():
    return {"ok": True, "zoom": round(_dz_zoom, 2),
            "cx": round(_dz_cx, 3), "cy": round(_dz_cy, 3)}


@app.get("/camera/ptz/presets")
def ptz_presets():
    cam, err = _cam_guard()
    if err:
        return err
    presets = cam.get_presets()
    return {"ok": True,
            "presets": [{"id": p.id, "name": p.name} for p in presets]}


@app.post("/camera/ptz/preset/goto")
def ptz_goto_preset(body: dict = {}):
    cam, err = _cam_guard()
    if err:
        return err
    pid = int(body.get("preset_id", 1))
    ok  = cam.goto_preset(pid)
    return {"ok": ok, "error": cam.last_error if not ok else None}


@app.post("/camera/ptz/preset/save")
def ptz_save_preset(req: PTZPresetSave):
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.save_preset(req.preset_id, req.name)
    return {"ok": ok, "error": cam.last_error if not ok else None}


@app.delete("/camera/ptz/preset/{preset_id}")
def ptz_delete_preset(preset_id: int):
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.delete_preset(preset_id)
    return {"ok": ok}


@app.post("/camera/focus/auto")
def focus_auto():
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.focus_auto()
    return {"ok": ok, "error": cam.last_error if not ok else None}


@app.post("/camera/focus/manual")
def focus_manual(req: FocusManual):
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.focus_manual(req.direction, req.speed, req.duration_ms)
    return {"ok": ok}


@app.post("/camera/focus/stop")
def focus_stop():
    cam, err = _cam_guard()
    if err:
        return err
    return {"ok": cam.focus_stop()}


@app.get("/camera/image")
def get_image_params():
    cam, err = _cam_guard()
    if err:
        return err
    return {"ok": True, **cam.get_image_params()}


@app.post("/camera/image")
def set_image_params(req: ImageParams):
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.set_image_params(
        req.brightness, req.contrast, req.saturation, req.sharpness)
    return {"ok": ok, "error": cam.last_error if not ok else None}


@app.get("/camera/daynight")
def get_day_night():
    cam, err = _cam_guard()
    if err:
        return err
    mode = cam.get_day_night_mode()
    return {"ok": True, "mode": mode}


@app.post("/camera/daynight")
def set_day_night(body: dict = {}):
    cam, err = _cam_guard()
    if err:
        return err
    mode = body.get("mode", "day").lower()
    ok   = cam.set_day_night_mode(mode)
    return {"ok": ok, "mode": mode,
            "error": cam.last_error if not ok else None}


@app.post("/camera/osd")
def set_osd(req: OSDText):
    cam, err = _cam_guard()
    if err:
        return err
    ok = cam.set_osd_text(req.text, req.enabled)
    return {"ok": ok}


@app.get("/camera/stream/info")
def stream_info():
    cam, err = _cam_guard()
    if err:
        return err
    return {"ok": True, **cam.get_stream_info()}


# ── Auto-zoom endpoints ───────────────────────────────────────────────────────

@app.get("/camera/autozoom")
def get_autozoom():
    current_zone = _SCAN_ZONES[_scan_idx % len(_SCAN_ZONES)]["name"] if _SCAN_ZONES else "none"
    scan_zones_out = [
        {"name": z["name"], **_scan_results.get(z["name"], {"players": -1, "ts": None})}
        for z in _SCAN_ZONES
    ]
    return {
        "ok":           True,
        "mode":         _autozoom_mode,
        "cam_ready":    _camera is not None,
        "scan_zone":    current_zone if _autozoom_mode == "scan" else None,
        "scan_results": scan_zones_out,
    }


@app.post("/camera/autozoom")
async def set_autozoom(body: dict = {}):
    global _autozoom_mode
    mode = body.get("mode", "off").lower()
    valid = ("off", "court", "players", "ball", "goal_a", "goal_b", "scan")
    if mode not in valid:
        return {"ok": False, "error": f"mode must be one of {valid}"}

    _autozoom_mode = mode

    if mode == "off" and _camera:
        await asyncio.get_event_loop().run_in_executor(
            None, _camera.ptz_stop)

    return {"ok": True, "mode": _autozoom_mode}


@app.post("/camera/analyze")
async def camera_analyze():
    if _camera:
        await asyncio.get_event_loop().run_in_executor(
            None, _camera.goto_preset, 1)
        await asyncio.sleep(1.5)   

    snap: dict = {}
    if _state:
        snap = _state.snapshot() if hasattr(_state, "snapshot") else {}
    if _rules:
        snap["score"]       = getattr(_rules, "score", {})
        snap["shots"]       = getattr(_rules, "shots", {})
        snap["fast_breaks"] = getattr(_rules, "fast_breaks", {})
        snap["shot_pct"]    = getattr(_rules, "shot_accuracy", {})

    if _frame_buffer:
        f = _frame_buffer[-1]
        snap["active_players"] = len([t for t in f.get("tracks", [])
                                      if not t.get("ball")])

    if _coach:
        recent = []
        try:
            recent = _coach.recent_alerts(5)
        except Exception:
            pass
        snap["recent_alerts"] = recent

    if not _aiohttp_session or _aiohttp_session.closed:
        return {"ok": False, "error": "Ollama session not ready"}

    prompt = (
        "Analyze this handball match snapshot and give 2 sharp tactical commands:\n"
        + json.dumps(snap, default=_safe_json)
    )
    try:
        async with _aiohttp_session.post(
            _OLLAMA_URL,
            json={"model": _OLLAMA_MODEL, "prompt": prompt,
                  "stream": False, "num_predict": 120},
        ) as resp:
            data    = await resp.json(content_type=None)
            insight = (data.get("response") or "").strip()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    return {"ok": True, "insight": insight, "snapshot": snap}


# ── Config management endpoints ───────────────────────────────────────────────

@app.get("/config")
def get_config():
    if not _cfg_mgr:
        return {"ok": False, "error": "config manager not ready"}
    return {"ok": True, "config": _cfg_mgr.snapshot(redact_passwords=True)}


@app.get("/config/{section}")
def get_config_section(section: str):
    if not _cfg_mgr:
        return {"ok": False, "error": "config manager not ready"}
    data = _cfg_mgr.section(section)
    if not data:
        return {"ok": False, "error": f"section '{section}' not found"}
    if section == "camera":
        pwd = data.get("password", "")
        data["password"] = "*" * len(pwd)
    return {"ok": True, "section": section, "data": data}


@app.post("/config/{section}")
def update_config_section(section: str, body: dict = {}):
    if not _cfg_mgr:
        return {"ok": False, "error": "config manager not ready"}

    if section == "camera" and "password" in body:
        if body["password"].startswith("*"):
            body.pop("password")

    _cfg_mgr.update_section(section, body, save=True)
    return {"ok": True, "section": section, "applied": body}


@app.post("/config/reload")
def force_config_reload():
    if not _cfg_mgr:
        return {"ok": False, "error": "config manager not ready"}
    _cfg_mgr.reload()
    return {"ok": True, "message": "config reloaded from disk"}


@app.patch("/config/set")
def set_config_key(body: dict = {}):
    if not _cfg_mgr:
        return {"ok": False, "error": "config manager not ready"}
    key   = body.get("key", "").strip()
    value = body.get("value")
    if not key or value is None:
        return {"ok": False, "error": "provide key and value"}
    _cfg_mgr.set(key, value, save=True)
    return {"ok": True, "key": key, "value": value}


# ── Report generation endpoint ────────────────────────────────────────────────

@app.post("/report/generate")
def report_generate(body: dict = {}):
    try:
        from report_generator import generate_report
    except ImportError:
        return {"ok": False, "error": "report_generator module not found"}

    phase    = body.get("phase", "fulltime")
    filename = body.get("filename", "")

    try:
        path = generate_report(
            phase=phase,
            state=_state,
            rules=_rules,
            analytics=_analytics,
            filename=filename,
        )
        return {"ok": True, "path": str(path)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── WebSocket connection manager ──────────────────────────────────────────────

class _WSManager:
    def __init__(self):
        self._active: list[WebSocket] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._get_lock():
            self._active.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._get_lock():
            if ws in self._active:
                self._active.remove(ws)

    @property
    def count(self) -> int:
        return len(self._active)


_ws_mgr = _WSManager()


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await _ws_mgr.connect(ws)
    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.008)
                try:
                    await _handle_ws_cmd(json.loads(raw), ws)
                except Exception:
                    pass   
            except asyncio.TimeoutError:
                pass
            except (WebSocketDisconnect, RuntimeError):
                break      

            while _notif_queue:
                try:
                    notif = _notif_queue[0]   
                    await ws.send_text(json.dumps(notif, default=_safe_json))
                    _notif_queue.popleft()
                except Exception:
                    break

            if _frame_buffer:
                try:
                    await ws.send_text(
                        json.dumps({"type": "frame", **_frame_buffer[-1]})
                    )
                except Exception:
                    break  

            await asyncio.sleep(1 / 25)

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await _ws_mgr.disconnect(ws)


async def _handle_ws_cmd(msg: dict, ws: WebSocket) -> None:
    cmd = msg.get("type")

    if cmd == "click" and _state and _frame_buffer:
        x, y      = float(msg.get("x", 0)), float(msg.get("y", 0))
        frame_w   = int(msg.get("frame_w", 1280))
        frame_h   = int(msg.get("frame_h", 720))
        tracks    = _frame_buffer[-1].get("tracks", [])

        buf_dummy = _frame_buffer[-1]
        best_id, best_d = None, float("inf")
        for t in tracks:
            if t["ball"]:
                continue
            cx = (t["x1"] + t["x2"]) / 2
            cy = (t["y1"] + t["y2"]) / 2
            d  = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_id = t["id"]

        entity = _state.get_entity(best_id) if best_id is not None else None
        await ws.send_text(json.dumps({
            "type":     "click_result",
            "track_id": best_id,
            "dist_px":  round(best_d, 1),
            "entity": {
                "name":         entity.name         if entity else "",
                "team":         entity.team         if entity else "",
                "role":         entity.role         if entity else "",
                "shirt_number": entity.shirt_number if entity else 0,
                "locked":       entity.locked       if entity else False,
            } if best_id is not None else None,
        }))

    elif cmd == "ping":
        await ws.send_text(json.dumps({"type": "pong"}))


# ── Server launcher ───────────────────────────────────────────────────────────

_server_instance: Optional["uvicorn.Server"] = None


def start_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    global _server_instance
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        loop="asyncio",
        ws_ping_interval=None,
        ws_ping_timeout=None,
        ws_max_size=16 * 1024 * 1024,
    )
    server = uvicorn.Server(config)
    _server_instance = server

    t = threading.Thread(
        target=server.run, daemon=True, name="APIServer"
    )
    t.start()
    print(f"[API] Server  : http://{host}:{port}/docs")
    print(f"[API] WebSocket: ws://{host}:{port}/ws")


def stop_server() -> None:
    global _server_instance
    if _server_instance is not None:
        _server_instance.should_exit = True
        _server_instance = None
        print("[API] Server shutdown requested.")