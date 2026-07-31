"""
Phase 2 Pipeline — FAST MODE + Async Analytics + Debug Mode + OCR Identity.
Stable version with OCR integration for robust player tracking.
"""

import argparse
import dataclasses
import multiprocessing as mp
import queue as _queue
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from capture            import CaptureThread
from detector           import Detector
from tracker            import Tracker
from player_reid        import PlayerReID
from multi_session_reid import MultiSessionReID
from team_color         import AutoColorDetector
from handball_rules     import HandballEngine, EventType
from state_manager      import MatchStateManager
from court_detector     import CourtDetector
from handball_coach     import HandballCoach
from match_analyzer     import MatchAnalyzer  
from static_marker_filter import StaticMarkerFilter
from goal_replay          import GoalReplayRecorder
from goal_post_detector   import GoalPostDetector
from court_mask           import CourtMask
from motion_ball          import MotionBall
from biomech              import BiomechAnalyzer
from spatial              import HeatMap, TeamShape, ShotMap
from event_verifier       import EventVerifier
from match_identity        import MatchIdentity

# استيراد الـ OCR مع حماية في حالة عدم التفعيل
try:
    from jersey_ocr import JerseyOCR
except ImportError:
    JerseyOCR = None

ROOT        = Path(__file__).parent.parent
HOMOG_PATH  = ROOT / "config" / "homography_matrix.npy"
LOGS_DIR    = ROOT / "logs"
MSR_PATH    = ROOT / "data" / "multi_session_reid.json"
REPORTS_DIR = ROOT / "reports"

BAR_H     = 90
SIDEBAR_W = 240
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_B    = cv2.FONT_HERSHEY_DUPLEX

BALL_CLASS_ID = 32

_VISUAL_COLORS = {
    "team_a":     (255, 80,  80),
    "team_b":     (50,  180, 255),
    "goalkeeper": (50,  255, 50),
    "unknown":    (200, 200, 200),
}

_FP_HS_BINS    = (24, 24)
_FP_HS_RANGES  = [0, 180, 0, 256]
_FP_CLAHE      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

def _color_fingerprint(frame: np.ndarray, t):
    H_frame, W_frame = frame.shape[:2]
    kpts = getattr(t, 'keypoints', None)
    tx1 = ty1 = tx2 = ty2 = 0
    valid_crop = False

    if kpts is not None and len(kpts) >= 13:
        pts_x, pts_y = [], []
        for idx in (5, 6, 11, 12):
            if kpts[idx][2] > 0.40:
                pts_x.append(kpts[idx][0]); pts_y.append(kpts[idx][1])
        if len(pts_x) >= 3:
            tx1, tx2 = int(min(pts_x)), int(max(pts_x))
            ty1, ty2 = int(min(pts_y)), int(max(pts_y))
            margin_x = int((tx2 - tx1) * 0.10)
            tx1, tx2 = tx1 + margin_x, tx2 - margin_x
            extra_y = int((ty2 - ty1) * 0.10)
            ty1 = max(0, ty1 - extra_y)
            valid_crop = True

    if not valid_crop:
        bw, bh = t.x2 - t.x1, t.y2 - t.y1
        if bh <= 6: return None
        tx1, tx2 = int(t.x1 + bw * 0.30), int(t.x2 - bw * 0.30)
        ty1, ty2 = int(t.y1 + bh * 0.18), int(t.y1 + bh * 0.50)

    tx1, tx2 = max(0, min(tx1, W_frame - 1)), max(0, min(tx2, W_frame))
    ty1, ty2 = max(0, min(ty1, H_frame - 1)), max(0, min(ty2, H_frame))
    if tx2 - tx1 < 4 or ty2 - ty1 < 4: return None

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0: return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = _FP_CLAHE.apply(hsv[:, :, 2])
    sat_mask = (hsv[:, :, 1] >= 30).astype(np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1], sat_mask, _FP_HS_BINS, _FP_HS_RANGES)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    feat_hs = hist.flatten().astype(np.float32)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    if len(pixels) >= 16:
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 2, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten(), minlength=2)
            dom = centers[int(np.argmax(counts))]
            anchor = np.array([dom[0] / 255.0, dom[1] / 255.0, dom[2] / 255.0], dtype=np.float32)
        except Exception:
            anchor = np.zeros(3, dtype=np.float32)
    else:
        anchor = np.zeros(3, dtype=np.float32)

    return np.concatenate([feat_hs, anchor]).astype(np.float32)


def background_analytics_worker(in_queue, homography_path, snapshot_path,
                                snapshot_every_s=10.0, heatmap_w=96, heatmap_h=54,
                                max_jump_px=250.0, speed_ema_alpha=0.3):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    H = np.load(homography_path).astype(np.float64) if homography_path and Path(homography_path).exists() else None
    def to_world(x, y):
        if H is None: return x, y
        v = H @ np.array([x, y, 1.0])
        return (v[0]/v[2], v[1]/v[2]) if v[2] != 0 else (x, y)
    snap = Path(snapshot_path); snap.parent.mkdir(parents=True, exist_ok=True)
    last_snapshot_t, received = time.perf_counter(), 0
    while True:
        try: payload = in_queue.get(timeout=1.0)
        except _queue.Empty:
            if (time.perf_counter() - last_snapshot_t) >= snapshot_every_s:
                last_snapshot_t = time.perf_counter()
            continue
        if payload == "__STOP__": break
        received += 1

def _draw_hud(frame, hud, rules, color_det, team_names, frame_n, state):
    h, w = frame.shape[:2]
    out  = frame.copy()
    ca, cb = _VISUAL_COLORS["team_a"], _VISUAL_COLORS["team_b"]
    name_a, name_b = team_names.get("team_a", "Team A"), team_names.get("team_b", "Team B")

    cv2.rectangle(out, (0, 0), (w, BAR_H), (20, 20, 20), -1)
    cv2.putText(out, name_a, (12, 28), FONT, 0.75, ca, 2)
    cv2.putText(out, str(hud["score"]["team_a"]), (12, 75), FONT_B, 1.8, ca, 3)

    rb_x = w - SIDEBAR_W - 10
    (tw, _), _ = cv2.getTextSize(name_b, FONT, 0.75, 2)
    cv2.putText(out, name_b, (rb_x - tw, 28), FONT, 0.75, cb, 2)
    (tw2, _), _ = cv2.getTextSize(str(hud["score"]["team_b"]), FONT_B, 1.8, 3)
    cv2.putText(out, str(hud["score"]["team_b"]), (rb_x - tw2, 75), FONT_B, 1.8, cb, 3)

    mid_x = (w - SIDEBAR_W) // 2
    cv2.putText(out, "-", (mid_x - 10, 72), FONT_B, 1.4, (200, 200, 200), 2)
    pm, ps = divmod(int(hud["period_elapsed_s"]), 60)
    cv2.putText(out, f"P{hud['period']}  {pm:02d}:{ps:02d}", (mid_x - 45, 26), FONT, 0.65, (200, 200, 200), 1)

    poss = hud["possession"]
    poss_color = ca if poss == "team_a" else cb if poss == "team_b" else (160, 160, 160)
    poss_label = name_a if poss == "team_a" else name_b if poss == "team_b" else "-"
    cv2.putText(out, f"BALL: {poss_label}  {hud['possession_duration_s']:.0f}s",
                (mid_x - 60, 60), FONT, 0.6, poss_color, 2)

    sx = w - SIDEBAR_W
    cv2.rectangle(out, (sx, 0), (w, h), (18, 18, 18), -1)
    cv2.line(out, (sx, 0), (sx, h), (50, 50, 50), 1)

    y = BAR_H + 20
    def sbar(label, val, color=(200, 200, 200)):
        nonlocal y
        cv2.putText(out, label, (sx + 8, y), FONT, 0.5, (120, 120, 120), 1)
        cv2.putText(out, val, (sx + 8, y + 18), FONT, 0.65, color, 2)
        y += 42
    sbar("SHOTS", f"{name_a}: {hud['shots']['team_a']}   {name_b}: {hud['shots']['team_b']}")
    sbar("FAST BREAKS", f"{name_a}: {hud['fast_breaks']['team_a']}   {name_b}: {hud['fast_breaks']['team_b']}")

    cv2.line(out, (sx + 5, y), (w - 5, y), (50, 50, 50), 1); y += 12
    cv2.putText(out, "EVENTS", (sx + 8, y), FONT, 0.5, (100, 200, 255), 1); y += 16
    for line in rules.recent_events_text(7):
        col = ca if "TA" in line else cb if "TB" in line else (200, 200, 200)
        cv2.putText(out, line, (sx + 6, y), FONT, 0.45, col, 1); y += 18

    cv2.putText(out, f"FPS: {hud.get('fps', 0):.1f}", (sx + 8, h - 8), FONT, 0.5, (80, 80, 80), 1)
    return out


def _draw_debug_panel(out, debug_info: dict):
    h, w = out.shape[:2]
    bar_top, bar_h = h - 28, 28
    cv2.rectangle(out, (0, bar_top), (w - SIDEBAR_W, h), (12, 12, 12), -1)
    cv2.line(out, (0, bar_top), (w - SIDEBAR_W, bar_top), (60, 60, 60), 1)
    s = (f"BALL:{debug_info.get('ball_state', '?')}  "
         f"TRACKS:{debug_info.get('n_tracks', 0)}  "
         f"REID:{debug_info.get('reid_known', 0)}known/{debug_info.get('reid_session', 0)}session  "
         f"LAST_SHOT:{debug_info.get('last_shot', '-')}  "
         f"LAST_GOAL:{debug_info.get('last_goal', '-')}")
    cv2.putText(out, s, (8, bar_top + 18), FONT, 0.45, (180, 220, 255), 1)


# COCO-17 skeleton edges for the pose wireframe overlay.
_COCO_SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
                  (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                  (0, 5), (0, 6), (0, 1), (0, 2), (1, 3), (2, 4)]


def _draw_skeleton(frame, kpts, color):
    """Draw a COCO-17 pose wireframe (broadcast-analysis style)."""
    if kpts is None or len(kpts) < 17:
        return
    for a, b in _COCO_SKELETON:
        if kpts[a][2] > 0.30 and kpts[b][2] > 0.30:
            cv2.line(frame, (int(kpts[a][0]), int(kpts[a][1])),
                     (int(kpts[b][0]), int(kpts[b][1])), color, 2, cv2.LINE_AA)
    for i in range(17):
        if kpts[i][2] > 0.30:
            cv2.circle(frame, (int(kpts[i][0]), int(kpts[i][1])), 3,
                       (245, 245, 245), -1, cv2.LINE_AA)


def _draw_trajectory(frame, trail):
    """Fading ball-trajectory trail. A segment longer than MAX_SEG px is NOT
    drawn, so the line breaks across teleports / gaps instead of painting big
    triangles across the court — the trail follows only the ball's real path."""
    MAX_SEG2 = 130 * 130
    pts = list(trail)
    n = len(pts)
    for i in range(1, n):
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 > MAX_SEG2:
            continue
        a = i / n
        cv2.line(frame, (x0, y0), (x1, y1),
                 (0, int(120 + 135 * a), 255), max(1, int(1 + 3 * a)), cv2.LINE_AA)


def _draw_goal_flash(out, scorer_label: str, team: str, frames_left: int):
    if frames_left <= 0: return
    h, w = out.shape[:2]
    alpha = min(1.0, frames_left / 30.0)
    overlay = out.copy()
    color = _VISUAL_COLORS.get(team, (200, 200, 200))
    cv2.rectangle(overlay, (0, h // 3), (w - SIDEBAR_W, h // 3 + 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65 * alpha, out, 1 - 0.65 * alpha, 0, out)
    text1 = "GOAL!"
    text2 = f"Scored by {scorer_label}"
    (tw1, _), _ = cv2.getTextSize(text1, FONT_B, 2.5, 5)
    (tw2, _), _ = cv2.getTextSize(text2, FONT, 1.0, 2)
    cv2.putText(out, text1, ((w - SIDEBAR_W - tw1) // 2, h // 3 + 60),  FONT_B, 2.5, color, 5)
    cv2.putText(out, text2, ((w - SIDEBAR_W - tw2) // 2, h // 3 + 105), FONT,   1.0, (255, 255, 255), 2)


def run(device="cuda:0", model="yolov8s-pose.pt", conf=0.40, ball_conf=0.02,
        imgsz=640, source="", scale=1.0, infer_every=2, display_every=1,
        analytics=True, debug=False, static_marker_enabled=True, no_ocr=False,
        motion_ball_enabled=False, court_mask_enabled=False,
        use_sahi=False, tracker_backend="botsort",
        use_osnet=False, max_frames=0, loop_source=False,
        gui_mode=False, frame_callback=None, status_callback=None,
        command_queue=None, controls=None, **kwargs):

    # ── Licensing gate (Pillars 1–3: node-lock + RSA expiry + clock guard) ──
    # OFF by default so development isn't blocked. Set env HB_LICENSE_ENFORCE=1 to
    # require a valid license.key. The check hard-exits (non-destructively) on any
    # failure. FOR A RELEASE BUILD: make this UNCONDITIONAL (delete the env check)
    # before compiling pipeline.py to a .pyd, so the gate can't be skipped by
    # unsetting an environment variable.
    import os as _os
    if _os.environ.get("HB_LICENSE_ENFORCE", "0") == "1":
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
            from licensing import require_valid_license
        except Exception as _e:
            print(f"[License] licensing module unavailable: {_e}")
            raise SystemExit(3)
        require_valid_license(_os.environ.get("HB_LICENSE", "license.key"))

    # ── Device safety ──
    # The .bat launchers hardcode cuda:0. If this machine has no GPU (or the
    # CUDA build of torch isn't installed) fall back to CPU with a clear notice
    # instead of crashing the whole run on the first inference call.
    if "cuda" in str(device).lower():
        try:
            import torch
            if not torch.cuda.is_available():
                print(f"[Pipeline] CUDA requested ({device}) but no GPU is "
                      f"available — falling back to CPU (slower).")
                device = "cpu"
        except Exception:
            print("[Pipeline] torch/CUDA probe failed — falling back to CPU.")
            device = "cpu"

    # ── Source validation ──
    # Give the end user a single clear line if they point at a missing file,
    # instead of a confusing silent exit from the capture thread.
    _src = str(source)
    _is_stream = _src.isdigit() or _src.startswith(("rtsp://", "http://", "https://"))
    if _src and not _is_stream and not Path(_src).exists():
        print(f"[Pipeline] ERROR: video file not found → {_src}")
        print("[Pipeline] Check the path/filename (drag the file onto the .bat) "
              "and try again.")
        return

    # ── Video / Test mode ──
    # A recorded broadcast has a MOVING, cutting camera, so anything that assumes
    # a fixed camera misbehaves: the static-marker filter's 10px cells are too
    # tight — camera jitter smears the goal/7m marker across several cells so it
    # never reaches the blacklist threshold and keeps getting picked as the ball.
    # When the source is a FILE (not a live stream) we auto-enter video mode and
    # widen the marker filter's cells so a jittering painted marker is still
    # caught, while its blacklist expires fast enough to survive a camera cut.
    # Override with video_mode=True/False if you ever need to force it.
    video_mode = kwargs.get("video_mode", None)
    if video_mode is None:
        video_mode = bool(_src) and not _is_stream

    print("[Pipeline] Starting...")
    print(f"  device={device}  model={model}  imgsz={imgsz}  source={source}")
    print(f"  conf={conf}  ball_conf={ball_conf}  infer_every={infer_every}  "
          f"display_every={display_every}  debug={debug}  no_ocr={no_ocr}")
    if video_mode:
        print("  [VIDEO/TEST MODE] moving-camera tolerant: wider marker cells, "
              "fast-expiring blacklist (no fixed-camera assumptions)")

    state    = MatchStateManager()
    if use_sahi:
        from sahi_detector import SAHIDetector
        detector = SAHIDetector(device=device, model=model, conf=conf,
                                  imgsz=imgsz, ball_conf=ball_conf)
    else:
        detector = Detector(device=device, model=model, conf=conf,
                             imgsz=imgsz, ball_conf=ball_conf)
    base_tracker = Tracker(device=device, backend=tracker_backend)

    # Phase-4: OSNet ReID — lazy-loaded; falls back to HSV fingerprint if init fails.
    osnet_reid = None
    if use_osnet:
        try:
            from osnet_reid import OSNetReID
            osnet_reid = OSNetReID(device=device, half=True)
            osnet_reid._ensure_loaded()
            print("[Pipeline] OSNet ReID enabled (replaces HSV fingerprint)")
        except Exception as e:
            print(f"[Pipeline] OSNet ReID init failed ({e}) — using HSV fingerprint")
            osnet_reid = None
    tracker  = PlayerReID(base_tracker, device=device)
    color_det = AutoColorDetector(min_crops=15)
    court_det = CourtDetector()
    coach     = HandballCoach(team_a_attacks_right=True)
    goal_detector = GoalPostDetector(frame_w=1280, frame_h=720)
    # ── Phase-2: Court Region Mask — kills crowd/bench/graphic detections ──
    court_mask = CourtMask(frame_w=1280, frame_h=720)
    # ── Motion-based ball finder: opt-in via --motion-ball ──
    # Catches fast/blurred balls YOLO misses but generates many extra
    # candidates. Default OFF — too noisy in busy scenes (lots of player
    # motion gets picked up too).
    motion_ball_finder = MotionBall() if motion_ball_enabled else None

    # تجهيز OCR
    ocr = None
    if not no_ocr and JerseyOCR is not None:
        ocr = JerseyOCR(device=device)

    msr = MultiSessionReID(MSR_PATH)
    # Real team + player names (France/Poland rosters from the league DB). Falls
    # back to Home/Away + number-only labels if the DB is empty. Press 'K' to
    # swap which colour cluster is which team if they come out backwards.
    match_id = MatchIdentity()
    if match_id.has_rosters:
        _tn = match_id.team_names()
        print(f"[Pipeline] match identity: team_a={_tn['team_a']}  "
              f"team_b={_tn['team_b']}  (press K to swap)")
    track_to_pid: dict[int, str] = {}
    track_to_ocr: dict[int, int] = {}  # حفظ أرقام التيشرتات لكل track id

    # Persistent OCR map: pid (MSR) → jersey_number. When a player disappears
    # and reappears with a new track-id, MSR matches on fingerprint → same pid,
    # and we restore their jersey number from this map. This is the "remember
    # who is who even if they leave and come back" persistence the user asked for.
    pid_to_ocr: dict[str, int] = {}

    # Bootstrap: restore previously-saved jersey numbers from MSR on startup.
    # MSR.rename() stores number in the player record, so any previous session
    # that locked numbers will hydrate this map immediately.
    try:
        for pid, rec in getattr(msr, "players", {}).items():
            num = rec.get("shirt_number", 0)
            if isinstance(num, int) and 1 <= num <= 99:
                pid_to_ocr[pid] = num
        if pid_to_ocr:
            print(f"[Pipeline] restored {len(pid_to_ocr)} jersey-number bindings from MSR")
    except Exception:
        pass

    capture = CaptureThread(source, queue_size=2, loop=loop_source); capture.start()

    analytics_q, analytics_proc, drops = None, None, 0
    if analytics:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        analytics_q = mp.Queue(maxsize=64)
        analytics_proc = mp.Process(
            target=background_analytics_worker,
            kwargs=dict(in_queue=analytics_q, homography_path=str(HOMOG_PATH),
                        snapshot_path=str(LOGS_DIR / "analytics_snapshot.npz"), snapshot_every_s=10.0),
            daemon=True, name="HandballAnalytics")
        analytics_proc.start()

    rules, frame_n, fps_ema, t_prev, _cached_tracks = None, 0, 0.0, time.perf_counter(), []
    _homography_for_rules = np.load(str(HOMOG_PATH)) if HOMOG_PATH.exists() else None

    # ── Runtime control plane (the "control center") ──
    # A shared dict the GUI/server mutates live: overlay toggles + AI thresholds.
    # Defaults give a clean broadcast look; nothing here breaks the headless path.
    if not isinstance(controls, dict):
        controls = {}
    controls.setdefault("skeleton",   False)
    controls.setdefault("boxes",      False)
    controls.setdefault("trajectory", True)
    controls.setdefault("ids",        True)
    controls.setdefault("markers",    True)
    controls.setdefault("heatmap",    False)
    controls.setdefault("player_conf", conf)
    controls.setdefault("ball_conf",   ball_conf)
    controls.setdefault("paused",     False)
    controls.setdefault("speed",      1.0)     # 0.25x .. 4x (files)
    controls.setdefault("seek",       None)    # one-shot jump 0..1
    controls.setdefault("step",       0)       # one-shot: advance N frames while paused
    _ball_trail = deque(maxlen=18)
    _last_vis   = None                         # last shown image (for freeze-on-pause)
    # Decoupled every-frame ball detection was tried and REVERTED: detecting the
    # ball on every frame makes the tracker pick a fresh best-scoring blob each
    # frame, so false positives (elbow/marker) make the ball TELEPORT to wrong
    # spots ("separated places, wrong route"). The infer-every-2 path with smooth
    # velocity extrapolation between frames keeps the ball on its real route and
    # measured higher continuity (83.6% vs 72.8% shown). Keep it OFF.
    _split_detect = False
    biomech     = BiomechAnalyzer()
    _technique: dict[int, dict] = {}     # track_id -> latest biomech metrics
    heatmap     = None                   # lazily created once frame size is known
    shot_map    = None
    _formation  = {}                     # latest team-shape metrics
    # AI Event Verifier (vision-LLM behind YOLO) — self-disables if no Ollama
    try:
        event_verifier = EventVerifier()
    except Exception as _e:
        print(f"[Pipeline] event verifier off ({_e})")
        event_verifier = None
    _recent_frames = deque(maxlen=20)    # rolling buffer for the LLM to judge goals

    _track_pos_hist: dict[int, deque] = {}
    _ball_pos_hist        = deque(maxlen=4)     # (frame_n, cx, cy) for ball interp
    _last_infer_frame_n   = 0
    _MAX_VEL_PX_PER_FRAME = 35.0
    _MAX_BALL_VEL_PX_PER_FRAME = 120.0         # fast shots move ~70-90 px/frame

    match_analyzer = MatchAnalyzer()
    static_marker = None

    session_id        = time.strftime("%Y%m%d_%H%M%S")
    _SOURCE_FPS_HINT  = 25.0
    _replay_fps       = max(5.0, _SOURCE_FPS_HINT / max(1, infer_every))
    goal_replay       = GoalReplayRecorder(
        out_dir=REPORTS_DIR / session_id / "goals",
        fps=_replay_fps,
        pre_seconds=4.0,
        post_seconds=1.5,
        downscale=0.5,
    )

    PIN_DURATION_S    = 4.5
    _pinned_freeze    = None
    _pinned_until_ts  = 0.0
    _pinned_idx_seen  = 0

    last_shot_label = "-"
    last_goal_label = "-"
    goal_flash_frames = 0
    goal_flash_team   = None
    goal_flash_label  = ""

    # ── Display sink ──
    # In standalone mode we own an OpenCV window. In gui_mode the host app
    # (handball_studio.py) supplies a frame_callback and we never touch cv2's
    # HighGUI — the same annotated frame is just handed to the GUI instead.
    def _show(img):
        nonlocal _last_vis
        _last_vis = img
        if gui_mode:
            if frame_callback is not None:
                frame_callback(img)
        else:
            cv2.imshow("Handball AI", img)

    if not gui_mode:
        cv2.namedWindow("Handball AI", cv2.WINDOW_NORMAL)

    # ── Active learning: pause-and-teach ──
    # Press 'P' to enter correction mode; L-click on ball, R-click on FPs,
    # C/SPACE to save. Press 'T' to fire a background retrain.
    from active_learning import ActiveLearningCorrector, trigger_retrain_background
    _video_stem = Path(source).stem if source else "live"
    al_corrector = ActiveLearningCorrector(
        dataset_root="training/dataset",
        video_stem=_video_stem,
    )
    if not gui_mode:
        cv2.setMouseCallback("Handball AI", al_corrector.on_mouse)
    print("[Pipeline] active learning READY — press P to pause+correct, T to retrain")

    def get_player_label(sid, team):
        # A jersey number is the only UNIQUE identity cue: _assign_ocr() keeps it
        # bound to exactly one track at a time (and the OCR block restores it
        # uniquely on re-entry), so a number-based name can never land on two
        # players at once. Appearance/colour ReID (msr) cannot tell same-team
        # players apart, so it's only the fallback for not-yet-numbered tracks.
        num = track_to_ocr.get(sid)
        if num is not None:
            named = match_id.player_label(team, num)   # "Karabatic #13"
            if named:
                return named
            prefix = "A" if team == "team_a" else ("B" if team == "team_b" else "U")
            return f"{prefix}#{num}"
        pid = track_to_pid.get(sid)
        return msr.label(pid) if pid else (f"S{sid}" if sid else "Unknown")

    def _assign_ocr(tid, num):
        # A jersey number belongs to ONE track at a time. OCR mis-reads sometimes
        # tagged several different players with the same number (the duplicate
        # "B#24"s). Newest lock wins; drop the number from any other track.
        for k in [k for k, v in track_to_ocr.items() if v == num and k != tid]:
            track_to_ocr.pop(k, None)
        track_to_ocr[tid] = num

    try:
        while True:
            # ── Playback control: seek / pause / step ──
            _seek = controls.get("seek")
            if _seek is not None:
                try: capture.seek_fraction(float(_seek))
                except Exception: pass
                controls["seek"] = None
                _ball_trail.clear()
            # ── Hot-reload court calibration (set by mobile_server /calib) ──
            if controls.get("reload_calib"):
                controls["reload_calib"] = False
                try:
                    if court_det._calib_path.exists():
                        court_det._load()
                    else:
                        court_det.clear_calibration()
                    _homography_for_rules = (np.load(str(HOMOG_PATH))
                                             if HOMOG_PATH.exists() else None)
                    if rules is not None:
                        if _homography_for_rules is not None:
                            rules.set_homography(_homography_for_rules)
                        if hasattr(rules, "court_transformer"):
                            rules.court_transformer.load_calibration()
                    print("[Pipeline] Court calibration reloaded from disk.")
                except Exception as e:
                    print(f"[Pipeline] Calibration reload failed: {e}")
            if bool(controls.get("paused")) and int(controls.get("step", 0) or 0) <= 0:
                if _last_vis is not None:
                    _show(_last_vis)                  # hold the frozen frame
                if gui_mode:
                    if command_queue is not None:
                        try:
                            if command_queue.get_nowait() == "q": break
                        except Exception:
                            pass
                else:
                    if (cv2.waitKey(30) & 0xFF) == ord("q"): break
                time.sleep(0.02)
                continue
            if int(controls.get("step", 0) or 0) > 0:
                controls["step"] = int(controls["step"]) - 1     # consume one step
            _frame_start = time.perf_counter()

            packet = capture.get_frame(timeout=0.5)
            if packet is None:
                if not capture.is_alive(): break
                continue

            frame = packet.frame
            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * (1280 / frame.shape[1]))),
                                   interpolation=cv2.INTER_LINEAR)

            frame_n += 1
            if max_frames and frame_n > max_frames:
                print(f"[Pipeline] reached --max-frames {max_frames} — stopping.")
                break
            if rules is None:
                rules = HandballEngine(frame_w=frame.shape[1], frame_h=frame.shape[0], fps=25.0)
                if _homography_for_rules is not None:
                    rules.set_homography(_homography_for_rules)
            
            goal_box = goal_detector.update(frame, frame_n)
            if rules is not None:
                rules._dynamic_goal_box = goal_box

            if not color_det.is_calibrated and frame_n > 90:
                color_det.force_calibration()

            # rolling buffer (downscaled) so the AI verifier can review a goal
            if event_verifier is not None and event_verifier.available:
                _bf = frame if frame.shape[1] <= 640 else cv2.resize(
                    frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                _recent_frames.append(_bf.copy())

            _do_infer = (frame_n % infer_every == 0) or (frame_n == 1)
            # Play-frame flag persists across skipped frames; only re-evaluated
            # at infer time so the pipeline stays cheap.
            if frame_n == 1:
                _is_play_frame = True

            if _do_infer:
                # Live AI threshold adjustment from the control center.
                _pc = controls.get("player_conf")
                if _pc is not None and hasattr(detector, "large_conf"):
                    detector.large_conf = float(_pc)
                    detector.small_conf = max(0.05, float(_pc) * 0.6)
                _bc = controls.get("ball_conf")
                if _bc is not None and hasattr(detector, "ball_conf"):
                    detector.ball_conf = max(0.02, float(_bc))
                if _split_detect:
                    _roi = getattr(tracker._tracker, "last_ball_pos", None)
                    _players_all = detector.detect_players(frame)
                    ball_cands   = detector.detect_ball(frame, roi_center=_roi)
                    other_dets_raw = [d for d in _players_all
                                      if d.y2 >= (frame.shape[0] * 0.30)]
                else:
                    raw_dets   = detector.detect(frame)
                    ball_cands = [d for d in raw_dets if getattr(d, 'class_id', 0) == BALL_CLASS_ID]
                    other_dets_raw = [d for d in raw_dets
                                      if getattr(d, 'class_id', 0) != BALL_CLASS_ID
                                      and d.y2 >= (frame.shape[0] * 0.30)]

                # ── Motion-based ball candidates (opt-in via --motion-ball) ──
                if motion_ball_finder is not None:
                    player_bboxes = [(p.x1, p.y1, p.x2, p.y2) for p in other_dets_raw]
                    motion_cands = motion_ball_finder.find(frame, player_bboxes)
                    if motion_cands:
                        ball_cands = list(ball_cands) + list(motion_cands)
                        if debug and (frame_n % 30 == 0):
                            print(f"[Pipeline] motion-ball added {len(motion_cands)} "
                                  f"fast-ball candidates")

                # ── Court-mask filter (opt-in via --court-mask) ──
                # Default OFF: matches the old stable pipeline. Only enable for
                # broadcasts where crowd / bench detections are a problem.
                if court_mask_enabled:
                    court_mask.update(frame, frame_n)
                    _is_play_frame = court_mask.is_play_frame()
                    if _is_play_frame:
                        other_dets = [d for d in other_dets_raw
                                      if court_mask.is_in_court((d.x1 + d.x2) * 0.5, d.y2)]
                        if debug and len(other_dets_raw) - len(other_dets) > 0 and (frame_n % 60 == 0):
                            print(f"[Pipeline] court-mask dropped "
                                  f"{len(other_dets_raw) - len(other_dets)}/"
                                  f"{len(other_dets_raw)} off-court "
                                  f"(visibility={court_mask.visibility:.2f})")
                    else:
                        other_dets = []
                        if debug and (frame_n % 30 == 0):
                            print(f"[Pipeline] graphic-frame guard ON  "
                                  f"(court visibility={court_mask.visibility:.2f})")
                else:
                    # Default path — no court filtering, all detections pass
                    _is_play_frame = True
                    other_dets = other_dets_raw

                if static_marker_enabled:
                    if static_marker is None:
                        if video_mode:
                            # Moving-camera tolerant: wider cells absorb camera
                            # jitter so the goal/7m marker stays in one cell long
                            # enough to be caught; ring=0 blocks only that tight
                            # cell (not a real flying ball nearby); short TTL so a
                            # camera cut clears stale spots within ~8s.
                            static_marker = StaticMarkerFilter(
                                frame_w=frame.shape[1], frame_h=frame.shape[0],
                                cell_size=22, neighbor_ring=0,
                                hit_window=60, hit_threshold=0.60,
                                warmup_frames=25, exclusion_ttl=200,
                                force_blacklist_ratio=0.75,
                            )
                        else:
                            static_marker = StaticMarkerFilter(
                                frame_w=frame.shape[1], frame_h=frame.shape[0],
                            )
                    player_xy = [((d.x1 + d.x2) * 0.5, (d.y1 + d.y2) * 0.5) for d in other_dets]
                   
                    static_marker.observe(ball_cands, frame_n, player_xy)
                    ball_cands_clean = static_marker.filter(ball_cands)
                    if debug and (frame_n % 60 == 0):
                        s = static_marker.stats()
                        if s["blacklist"] > 0:
                            print(f"[Pipeline] static-markers: {s['blacklist']} cells muted "
                                  f"({len(ball_cands)-len(ball_cands_clean)} dropped this frame)")
                else:
                    ball_cands_clean = ball_cands

                clean_dets = other_dets + ball_cands_clean
                tracks = tracker.update(clean_dets, frame)
                _cached_tracks = tracks
                _last_infer_frame_n = frame_n

                # تحديث الـ OCR
                if ocr is not None:
                    ocr.feed_tracks(frame, tracks, frame_n)
                    for t in tracks:
                        if getattr(t, 'class_id', 0) == BALL_CLASS_ID:
                            continue
                        tid = int(t.track_id)
                        num = ocr.locked_number(tid)
                        # 1. If OCR locked a number, save it AND bind to pid
                        if num is not None:
                            t.shirt_number = num
                            _assign_ocr(tid, num)
                            # Persist on the pid so if this player disappears
                            # and reappears with a new tid, we restore the number.
                            pid = track_to_pid.get(tid)
                            if pid is not None:
                                pid_to_ocr[pid] = num
                                # Also store in MSR's name field for cross-session persistence
                                try:
                                    team_letter = {"team_a": "A", "team_b": "B",
                                                    "goalkeeper": "GK"}.get(
                                        getattr(t, "team", "unknown"), "P")
                                    msr.rename(pid, name=f"{team_letter}#{num}",
                                                number=int(num))
                                except Exception:
                                    pass
                        # 2. Otherwise, RESTORE from pid_to_ocr if a previous
                        #    session bound this player's pid to a number
                        elif tid not in track_to_ocr:
                            pid = track_to_pid.get(tid)
                            if pid is not None and pid in pid_to_ocr \
                                    and pid_to_ocr[pid] not in track_to_ocr.values():
                                # only restore if that number isn't already on
                                # another active track (prevents duplicates)
                                track_to_ocr[tid] = pid_to_ocr[pid]
                                t.shirt_number = pid_to_ocr[pid]

                for t in tracks:
                    if getattr(t, 'class_id', 0) == BALL_CLASS_ID:
                        continue
                    hist = _track_pos_hist.setdefault(t.track_id, deque(maxlen=4))
                    hist.append((frame_n, float(t.cx), float(t.cy)))
                # Record the ball too, so skipped frames can interpolate it
                # (keeps a fast ball smooth instead of frozen-then-jumping).
                _bt = next((t for t in tracks
                            if getattr(t, 'class_id', 0) == BALL_CLASS_ID), None)
                if _bt is not None:
                    _ball_pos_hist.append((frame_n, float(_bt.cx), float(_bt.cy)))

                # ── Biomechanics + spatial: technique, heatmap, team shape ──
                if heatmap is None:
                    heatmap  = HeatMap(frame.shape[1], frame.shape[0])
                    shot_map = ShotMap(frame.shape[1], frame.shape[0])
                _active_ids = set()
                for t in tracks:
                    if getattr(t, 'class_id', 0) == BALL_CLASS_ID:
                        continue
                    _active_ids.add(t.track_id)
                    heatmap.add(float(t.cx), float(t.y2))
                    _m = biomech.update(t.track_id, getattr(t, 'keypoints', None),
                                        float(t.y2 - t.y1))
                    if _m:
                        _technique[t.track_id] = _m
                biomech.drop(_active_ids)
                for _tid in [tid for tid in _technique if tid not in _active_ids]:
                    _technique.pop(_tid, None)

                if frame_n % (infer_every * 8) == 0:
                    cutoff = frame_n - infer_every * 8
                    stale = [tid for tid, h in _track_pos_hist.items()
                             if (not h) or h[-1][0] < cutoff]
                    for tid in stale:
                        _track_pos_hist.pop(tid, None)
            else:
                elapsed = max(1, frame_n - _last_infer_frame_n)
                tracks = []
                for t in _cached_tracks:
                    if getattr(t, 'class_id', 0) == BALL_CLASS_ID:
                        if _split_detect:
                            continue   # ball is RE-DETECTED every frame below
                        # Smoothly advance the ball along its last known velocity
                        # so it doesn't visibly freeze between inference frames.
                        if len(_ball_pos_hist) >= 2:
                            (bf0, bx0, by0), (bf1, bx1, by1) = _ball_pos_hist[-2], _ball_pos_hist[-1]
                            bdf = max(bf1 - bf0, 1)
                            bvx = (bx1 - bx0) / bdf
                            bvy = (by1 - by0) / bdf
                            if (abs(bvx) <= _MAX_BALL_VEL_PX_PER_FRAME and
                                    abs(bvy) <= _MAX_BALL_VEL_PX_PER_FRAME):
                                bdx, bdy = bvx * elapsed, bvy * elapsed
                                try:
                                    tracks.append(dataclasses.replace(
                                        t,
                                        x1=t.x1 + bdx, y1=t.y1 + bdy,
                                        x2=t.x2 + bdx, y2=t.y2 + bdy,
                                    ))
                                    continue
                                except Exception:
                                    pass
                        tracks.append(t)
                        continue
                    hist = _track_pos_hist.get(t.track_id)
                    if hist and len(hist) >= 2:
                        (f0, x0, y0), (f1, x1, y1) = hist[-2], hist[-1]
                        df = max(f1 - f0, 1)
                        vx = (x1 - x0) / df
                        vy = (y1 - y0) / df
                        if abs(vx) > _MAX_VEL_PX_PER_FRAME or abs(vy) > _MAX_VEL_PX_PER_FRAME:
                            vx = vy = 0.0
                        dx, dy = vx * elapsed, vy * elapsed
                        try:
                            tracks.append(dataclasses.replace(
                                t,
                                x1=t.x1 + dx, y1=t.y1 + dy,
                                x2=t.x2 + dx, y2=t.y2 + dy,
                            ))
                        except Exception:
                            tracks.append(t)
                    else:
                        tracks.append(t)

                # ── Every-frame ball: real detection (ROI) + ball-only update ──
                # Pose was skipped this frame, but the cheap ball net still runs,
                # so the ball stays REAL instead of guessed between pose frames.
                if _split_detect:
                    _roi = getattr(tracker._tracker, "last_ball_pos", None)
                    _bc  = detector.detect_ball(frame, roi_center=_roi)
                    if static_marker_enabled and static_marker is not None:
                        _bc = static_marker.filter(_bc)
                    _bt2 = tracker._tracker.update_ball(_bc, tracks, frame)
                    if _bt2 is not None:
                        tracks.append(_bt2)
                        _ball_pos_hist.append((frame_n, float(_bt2.cx), float(_bt2.cy)))

            ball_track = next((t for t in tracks if getattr(t, 'class_id', 0) == BALL_CLASS_ID), None)
            ball_hallucinated_now = bool(getattr(tracker._tracker, 'ball_was_hallucinated', False))
            ball_state_str = "real" if ball_track and not ball_hallucinated_now else \
                             "phantom" if ball_track else "lost"

            if _do_infer:
                color_det.feed(frame, tracks)
                for t in tracks:
                    if getattr(t, 'class_id', 0) != BALL_CLASS_ID:
                        t.team = state.resolve_team(t.track_id, color_det.classify(frame, t))

            # Rules engine only runs during real play frames — kills phantom
            # events generated during pre-match graphics / replays / non-play.
            if _is_play_frame:
                try:
                    new_events = rules.update(tracks, ball_track, frame_n,
                                              ball_hallucinated=ball_hallucinated_now)
                except Exception as _e:
                    new_events = []
                    if debug:
                        print(f"[Pipeline] rules.update error on frame {frame_n}: "
                              f"{type(_e).__name__}: {_e}")
            else:
                new_events = []

            for ev in new_events:
                match_analyzer.process_events([ev])

            # Don't pollute the goal-replay buffer with graphic / replay frames
            if _do_infer and _is_play_frame:
                goal_replay.push_frame(frame, tracks, ball_track, frame_n,
                                        ball_was_hallucinated=ball_hallucinated_now)

            for ev in new_events:
                if ev.event_type == EventType.SHOT:
                    sid = ev.data.get("shooter_id")
                    last_shot_label = get_player_label(sid, ev.team)
                    if shot_map is not None and ball_track is not None:
                        shot_map.add(ball_track.cx, ball_track.cy, ev.team, False)
                    if debug:
                        print(f"[f{frame_n}] SHOT      {ev.team}  shooter={last_shot_label}  "
                              f"speed={ev.data.get('speed_px', 0)} px/f")
                elif ev.event_type == EventType.GOAL:
                    sid = ev.data.get("scored_by_player_id") or ev.data.get("shooter_id")
                    label = get_player_label(sid, ev.team)
                    team_name = match_id.team_label(ev.team)
                    last_goal_label = f"{label} ({team_name})"
                    # Ask the vision-LLM to verify the outcome (async, non-blocking)
                    if event_verifier is not None and event_verifier.available:
                        event_verifier.verify_async(list(_recent_frames), "goal",
                                                    ev.team, frame_n)
                    if shot_map is not None:
                        _bxy = (ball_track.cx, ball_track.cy) if ball_track is not None \
                               else (rules.fw if ev.data.get('side') == 'right' else 0, rules.fh * 0.5)
                        shot_map.add(_bxy[0], _bxy[1], ev.team, True)
                    goal_flash_frames, goal_flash_team, goal_flash_label = 90, ev.team, label
                    src = ev.data.get("attribution_source", "?")
                    print(f"\n*** GOAL ! ***  {team_name.upper()}  scored by {label}  "
                          f"[via {src}]  side={ev.data.get('side', '-')}\n")

                    team_colors = color_det.team_bgr_colors() if hasattr(color_det, 'team_bgr_colors') else {}
                    opp_team = "team_b" if ev.team == "team_a" else "team_a"
                    goal_replay.trigger_goal(
                        goal_event=ev.to_dict(),
                        scorer_label=label,
                        team=ev.team,
                        team_color_bgr=team_colors.get(ev.team, _VISUAL_COLORS["team_a"]),
                        opponent_color_bgr=team_colors.get(opp_team, _VISUAL_COLORS["team_b"]),
                    )
                elif ev.event_type == EventType.POSSESSION_CHANGE and debug:
                    pid = ev.data.get("to_player_id") or ev.data.get("player_id") or "-"
                    print(f"[f{frame_n}] POSSESSION → {ev.team}  player_id={pid}")
                elif ev.event_type == EventType.FAST_BREAK and debug:
                    print(f"[f{frame_n}] FAST_BREAK  {ev.team}  defenders_back={ev.data.get('defenders_back', '?')}")

            if _do_infer:
                # Phase-4: prefer OSNet deep embeddings, fallback to HSV histogram
                if osnet_reid is not None:
                    player_tracks = [t for t in tracks
                                      if getattr(t, 'class_id', 0) != BALL_CLASS_ID]
                    if player_tracks:
                        boxes = np.array([[t.x1, t.y1, t.x2, t.y2] for t in player_tracks],
                                          dtype=np.float32)
                        feats = osnet_reid.embed_image_boxes(frame, boxes)
                        if feats is not None and len(feats) == len(player_tracks):
                            for t, fp in zip(player_tracks, feats):
                                t.fingerprint = fp
                        else:
                            # Fallback for this frame only
                            for t in player_tracks:
                                fp = _color_fingerprint(frame, t)
                                if fp is not None:
                                    t.fingerprint = fp
                else:
                    for t in tracks:
                        if getattr(t, 'class_id', 0) == BALL_CLASS_ID: continue
                        fp = _color_fingerprint(frame, t)
                        if fp is not None: t.fingerprint = fp
                for t in tracks:
                    if getattr(t, 'class_id', 0) == BALL_CLASS_ID: continue
                    fp_t = getattr(t, 'fingerprint', None)
                    team = getattr(t, 'team', 'unknown') or 'unknown'
                    
                    # استمرار الـ MSR كنسخة احتياطية
                    pid, is_new = msr.match_or_enroll(
                        t.track_id, fp_t if isinstance(fp_t, np.ndarray) else None, team)
                    if pid is not None:
                        track_to_pid[t.track_id] = pid
                        if is_new and debug:
                            print(f"[f{frame_n}] ReID new player  pid={pid}  label={msr.label(pid)}  team={team}")

            goal_events_payload = [{"type": "goal", "team": ev.team, "frame_n": ev.frame_n}
                                    for ev in new_events if ev.event_type == EventType.GOAL]
            if analytics_q is not None and _do_infer:
                payload_tracks = []
                for t in tracks:
                    if getattr(t, 'class_id', 0) == BALL_CLASS_ID:
                        payload_tracks.append((-1, float(t.cx), float(t.cy), BALL_CLASS_ID, 'unknown', None, None))
                    else:
                        pid_for = track_to_pid.get(t.track_id)
                        ana_id  = (hash(pid_for) & 0x7FFFFFFF) if pid_for else int(t.track_id)
                        kpts = getattr(t, 'keypoints', None)
                        fp_p = getattr(t, 'fingerprint', None)
                        payload_tracks.append((
                            ana_id, float(t.cx), float(t.cy), 0,
                            getattr(t, 'team', 'unknown') or 'unknown',
                            kpts if isinstance(kpts, np.ndarray) else None,
                            fp_p if isinstance(fp_p, np.ndarray) else None,
                        ))
                try:
                    analytics_q.put_nowait({
                        "frame_n": frame_n, "ts": time.perf_counter(),
                        "frame_size": (frame.shape[1], frame.shape[0]),
                        "tracks": payload_tracks, "events": goal_events_payload,
                    })
                except _queue.Full: drops += 1

            msr.maybe_save()

            now = time.perf_counter()
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            if frame_n % display_every == 0:
                if heatmap is not None:
                    heatmap.step()
                    if controls.get("heatmap"):
                        frame = heatmap.render(frame)
                t_colors = _VISUAL_COLORS.copy()
                _ov_skel = controls.get("skeleton")
                _ov_box  = controls.get("boxes")
                _ov_ids  = controls.get("ids", True)
                _ov_mark = controls.get("markers", True)
                _ov_traj = controls.get("trajectory", True)
                _frame_label_seen: dict[str, int] = {}   # de-dup names within a frame
                for t in tracks:
                    col = t_colors.get(getattr(t, 'team', 'unknown'), t_colors["unknown"])
                    if getattr(t, 'class_id', 0) == BALL_CLASS_ID:
                        ball_col = (0, 220, 220) if not ball_hallucinated_now else (0, 140, 255)
                        cv2.circle(frame, (int(t.cx), int(t.cy)), 8, ball_col, -1)
                        # Only trail the REAL ball — phantom/coast guesses would
                        # paint a messy trail to wherever they jump.
                        if not ball_hallucinated_now:
                            _ball_trail.append((int(t.cx), int(t.cy)))
                        else:
                            _ball_trail.clear()   # break the trail while guessing
                        if ball_hallucinated_now and debug:
                            cv2.putText(frame, "ghost", (int(t.cx) + 10, int(t.cy)),
                                        FONT, 0.4, ball_col, 1)
                        continue
                    if _ov_skel:
                        _draw_skeleton(frame, getattr(t, 'keypoints', None), col)
                    if _ov_box:
                        cv2.rectangle(frame, (int(t.x1), int(t.y1)),
                                      (int(t.x2), int(t.y2)), col, 1, cv2.LINE_AA)
                    if _ov_mark:
                        cv2.ellipse(frame, (int(t.cx), int(t.y2)), (16, 8), 0, 0, 360, col, 2)
                    if _ov_ids:
                        label = get_player_label(t.track_id, getattr(t, 'team', 'unknown'))
                        # Two co-visible players can't be the same person. If an
                        # appearance/colour ReID stamped this name on another
                        # track already this frame, the later one is a mis-match —
                        # show a neutral per-track id so one name never appears on
                        # multiple players (the "same name on everyone" bug).
                        prev = _frame_label_seen.get(label)
                        if prev is not None and prev != t.track_id:
                            label = f"id{t.track_id}"
                        else:
                            _frame_label_seen[label] = t.track_id
                        cv2.putText(frame, label, (int(t.x1), int(t.y1) - 5), FONT_B, 0.55, col, 2)
                if _ov_traj and len(_ball_trail) > 1:
                    _draw_trajectory(frame, _ball_trail)

                vis = _draw_hud(frame, rules.hud_data() | {"fps": fps_ema}, rules, color_det,
                                match_id.team_names(), frame_n, state)

                if goal_flash_frames > 0:
                    _draw_goal_flash(vis, goal_flash_label, goal_flash_team, goal_flash_frames)
                    goal_flash_frames -= 1

                if debug:
                    _draw_debug_panel(vis, {
                        "ball_state": ball_state_str,
                        "n_tracks":   sum(1 for t in tracks if getattr(t, 'class_id', 0) != BALL_CLASS_ID),
                        "reid_known": len(msr),
                        "reid_session": len(track_to_pid),
                        "last_shot":  last_shot_label,
                        "last_goal":  last_goal_label,
                    })
                    if static_marker is not None:
                        for x1, y1, x2, y2 in static_marker.blacklist_boxes():
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    # Stationary-killer blacklist (red dashed circles)
                    base_t = getattr(tracker, "_tracker", None)
                    if base_t is not None and hasattr(base_t, "get_stationary_blacklist"):
                        for sx, sy, sr in base_t.get_stationary_blacklist():
                            cv2.circle(vis, (int(sx), int(sy)), int(sr),
                                        (0, 50, 255), 2, lineType=cv2.LINE_AA)
                            cv2.putText(vis, "MARKER", (int(sx) - 28, int(sy) - sr - 4),
                                         FONT, 0.4, (0, 50, 255), 1, cv2.LINE_AA)

                if scale != 1.0:
                    vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))

                if goal_box is not None:
                    goal_detector.draw_overlay(vis)

                if goal_replay.last_freeze_idx > _pinned_idx_seen \
                        and goal_replay.last_freeze is not None:
                    _pinned_idx_seen = goal_replay.last_freeze_idx
                    _pinned_freeze   = goal_replay.last_freeze
                    _pinned_until_ts = time.perf_counter() + PIN_DURATION_S
                    print(f"[Pipeline] tactical freeze pinned for {PIN_DURATION_S:.1f}s "
                          f"({goal_replay.last_scorer_label} / {goal_replay.last_team})  "
                          f"— press SPACE to dismiss")

                now_ts = time.perf_counter()
                if _pinned_freeze is not None and now_ts < _pinned_until_ts:
                    pin_show = cv2.resize(_pinned_freeze, (vis.shape[1], vis.shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
                    remaining = max(0.0, _pinned_until_ts - now_ts)
                    msg = f"[ TACTICAL FREEZE  {remaining:0.1f}s ]   SPACE = resume"
                    cv2.rectangle(pin_show, (0, pin_show.shape[0] - 36),
                                   (pin_show.shape[1], pin_show.shape[0]),
                                   (0, 0, 0), -1)
                    cv2.putText(pin_show, msg, (16, pin_show.shape[0] - 12),
                                FONT_B, 0.75, (0, 230, 255), 2, cv2.LINE_AA)
                    _show(pin_show)
                else:
                    if _pinned_freeze is not None and now_ts >= _pinned_until_ts:
                        _pinned_freeze = None
                    _show(vis)

                # ── Push live status to the GUI (every ~5 displayed frames) ──
                if gui_mode and status_callback is not None and frame_n % 5 == 0:
                    _players_live, _seen, _tech = [], set(), []
                    for t in tracks:
                        if getattr(t, 'class_id', 0) == BALL_CLASS_ID:
                            continue
                        lbl = get_player_label(t.track_id, getattr(t, 'team', 'unknown'))
                        if lbl in _seen:
                            continue
                        _seen.add(lbl)
                        _team = getattr(t, 'team', 'unknown')
                        _players_live.append({"label": lbl, "team": _team})
                        _m = _technique.get(t.track_id)
                        if _m:
                            _tech.append({"label": lbl, "team": _team, **_m})
                    # team shape uses teams (now resolved) → foot positions
                    _shape_pts = [(float(t.cx), float(t.y2), getattr(t, 'team', 'unknown'))
                                  for t in tracks if getattr(t, 'class_id', 0) != BALL_CLASS_ID]
                    _formation = TeamShape.analyse(_shape_pts, frame.shape[1], frame.shape[0])
                    try:
                        status_callback({
                            "hud":         rules.hud_data(),
                            "fps":         round(fps_ema, 1),
                            "ball_state":  ball_state_str,
                            "last_shot":   last_shot_label,
                            "last_goal":   last_goal_label,
                            "events":      rules.recent_events_text(12),
                            "players":     _players_live,
                            "reid_known":  len(msr),
                            "session_id":  session_id,
                            "goals_dir":   str(REPORTS_DIR / session_id / "goals"),
                            "frame_n":     frame_n,
                            "pos_frames":  capture.position()[0],
                            "total_frames": capture.position()[1],
                            "src_fps":     capture.fps(),
                            "controls":    dict(controls),
                            "technique":   _tech,
                            "formation":   _formation,
                            "shotmap":     shot_map.data() if shot_map is not None else [],
                            "ai_verdict":  (event_verifier.get_latest()
                                            if event_verifier is not None else None),
                        })
                    except Exception:
                        pass

            if gui_mode:
                # Controls arrive from the GUI as single-char commands.
                key = 255
                if command_queue is not None:
                    try:
                        cmd = command_queue.get_nowait()
                        if isinstance(cmd, str) and len(cmd) == 1:
                            key = ord(cmd)
                    except Exception:
                        key = 255
            else:
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord(" "):
                if _pinned_freeze is not None:
                    _pinned_freeze = None
                    _pinned_until_ts = 0.0
                    print("[Pipeline] tactical freeze dismissed by user")
            elif key == ord("p") and not gui_mode:
                # ── Active-learning correction mode ──
                # Enter inner pause loop: pipeline freezes, user clicks to label.
                # Use the current displayed frame (with overlays) for context,
                # but save the RAW frame so the trainer sees clean pixels.
                al_corrector.begin(frame, frame_n)
                print(f"[Pipeline] PAUSED at frame {frame_n} for correction. "
                      f"L=ball  R=NO-ball  U=undo  C/SPACE=save  X=discard")
                while al_corrector.active:
                    # Re-render the overlay each tick so the boxes/X marks
                    # update as the user clicks
                    overlay = al_corrector.render_overlay(frame)
                    if scale != 1.0:
                        overlay = cv2.resize(overlay,
                                              (int(overlay.shape[1] * scale),
                                               int(overlay.shape[0] * scale)))
                    cv2.imshow("Handball AI", overlay)
                    k2 = cv2.waitKey(30) & 0xFF
                    if k2 == 0xFF:
                        continue
                    action = al_corrector.handle_key(k2)
                    if action == 'save':
                        al_corrector.end_save()
                        print(f"[Pipeline] correction saved. total session: "
                              f"{al_corrector.total_saved}")
                        break
                    elif action == 'discard':
                        al_corrector.end_discard()
                        print("[Pipeline] correction discarded")
                        break
                # Re-install mouse callback in case it got cleared
                cv2.setMouseCallback("Handball AI", al_corrector.on_mouse)
            elif key == ord("t"):
                s = al_corrector.stats()
                print(f"[Pipeline] retrain trigger — protected frames so far: "
                      f"{s['total_protected']}")
                if s['total_protected'] < 5:
                    print("[Pipeline] need at least 5 corrected frames before retrain")
                else:
                    trigger_retrain_background()
            elif key == ord("r") and color_det.is_calibrated: color_det.recalibrate()
            elif key == ord("f"): rules.flip_sides(); coach.flip_sides()
            elif key == ord("a"): rules.manual_goal("team_a"); print("[Pipeline] manual goal team_a")
            elif key == ord("b"): rules.manual_goal("team_b"); print("[Pipeline] manual goal team_b")
            elif key == ord("d"): debug = not debug; print(f"[Pipeline] debug={'ON' if debug else 'OFF'}")
            elif key == ord("k"):
                match_id.swap()
                _tn = match_id.team_names()
                print(f"[Pipeline] swapped teams → team_a={_tn['team_a']}  "
                      f"team_b={_tn['team_b']}")

            # ── Speed pacing (variable-speed playback for files) ──
            _spd = float(controls.get("speed", 1.0) or 1.0)
            _target_dt = 1.0 / max(1.0, _SOURCE_FPS_HINT * max(0.1, _spd))
            _dt = time.perf_counter() - _frame_start
            if _dt < _target_dt:
                time.sleep(_target_dt - _dt)

    finally:
        capture.stop()
        if not gui_mode:
            try: cv2.destroyAllWindows()
            except Exception: pass
        try:
            goal_replay.shutdown()
        except Exception as e:
            print(f"[Pipeline] goal_replay shutdown failed: {e}")
        try:
            if ocr is not None and hasattr(ocr, "shutdown"):
                ocr.shutdown()
        except Exception as e:
            print(f"[Pipeline] jersey_ocr shutdown failed: {e}")
        try:
            msr.save()
            print(f"[Pipeline] saved {len(msr)} persistent players → {MSR_PATH}")
        except Exception as e:
            print(f"[Pipeline] msr save failed: {e}")
        if analytics_proc is not None:
            try: analytics_q.put("__STOP__", timeout=1.0)
            except Exception: pass
            analytics_proc.join(timeout=8.0)
            if analytics_proc.is_alive(): analytics_proc.terminate()
            print(f"[Pipeline] analytics shutdown. dropped_payloads={drops}")
        if rules is not None:
            print(f"[Pipeline] final score — Home {rules.score['team_a']} : {rules.score['team_b']} Away")
            print(f"[Pipeline] events fired = {len(rules.events)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model",  default="yolov8s-pose.pt")
    parser.add_argument("--conf",   type=float, default=0.40)
    parser.add_argument("--ball-conf", type=float, default=0.02)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--source", default="")
    parser.add_argument("--scale",  type=float, default=1.0)
    parser.add_argument("--infer-every",   type=int, default=2)
    parser.add_argument("--display-every", type=int, default=1,
                         help="Render every Nth frame. Default 1 = show every "
                              "frame (smooth). Inference still runs every "
                              "--infer-every frames; skipped frames are filled "
                              "with motion-interpolated tracks.")
    parser.add_argument("--max-frames", type=int, default=0,
                         help="Stop after N processed frames (0 = run to end). "
                              "Handy for a quick smoke-test on a clip.")
    parser.add_argument("--no-server",    action="store_true")
    parser.add_argument("--no-analytics", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR completely.")
    parser.add_argument("--no-static-marker", action="store_true",
                         help="Disable static-marker filter (default ON — uses "
                              "smart player-aware detection so it won't suppress held balls).")
    parser.add_argument("--motion-ball", action="store_true",
                         help="Enable motion-based ball candidate generator (default OFF).")
    parser.add_argument("--court-mask", action="store_true",
                         help="Enable court-region mask filtering (default OFF — "
                              "matches the old stable pipeline behaviour).")
    parser.add_argument("--use-sahi", action="store_true",
                         help="Use SAHI sliced inference (Phase-2 detector). Better recall "
                              "on distant/small players via overlapping 640x640 patches. "
                              "Cost: ~3x slower than full-frame YOLO. Recommended for "
                              "wide-angle broadcast footage.")
    parser.add_argument("--tracker", default="botsort", choices=["botsort", "ocsort"],
                         help="Player-tracking backend. ocsort = better HOTA on sports "
                              "benchmarks, fewer ID switches (Phase-3 upgrade).")
    parser.add_argument("--use-osnet", action="store_true",
                         help="Use OSNet deep ReID embeddings (Phase-4 upgrade) instead "
                              "of HSV color fingerprint. Robust to court-color bleed.")
    parser.add_argument("--video-mode", dest="force_video", action="store_true",
                         help="Force moving-camera (broadcast/test) mode. Auto-on for "
                              "file sources; this forces it even for a stream.")
    parser.add_argument("--live-mode", dest="force_live", action="store_true",
                         help="Force fixed-camera (live) mode even for a file source.")
    args = vars(parser.parse_args())
    _fv, _fl = args.pop("force_video", False), args.pop("force_live", False)
    if _fv:
        args["video_mode"] = True
    elif _fl:
        args["video_mode"] = False   # else: leave unset → auto-detect from source
    args["motion_ball_enabled"]   = args.pop("motion_ball", False)
    args["static_marker_enabled"] = not args.pop("no_static_marker", False)
    args["court_mask_enabled"]    = args.pop("court_mask", False)
    args["use_sahi"]              = args.pop("use_sahi", False)
    args["tracker_backend"]       = args.pop("tracker", "botsort")
    args["use_osnet"]             = args.pop("use_osnet", False)
    args["analytics"]             = not args.pop("no_analytics", False)
    args.pop("no_server", None)
    run(**args)