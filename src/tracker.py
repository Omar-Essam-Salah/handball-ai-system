"""
BoT-SORT tracker wrapper — assigns stable IDs to detections across frames.

Hybrid version:
    - Player tracking: keeps the ultralytics-8.4 _TrackerInput adapter and
      the lowered confidence thresholds (post-mortem fix that restored
      player tracking after BoT-SORT API broke).
    - Ball tracking: reverted to the simpler, more stable scoring from the
      pre-Phase-2 version. The continuity bonuses + free-flight extrapolation
      I added later turned out to over-correct — the ball started teleporting
      around (the user's "ghost flying everywhere" complaint). Going back to:
          * one penalty for big jumps
          * hand-anchor phantom only (Mode-1)
          * no Mode-2 linear extrapolation
          * short holder grace (10 frames) for brief releases during a pass.
"""
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

from ultralytics.trackers.bot_sort import BOTSORT

from ball_kalman   import BallKalman
from ball_appearance import BallAppearance, skin_ratio
# Phase-2 (TemporalBallVoter) was REMOVED from the live tracker: its bonus
# rewarded any candidate with neighbours in the last 3 frames within 120px,
# which inadvertently favoured STATIC false positives (heads, floor markers)
# over real balls that move >120px between frames. The Kalman filter already
# handles the temporal aspect correctly. Module kept at src/temporal_ball.py
# for reference but no longer imported.


# ── Adapter for ultralytics ≥ 8.3 ─────────────────────────────────────────
# The new BoT-SORT API expects an object with .conf / .cls / .xyxy / .xywh
# instead of a raw (N, 6) numpy array. We wrap the array in a thin shim so
# the rest of this module keeps the simple `det_array` flow.
class _TrackerInput:
    __slots__ = ("xyxy", "conf", "cls", "_xywh")

    def __init__(self, det_array: np.ndarray):
        if det_array.ndim != 2 or det_array.shape[1] < 6:
            det_array = np.zeros((0, 6), dtype=np.float32)
        self.xyxy = det_array[:, :4].astype(np.float32, copy=False)
        self.conf = det_array[:, 4].astype(np.float32, copy=False)
        self.cls  = det_array[:, 5].astype(np.float32, copy=False)
        self._xywh: Optional[np.ndarray] = None

    @property
    def xywh(self) -> np.ndarray:
        if self._xywh is None:
            wh  = self.xyxy[:, 2:4] - self.xyxy[:, 0:2]
            cxy = (self.xyxy[:, 0:2] + self.xyxy[:, 2:4]) * 0.5
            self._xywh = np.concatenate([cxy, wh], axis=1).astype(np.float32)
        return self._xywh

    def __getitem__(self, idx) -> "_TrackerInput":
        sub = _TrackerInput.__new__(_TrackerInput)
        sub.xyxy = self.xyxy[idx]
        sub.conf = self.conf[idx]
        sub.cls  = self.cls[idx]
        sub._xywh = None
        return sub

    def __len__(self) -> int:
        return int(self.conf.shape[0])


PERSON_CLASS = 0
BALL_CLASS   = 32

BALL_MIN_AREA_PX      = 10
BALL_MAX_AREA_PX      = 15000
BALL_ASPECT_RATIO_MAX = 7.0
BALL_SCORE_MIN        = 0.03

ANTI_SHOE_PENALTY     = -1.5
HAND_MAGNET_PX        = 140.0
HAND_MAGNET_BONUS     = 0.90
AIRWAY_BONUS          = 0.50

KP_NOSE           = 0     # COCO pose nose keypoint
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_CONF_MIN       = 0.35


def _iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = x2 - x1, y2 - y1
    if iw <= 0.0 or ih <= 0.0: return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    keypoints: Optional[np.ndarray] = None
    team: str = "unknown"
    fingerprint: Optional[np.ndarray] = None

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def is_ball(self) -> bool: return self.class_id == BALL_CLASS


class Tracker:
    BALL_TRACK_ID = 99999

    # Hand-magnet memory: brief release during a pass / fake doesn't lose the
    # holder. 10 frames = ~0.4s @ 25 fps — enough to bridge a fake but short
    # enough to not over-trust a stale holder.
    HOLDER_MEMORY_FRAMES = 10

    # Kalman-coast bridging: when YOLO misses the ball mid-flight (fast pass /
    # shot blur) we emit the Kalman-predicted position for up to this many
    # frames so the ball doesn't blink out. BOUNDED on purpose — this is NOT
    # the old unbounded linear extrapolation that flew off into the crowd.
    # 10 frames ≈ 0.33-0.40s; long enough to bridge a blur, short enough that a
    # genuinely-lost ball disappears quickly instead of drifting.
    BALL_COAST_MAX = 10

    # Re-acquisition: once the ball has been missing for this many frames, the
    # Kalman prediction has drifted and is no longer trustworthy as a gate — a
    # real detection that reappears elsewhere keeps getting rejected for "not
    # matching the trajectory" (measured: ~11% of frames had a candidate the
    # gate threw away). After this many misses we DROP the gate so a confident
    # detection can re-lock immediately. Kept small so it only kicks in during a
    # genuine gap, not during normal frame-to-frame tracking.
    #
    # Tuned 3→2 against the 67%-recall ball model (handball_ball.pt, 2026-06-06):
    # the higher-recall model puts a real candidate back on screen sooner, so
    # dropping the drifted gate one frame earlier lets the track re-lock faster.
    # A/B on handball.mp4 (1200 frames, diag_ball_trace.py): real detections
    # 40.0%→41.9%, rejected candidates 82→74, continuity 68.9%→69.4%. The few
    # extra >150px "jumps" are real frames snapping back to truth, not teleports.
    REACQUIRE_AFTER_FRAMES = 2
    REACQUIRE_MIN_CONF     = 0.12   # don't re-lock onto near-zero-conf noise

    # Anti-teleport: the ball cannot jump more than this many px between ball
    # updates. A real fast shot moves ~180px per 2-frame update; a false
    # positive across the court is 300px+. A candidate beyond the cap is only
    # accepted if it CONFIRMS (a candidate reappears near it next update) — that
    # distinguishes a genuine re-acquisition from a one-frame phantom blob.
    BALL_MAX_JUMP_PX   = 230.0
    JUMP_CONFIRM_PX    = 90.0

    # Phantom budget: after this many consecutive ball-updates with NO real
    # detection, stop drawing a guessed ball entirely. Better to show an honest
    # gap than a confident-but-wrong ball stuck on a player's shoulder.
    PHANTOM_MAX_FRAMES = 12

    # Stationary-ball killer: if the "best" ball candidate has stayed in
    # essentially the same pixel for STATIONARY_FRAMES_LIMIT frames, it's
    # almost certainly a painted floor marker — not a real ball. Real balls
    # ALWAYS move (player breathing alone shifts a held ball >2px). We blacklist
    # that exact pixel for STATIONARY_BLACKLIST_FRAMES so we don't keep
    # re-acquiring the same false positive.
    # Anti-head appearance filter. Heads (players AND crowd fans) are skin-
    # coloured; a handball is not. A candidate that is mostly skin and NOT in a
    # hand is rejected as a head; a partly-skin one is discouraged. The adaptive
    # ball-colour model adds a bonus so the candidate that actually looks like
    # the recently-seen ball beats a look-alike blob.
    SKIN_HARD_REJECT = 0.58   # ≥58% skin pixels + no hand → it's a head, drop it
    SKIN_SOFT        = 0.32   # ≥32% skin → penalise (could be head edge / arm)
    SKIN_PENALTY     = -0.70
    APPEARANCE_W     = 0.55   # weight of the learned-ball-colour similarity bonus
    APPEARANCE_LEARN_CONF = 0.30   # only learn colour from confident real detections

    STATIONARY_PIXEL_TOL         = 4      # ≤4px movement between frames = "stationary"
    STATIONARY_FRAMES_LIMIT      = 4      # 4 stationary frames → blacklist (≈0.16s @25fps)
    STATIONARY_BLACKLIST_FRAMES  = 1500   # ~60s before re-checking
    STATIONARY_BLACKLIST_RADIUS  = 22     # any candidate within 22px of a blacklisted point dies

    def __init__(self, device="cuda:0", frame_rate=30, backend="botsort",
                 ball_coast_max=None, reacquire_after=None):
        """``backend``: 'botsort' (default, ultralytics built-in) or 'ocsort'
        (via boxmot — better HOTA on sports benchmarks, fewer ID switches).
        ``ball_coast_max``: override the Kalman-coast length (frames).
        ``reacquire_after``: override gate-drop threshold (frames missed)."""
        self.device = device
        self.backend = backend.lower()
        if ball_coast_max is not None:
            self.BALL_COAST_MAX = int(ball_coast_max)
        if reacquire_after is not None:
            self.REACQUIRE_AFTER_FRAMES = int(reacquire_after)
        self.last_ball_pos = None
        self.last_holder = None              # most recently confirmed holder
        self._holder_grace = 0               # frames remaining of "soft" holder memory
        self._ball_coast = 0                 # consecutive Kalman-coast frames so far
        self._pending_jump = None            # candidate awaiting jump-confirmation
        self._frames_since_real = 0          # ball-updates since the last REAL detection
        # Stationary-killer state
        self._stationary_streak: int = 0
        self._stationary_blacklist: list[tuple[float, float, int]] = []   # (x, y, expire_frame)

        # ── Kalman ball tracker ──
        # Predicts ball position from velocity + acceleration history. Detections
        # that don't match the trajectory are gated out automatically — this is
        # what kills head/marker false positives WITHOUT relying on filters.
        self.ball_kf = BallKalman()

        # Adaptive ball-colour model (anti-head appearance check).
        self._ball_appear = BallAppearance()

        # ── BoT-SORT config (player tracking) ──
        # The thresholds here are the Phase-2 fix that restored player tracking
        # after ultralytics changed the BoT-SORT API. Don't raise these without
        # checking diagnose.py first — too high → no player tracks at all.
        try:
            from ultralytics.cfg import get_cfg
            cfg = get_cfg("botsort.yaml")
            cfg.track_buffer = 150
            cfg.match_thresh = 0.85
            cfg.track_high_thresh = 0.15
            cfg.track_low_thresh = 0.05
            cfg.new_track_thresh = 0.20
            if not hasattr(cfg, 'with_reid'):    cfg.with_reid = False
            if not hasattr(cfg, 'model'):        cfg.model = 'auto'
            if not hasattr(cfg, 'fuse_score'):   cfg.fuse_score = True
            if not hasattr(cfg, 'gmc_method'):   cfg.gmc_method = 'none'
            if not hasattr(cfg, 'proximity_thresh'):  cfg.proximity_thresh = 0.5
            if not hasattr(cfg, 'appearance_thresh'): cfg.appearance_thresh = 0.85
            args = cfg
        except Exception:
            args = SimpleNamespace(
                tracker_type='botsort',
                track_high_thresh=0.15, track_low_thresh=0.05,
                new_track_thresh=0.20, match_thresh=0.85, track_buffer=150,
                fuse_score=True,
                gmc_method='none', proximity_thresh=0.5, appearance_thresh=0.85,
                fallback_id_iou_thresh=0.2, shared_kalman=False,
                with_reid=False, model='auto',
            )

        # Initialize the chosen player tracker backend.
        if self.backend == "ocsort":
            try:
                from boxmot.trackers.ocsort.ocsort import OcSort
                self.player_tracker = OcSort(
                    det_thresh=0.15,       # match our BoT-SORT track_high_thresh
                    max_age=30,
                    min_hits=2,
                    asso_threshold=0.3,
                    delta_t=3,
                    asso_func="iou",
                    inertia=0.2,
                    use_byte=True,         # ByteTrack-style two-stage assoc → better recall
                    Q_xy_scaling=0.01,
                    Q_s_scaling=0.0001,
                )
                self._is_ocsort = True
                print("[Tracker] Using OC-SORT backend (boxmot)")
            except Exception as e:
                print(f"[Tracker] OC-SORT init failed ({e}), falling back to BoT-SORT")
                self.player_tracker = BOTSORT(args=args, frame_rate=frame_rate)
                self._is_ocsort = False
        else:
            self.player_tracker = BOTSORT(args=args, frame_rate=frame_rate)
            self._is_ocsort = False

        self.ball_was_hallucinated = False
        self.reid_model = getattr(self.player_tracker, 'reid_model', None)

    # ──────────────────────────────────────────────────────────────────
    def update(self, detections: list, frame: np.ndarray) -> list:
        tracks = []
        players_raw, ball_candidates_raw = [], []

        for d in detections:
            cls = getattr(d, 'class_id', 0)
            if cls == PERSON_CLASS:    players_raw.append(d)
            elif cls == BALL_CLASS:    ball_candidates_raw.append(d)

        # ── Player tracking ────────────────────────────────────────────
        tracked_players = []
        if players_raw:
            det_array = np.zeros((len(players_raw), 6), dtype=np.float32)
            for i, p in enumerate(players_raw):
                det_array[i] = [p.x1, p.y1, p.x2, p.y2, p.confidence, p.class_id]

            try:
                if self._is_ocsort:
                    # OC-SORT expects raw (N,6) numpy array; returns (M, 7) with
                    # columns [x1, y1, x2, y2, track_id, conf, cls].
                    raw_res = self.player_tracker.update(det_array, frame)
                    # Normalise to our (M, 6) format: [x1,y1,x2,y2,tid,conf]
                    if raw_res is not None and len(raw_res) > 0:
                        res = np.column_stack([
                            raw_res[:, 0:4],          # xyxy
                            raw_res[:, 4],            # track_id
                            raw_res[:, 5],            # conf
                        ])
                    else:
                        res = np.zeros((0, 6))
                else:
                    tracker_input = _TrackerInput(det_array)
                    res = self.player_tracker.update(tracker_input, frame)

                for r in res:
                    tid, best_iou, kpts = int(r[4]), 0.0, None
                    for p in players_raw:
                        iou = _iou_xyxy(r[:4], [p.x1, p.y1, p.x2, p.y2])
                        if iou > best_iou:
                            best_iou, kpts = iou, getattr(p, 'keypoints', None)
                    tracks.append(Track(
                        x1=float(r[0]), y1=float(r[1]),
                        x2=float(r[2]), y2=float(r[3]),
                        track_id=tid, confidence=float(r[5]),
                        class_id=PERSON_CLASS, keypoints=kpts,
                    ))
                    tracked_players.append(tracks[-1])
            except Exception as _e:
                if not getattr(self, "_tracker_err_logged", False):
                    import traceback
                    backend_name = "OC-SORT" if self._is_ocsort else "BoT-SORT"
                    print(f"[Tracker] {backend_name} update failed: "
                          f"{type(_e).__name__}: {_e}")
                    traceback.print_exc()
                    self._tracker_err_logged = True

        # ── Ball (separated so the pipeline can run it every frame) ──
        ball_track = self.update_ball(ball_candidates_raw, tracked_players, frame)
        if ball_track is not None:
            tracks.append(ball_track)
        return tracks

    # ──────────────────────────────────────────────────────────────────
    def update_ball(self, ball_candidates_raw: list, tracked_players: list,
                    frame: np.ndarray) -> "Optional[Track]":
        """Pick / track the single ball from candidates, independent of player
        tracking — so the pipeline can run it EVERY frame (the ball model is
        cheap) while the expensive pose model runs every ``infer_every`` frames.
        Returns one ball Track (real / phantom / coast) or None."""
        # Frame counter for blacklist expiry (one tick per call = per frame).
        self._stationary_frame_n = getattr(self, "_stationary_frame_n", 0) + 1
        fn = self._stationary_frame_n
        if self._stationary_blacklist:
            self._stationary_blacklist = [
                (x, y, exp) for (x, y, exp) in self._stationary_blacklist if exp > fn]
        if self._stationary_blacklist:
            ball_candidates = []
            R2 = self.STATIONARY_BLACKLIST_RADIUS ** 2
            for b in ball_candidates_raw:
                bcx = (b.x1 + b.x2) * 0.5; bcy = (b.y1 + b.y2) * 0.5
                if not any((bcx - bx) ** 2 + (bcy - by) ** 2 < R2
                           for (bx, by, _e) in self._stationary_blacklist):
                    ball_candidates.append(b)
        else:
            ball_candidates = ball_candidates_raw

        self.ball_was_hallucinated = False
        best_ball, best_score = None, BALL_SCORE_MIN
        ball_in_hand = False
        current_holder = None
        ball_track = None

        # ── Kalman predict (one-step extrapolation) ──
        self.ball_kf.predict()

        # ── HARD head/face rejection (uses nose + eye keypoints) ──
        # The fine-tuned model was auto-labelled and inherited some
        # head-as-ball false positives from the bootstrap data. The user
        # repeatedly reports "system sees player's head as the ball" and
        # "ball in net not counted because tracker is on a head". We kill
        # this once and for all by REJECTING any ball candidate whose
        # centroid sits within HEAD_REJECT_PX of any visible nose/eye
        # keypoint, regardless of confidence or hand proximity.
        HEAD_REJECT_PX = 30.0
        head_reject_pts: list[tuple[float, float]] = []
        for p in tracked_players:
            kp = getattr(p, 'keypoints', None)
            if kp is None or kp.shape[0] < 17:
                continue
            for idx in (KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE):
                if float(kp[idx][2]) >= 0.30:
                    head_reject_pts.append((float(kp[idx][0]), float(kp[idx][1])))

        for _idx, b in enumerate(ball_candidates):
            w, h = b.x2 - b.x1, b.y2 - b.y1
            area = w * h
            if area < BALL_MIN_AREA_PX or area > BALL_MAX_AREA_PX:
                continue
            if h > 0 and not (1.0 / BALL_ASPECT_RATIO_MAX <= w / h <= BALL_ASPECT_RATIO_MAX):
                continue

            bcx, bcy = (b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0

            # HARD reject: centroid right on a face keypoint
            head_hit = False
            for hx, hy in head_reject_pts:
                if (bcx - hx) ** 2 + (bcy - hy) ** 2 < HEAD_REJECT_PX ** 2:
                    head_hit = True
                    break
            if head_hit:
                continue   # never accept this candidate

            # ── Kalman gating: reject detections off the predicted trajectory ──
            # Only gates when the filter is locked (≥2 confirmed hits). Cold
            # start: every detection is accepted as a candidate.
            #
            # Re-acquisition exception: after REACQUIRE_AFTER_FRAMES misses the
            # prediction has drifted and would wrongly reject the ball's real
            # reappearance, so we drop the gate for confident candidates and let
            # the trajectory re-lock around the new detection.
            _missed = self.ball_kf.missed_frames
            _reacquiring = _missed >= self.REACQUIRE_AFTER_FRAMES
            if self.ball_kf.locked and not _reacquiring \
                    and not self.ball_kf.gate((bcx, bcy)):
                continue
            if _reacquiring and float(getattr(b, 'confidence', 0.0)) < self.REACQUIRE_MIN_CONF:
                # During a gap, ignore near-zero-conf blobs so we don't re-lock
                # onto noise — wait for a real detection.
                continue

            score = float(getattr(b, 'confidence', 0.0))

            # Single penalty: implausibly large jump
            if self.last_ball_pos is not None:
                dist_moved = math.hypot(bcx - self.last_ball_pos[0],
                                         bcy - self.last_ball_pos[1])
                if dist_moved > 250:
                    score -= (dist_moved / 500.0)

            is_shoe, in_hand = False, False
            nearest_p, min_p_dist = None, 99999.0
            for p in tracked_players:
                pcx, pcy = p.cx, p.cy
                dist = math.hypot(bcx - pcx, bcy - pcy)
                if dist < min_p_dist:
                    min_p_dist, nearest_p = dist, p
                # Anti-shoe only (lower 15% of player bbox)
                if p.x1 < bcx < p.x2 and bcy > p.y2 - (0.15 * (p.y2 - p.y1)):
                    is_shoe = True
                # Hand-magnet: ball near any wrist
                kp = getattr(p, 'keypoints', None)
                if kp is not None and kp.shape[0] >= 17:
                    for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
                        if (float(kp[idx][2]) >= KP_CONF_MIN and
                            math.hypot(bcx - float(kp[idx][0]),
                                       bcy - float(kp[idx][1])) < HAND_MAGNET_PX):
                            in_hand = True
                            break

            if is_shoe: score += ANTI_SHOE_PENALTY
            if in_hand:
                score += HAND_MAGNET_BONUS
                ball_in_hand = True
                current_holder = nearest_p

            # Airway bonus: ball above the nearest player's shoulders
            if nearest_p is not None:
                kp = getattr(nearest_p, 'keypoints', None)
                if kp is not None and kp.shape[0] >= 17:
                    shoulder_ys = [
                        float(kp[i][1])
                        for i in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
                        if float(kp[i][2]) >= KP_CONF_MIN
                    ]
                    if shoulder_ys and bcy < min(shoulder_ys):
                        score += AIRWAY_BONUS

            # ── Anti-head appearance filter ──
            # A blob that is mostly skin and is NOT in a hand is a head (player
            # OR a fan in the stands — neither has to have pose keypoints). Drop
            # it so the tracker can't lock onto a head while the real ball shows.
            sr = skin_ratio(frame, b.x1, b.y1, b.x2, b.y2)
            if sr >= self.SKIN_HARD_REJECT and not in_hand:
                continue
            if sr >= self.SKIN_SOFT and not in_hand:
                score += self.SKIN_PENALTY
            # Prefer the candidate that actually LOOKS like the recently-seen ball
            # (neutral until the colour model has learned, so cold start is safe).
            score += self.APPEARANCE_W * (
                self._ball_appear.similarity(frame, (b.x1, b.y1, b.x2, b.y2)) - 0.5)

            if score > best_score:
                best_score, best_ball = score, b

        # ── Output the ball + maintain holder grace (Mode-1 phantom only) ──
        if best_ball is not None:
            bx = (best_ball.x1 + best_ball.x2) / 2.0
            by = (best_ball.y1 + best_ball.y2) / 2.0

            # ── Stationary-killer ────────────────────────────────────────
            # If the "best" ball has barely moved from last frame AND there's
            # no player wrist nearby, it's almost certainly a painted floor
            # marker. After STATIONARY_FRAMES_LIMIT frames of this, blacklist
            # the pixel and DROP this candidate.
            is_stationary = False
            if self.last_ball_pos is not None and not ball_in_hand:
                dx = bx - self.last_ball_pos[0]
                dy = by - self.last_ball_pos[1]
                if (dx * dx + dy * dy) <= (self.STATIONARY_PIXEL_TOL ** 2):
                    is_stationary = True

            if is_stationary:
                self._stationary_streak += 1
                if self._stationary_streak >= self.STATIONARY_FRAMES_LIMIT:
                    expire = fn + self.STATIONARY_BLACKLIST_FRAMES
                    self._stationary_blacklist.append((bx, by, expire))
                    self._stationary_streak = 0
                    self.ball_was_hallucinated = True
                    print(f"[Tracker] stationary marker blacklisted at "
                          f"({bx:.0f},{by:.0f})  total={len(self._stationary_blacklist)}")
                    best_ball = None
            else:
                self._stationary_streak = 0

        # ── Anti-teleport guard ──────────────────────────────────────────
        # Reject a single-update jump to a far blob (false positive) unless it
        # CONFIRMS — a candidate reappears near it on the next update. This is
        # what stops the ball "moving to separated places on the wrong route".
        if best_ball is not None and self.last_ball_pos is not None:
            bcx = (best_ball.x1 + best_ball.x2) / 2.0
            bcy = (best_ball.y1 + best_ball.y2) / 2.0
            jump = math.hypot(bcx - self.last_ball_pos[0], bcy - self.last_ball_pos[1])
            if jump > self.BALL_MAX_JUMP_PX:
                pj = self._pending_jump
                if pj is not None and math.hypot(bcx - pj[0], bcy - pj[1]) < self.JUMP_CONFIRM_PX:
                    self._pending_jump = None          # confirmed → accept the move
                else:
                    self._pending_jump = (bcx, bcy)    # first far sighting → hold + reject
                    best_ball = None
                    self.ball_was_hallucinated = True
            else:
                self._pending_jump = None

        if best_ball is not None:
            bx = (best_ball.x1 + best_ball.x2) / 2.0
            by = (best_ball.y1 + best_ball.y2) / 2.0

            # ── Fuse detection into Kalman state ──
            # The filter blends our prediction with the new measurement;
            # detection confidence scales how much we trust it.
            _bconf = float(getattr(best_ball, 'confidence', 0.5))
            self.ball_kf.update((bx, by), conf=_bconf)

            # Learn the ball's colour from confident, free-flight detections (a
            # held ball's box includes the hand, so skip those to keep skin out
            # of the model). This is what lets the appearance bonus prefer the
            # real ball over a head next time.
            if _bconf >= self.APPEARANCE_LEARN_CONF and not ball_in_hand:
                self._ball_appear.update(
                    frame, (best_ball.x1, best_ball.y1, best_ball.x2, best_ball.y2))

            # Use the SMOOTHED Kalman position (stable, less jittery)
            kf_pos = self.ball_kf.position()
            if kf_pos is not None:
                kx, ky = kf_pos
                # Re-centre the bounding box on the Kalman position
                w = best_ball.x2 - best_ball.x1
                h = best_ball.y2 - best_ball.y1
                self.last_ball_pos = (kx, ky)
                emit_x1, emit_y1 = kx - w / 2, ky - h / 2
                emit_x2, emit_y2 = kx + w / 2, ky + h / 2
            else:
                self.last_ball_pos = (bx, by)
                emit_x1, emit_y1 = best_ball.x1, best_ball.y1
                emit_x2, emit_y2 = best_ball.x2, best_ball.y2

            if ball_in_hand:
                self.last_holder = current_holder
                self._holder_grace = self.HOLDER_MEMORY_FRAMES
            else:
                self._holder_grace = max(0, self._holder_grace - 1)
                if self._holder_grace == 0:
                    self.last_holder = None

            self._ball_coast = 0          # real detection → reset coast budget
            self._frames_since_real = 0   # ← real ball seen; phantom budget refills
            ball_track = Track(
                x1=emit_x1, y1=emit_y1,
                x2=emit_x2, y2=emit_y2,
                track_id=self.BALL_TRACK_ID,
                confidence=float(getattr(best_ball, 'confidence', 0.0)),
                class_id=BALL_CLASS,
            )
        else:
            # Ball detection failed. Two BOUNDED fallbacks, in priority order:
            #   1. Hand-anchor at the last holder's wrist  (ball is held).
            #   2. Short Kalman coast along the trajectory  (ball is in flight).
            # Both are capped so we never reproduce the old "ghost flying
            # outside the court" bug (that was UNbounded linear extrapolation).
            self.ball_was_hallucinated = True
            self._frames_since_real += 1
            ball_emitted = False
            # After the phantom budget runs out, show NOTHING (honest gap)
            # rather than a wrong guess.
            _allow_phantom = self._frames_since_real <= self.PHANTOM_MAX_FRAMES
            if self.last_holder is not None:
                self._holder_grace = max(0, self._holder_grace - 1)
                if self._holder_grace == 0:
                    self.last_holder = None
            if _allow_phantom and self.last_holder is not None:
                holder_now = next((p for p in tracked_players
                                    if p.track_id == self.last_holder.track_id), None)
                if (holder_now and holder_now.keypoints is not None
                        and len(holder_now.keypoints) >= 17):
                    lw, rw = (holder_now.keypoints[KP_LEFT_WRIST],
                              holder_now.keypoints[KP_RIGHT_WRIST])
                    best_w = lw if lw[2] > rw[2] else rw
                    if best_w[2] > 0.30:
                        bx, by = float(best_w[0]), float(best_w[1])
                        ball_track = Track(
                            x1=bx - 10, y1=by - 10,
                            x2=bx + 10, y2=by + 10,
                            track_id=self.BALL_TRACK_ID,
                            confidence=0.10, class_id=BALL_CLASS,
                        )
                        self.last_ball_pos = (bx, by)
                        ball_emitted = True
                        self._ball_coast = 0

            # ── Kalman coast (flight bridging) ──
            # No holder → the ball is in the air (just passed / shot). If the
            # filter is locked, emit its predicted position for up to
            # BALL_COAST_MAX frames so the ball keeps tracing smoothly instead
            # of vanishing. Guarded by a frame-bounds check: the instant the
            # prediction would leave the image we stop coasting.
            if _allow_phantom and not ball_emitted and self.ball_kf.locked:
                kf_pos = self.ball_kf.position()
                if kf_pos is not None and self._ball_coast < self.BALL_COAST_MAX:
                    kx, ky = kf_pos
                    fh, fw = frame.shape[:2]
                    if 0.0 <= kx < fw and 0.0 <= ky < fh:
                        self._ball_coast += 1
                        ball_track = Track(
                            x1=kx - 10, y1=ky - 10,
                            x2=kx + 10, y2=ky + 10,
                            track_id=self.BALL_TRACK_ID,
                            confidence=0.08, class_id=BALL_CLASS,
                        )
                        self.last_ball_pos = (kx, ky)
                        ball_emitted = True
            if not ball_emitted:
                self._ball_coast = 0

        return ball_track

    # ──────────────────────────────────────────────────────────────────
    def get_stationary_blacklist(self) -> list:
        """Returns list of (x, y, radius) for currently active stationary blacklist
        spots — for debug visualisation."""
        return [(x, y, self.STATIONARY_BLACKLIST_RADIUS)
                for (x, y, _exp) in getattr(self, "_stationary_blacklist", [])]

    # ──────────────────────────────────────────────────────────────────
    def get_track_features(self) -> dict:
        features = {}
        if hasattr(self.player_tracker, 'tracked_stracks'):
            for track in self.player_tracker.tracked_stracks:
                feat = getattr(track, 'smooth_feat',
                               getattr(track, 'curr_feat', None))
                if feat is not None:
                    features[track.track_id] = feat
        return features
