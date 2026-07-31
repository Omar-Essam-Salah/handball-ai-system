"""
Track Bridge — permanent player identity across any occlusion.

Problem
───────
BotSort assigns a new integer track_id every time a player re-enters
the frame after being occluded or out-of-frame.  A player who exits for
2 seconds returns as a completely different track.  Every label, stat,
and recognition result resets — the "glitch" the user sees.

Solution
────────
The TrackBridge sits between BotSort and the rest of the pipeline.
It maintains a "ghost" for every player seen in the last MAX_INVISIBLE
frames.  When BotSort introduces a new track_id, the bridge tries to
match it to an existing ghost using FIVE independent cues:

  1. Spatial proximity + kinematic prediction    weight 0.35
     Weighted velocity regression over VEL_HISTORY positions →
     predicted position with per-frame decay for long occlusions.
     Adaptive search radius grows with gap length.

  2. ReID embedding cosine similarity            weight 0.35
     Uses BotSort's already-computed embeddings (zero extra GPU cost).
     EMA update is GATED: skipped if detection confidence is low or
     the player is heavily overlapped by another box.

  3. Jersey color histogram                      weight 0.20
     16-bin HSV hue histogram of the torso crop.

  4. Body proportion (aspect ratio)              weight 0.10
     Height/width ratio.

  5. Conditional OCR (pre-pass, highest priority)
     PaddleOCR reads the jersey number ONLY when the bbox area exceeds
     OCR_BBOX_AREA_THRESH (player is close to camera).  A confident
     OCR hit force-assigns the slot_id, bypassing cues 1-4 entirely.

Output
──────
bridge.update() returns {track_id → slot_id}.
"""

import math
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── Tuning ────────────────────────────────────────────────────────────────────
MAX_INVISIBLE    = 90    # frames to keep a ghost alive  (~3 s at 30 fps)
MATCH_THRESHOLD  = 0.45  # combined score to accept a re-identification
SPATIAL_RADIUS   = 180   # px — base search radius at gap == 1
SPATIAL_W        = 0.35
REID_W           = 0.35
COLOR_W          = 0.20
PROP_W           = 0.10
MIN_EMBED_NORM   = 0.05  # reject near-zero embedding vectors
VEL_HISTORY      = 12    # positions used for weighted velocity regression
COLOR_BINS       = 16

# FAISS direct re-id
REID_DIRECT_THRESH = 0.65
REID_DIRECT_TOPK   = 3
REID_MIN_SAMPLES   = 5

# ── OCR (Conditional — only for large/close bboxes) ──────────────────────────
OCR_BBOX_AREA_THRESH = 6_000   # px²  e.g. ≈ 80 × 75 px bounding box
OCR_CONF_THRESH      = 0.60    # PaddleOCR text confidence floor
OCR_UPSCALE_MIN_DIM  = 64      # upscale small number crops before OCR

# ── Embedding quality gates ───────────────────────────────────────────────────
# Prevent database corruption from low-quality observations.
EMBED_MIN_DET_CONF = 0.65   # skip appearance EMA if BotSort conf is below this
EMBED_MAX_IOC      = 0.30   # skip if another box covers > 30 % of this one
                             # (Intersection-over-self — occlusion proxy)

# ── Kinematic prediction ──────────────────────────────────────────────────────
PRED_LINEAR_FRAMES   = 6    # pure linear extrapolation up to this gap
PRED_VELOCITY_DECAY  = 0.92 # per-frame velocity retention beyond linear horizon
SPATIAL_RADIUS_GROWTH = 8.0 # extra px/frame added to search radius past horizon


@dataclass
class Ghost:
    """One persistent player identity slot."""
    slot_id:    int
    track_id:   int

    positions:  deque = field(default_factory=lambda: deque(maxlen=VEL_HISTORY))
    vx:         float = 0.0
    vy:         float = 0.0

    embedding:    Optional[np.ndarray] = None
    color_hist:   Optional[np.ndarray] = None
    aspect_ratio: float = 0.0

    team:       str = "unknown"
    player_id:  str = ""
    jersey_num: int = 0        # 0 = unknown; set via register_jersey()

    last_seen:  int  = 0
    invisible:  int  = 0
    confirmed:  bool = False
    seen_count: int  = 0


class TrackBridge:
    """
    Maintains permanent slot_ids across BotSort track_id changes.

    Usage
    ─────
        bridge = TrackBridge()
        bridge.set_player_db(player_db)
        bridge.register_jersey(slot_id=1, jersey_num=7)   # optional

        # each frame:
        slot_map = bridge.update(tracks, track_feats, frame, frame_n)
        label = player_labels.get(slot_map[t.track_id], "?")
    """

    MIN_CONFIRM = 6

    def __init__(self,
                 ocr_area_thresh: int = OCR_BBOX_AREA_THRESH,
                 enable_ocr: bool = True):
        self._ghosts:    dict[int, Ghost] = {}
        self._tid_slot:  dict[int, int]   = {}
        self._next_slot: int              = 1

        self._player_db                    = None
        self._pid_to_slot: dict[str, int]  = {}
        self._jersey_to_slot: dict[int, int] = {}  # jersey_num → slot_id
        self._slot_to_jersey: dict[int, int] = {}  # slot_id   → jersey_num

        self._ocr              = None
        self._ocr_area_thresh  = ocr_area_thresh

        if enable_ocr:
            self._init_ocr()

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _init_ocr(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr = PaddleOCR(
                use_angle_cls=False, lang="en",
                use_gpu=False, show_log=False,
                det_db_thresh=0.3, rec_thresh=0.5,
            )
            print("[TrackBridge] PaddleOCR initialised.")
        except ImportError:
            print("[TrackBridge] paddleocr not installed — OCR disabled.")
        except Exception as e:
            print(f"[TrackBridge] OCR init failed: {e}")

    def set_player_db(self, player_db) -> None:
        self._player_db = player_db

    def register_jersey(self, slot_id: int, jersey_num: int) -> None:
        """Bind a jersey number to a permanent slot for OCR-based identification."""
        self._jersey_to_slot[jersey_num] = slot_id
        self._slot_to_jersey[slot_id]    = jersey_num
        g = self._ghosts.get(slot_id)
        if g:
            g.jersey_num = jersey_num

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        tracks:      list,
        track_feats: dict[int, np.ndarray],
        frame:       np.ndarray,
        frame_n:     int,
    ) -> dict[int, int]:
        person_tracks = [t for t in tracks if not t.is_ball]
        self._tid_slot = {}

        # ── 0. OCR pre-pass — force-assign close/large players ───────────────
        ocr_forced_tids  : set[int] = set()
        ocr_forced_slots : set[int] = set()

        for t in person_tracks:
            sid = self._try_ocr_reid(frame, t)
            if sid is None:
                continue
            # Resurrect ghost if it expired
            if sid not in self._ghosts:
                g = Ghost(slot_id=sid, track_id=t.track_id, last_seen=frame_n)
                g.confirmed   = True
                g.jersey_num  = self._slot_to_jersey.get(sid, 0)
                self._ghosts[sid] = g
            self._ghosts[sid].invisible  = 0
            self._ghosts[sid].track_id   = t.track_id
            self._tid_slot[t.track_id]   = sid
            ocr_forced_tids.add(t.track_id)
            ocr_forced_slots.add(sid)

        # ── 1. Score remaining (track × ghost) pairs ─────────────────────────
        candidates: list[tuple[float, int, int]] = []

        for t in person_tracks:
            if t.track_id in ocr_forced_tids:
                continue
            feat  = track_feats.get(t.track_id)
            color = _extract_color_hist(frame, t)
            prop  = (t.y2 - t.y1) / max(t.x2 - t.x1, 1)

            for ghost in self._ghosts.values():
                if ghost.slot_id in ocr_forced_slots:
                    continue
                if ghost.track_id == t.track_id and ghost.invisible == 0:
                    candidates.append((1.0, t.track_id, ghost.slot_id))
                    continue
                score = self._score(t, feat, color, prop, ghost, frame_n)
                if score > 0.0:
                    candidates.append((score, t.track_id, ghost.slot_id))

        # ── 2. Greedy assignment ──────────────────────────────────────────────
        assigned_tids  = set(ocr_forced_tids)
        assigned_slots = set(ocr_forced_slots)

        for score, tid, sid in sorted(candidates, reverse=True):
            if tid in assigned_tids or sid in assigned_slots:
                continue
            if score >= MATCH_THRESHOLD:
                self._tid_slot[tid] = sid
                assigned_tids.add(tid)
                assigned_slots.add(sid)

        # ── 3. Unmatched → FAISS fallback → new ghost ────────────────────────
        active_sids_so_far = set(self._tid_slot.values())
        for t in person_tracks:
            if t.track_id in self._tid_slot:
                continue
            feat  = track_feats.get(t.track_id)
            color = _extract_color_hist(frame, t)

            sid = self._try_faiss_reid(feat, active_sids_so_far)
            if sid is not None:
                active_sids_so_far.add(sid)
                if sid not in self._ghosts:
                    g = Ghost(slot_id=sid, track_id=t.track_id, last_seen=frame_n)
                    g.confirmed = True
                    self._ghosts[sid] = g
                self._ghosts[sid].invisible = 0
                self._ghosts[sid].track_id  = t.track_id
            else:
                sid = self._new_ghost(t, feat, color, frame_n)
                active_sids_so_far.add(sid)

            self._tid_slot[t.track_id] = sid

        # ── 4. Update matched ghosts (with appearance quality gate) ──────────
        active_slots = set(self._tid_slot.values())
        for t in person_tracks:
            sid   = self._tid_slot[t.track_id]
            ghost = self._ghosts[sid]
            feat  = track_feats.get(t.track_id)
            color = _extract_color_hist(frame, t)
            prop  = (t.y2 - t.y1) / max(t.x2 - t.x1, 1)

            det_conf     = float(getattr(t, "conf", 1.0))
            occluded     = _is_heavily_occluded(t, person_tracks)
            update_appear = (det_conf >= EMBED_MIN_DET_CONF) and (not occluded)

            self._refresh_ghost(ghost, t, feat, color, prop,
                                frame_n, update_appear)

        # ── 5. Age invisible ghosts ───────────────────────────────────────────
        stale = []
        for sid, g in self._ghosts.items():
            if sid not in active_slots:
                g.invisible += 1
                if g.invisible > MAX_INVISIBLE:
                    stale.append(sid)
                    
        for sid in stale:
            del self._ghosts[sid]

        return dict(self._tid_slot)
    # ── Public helpers ────────────────────────────────────────────────────────

    def slot_to_track(self) -> dict[int, int]:
        return {sid: tid for tid, sid in self._tid_slot.items()}

    def get_ghost(self, slot_id: int) -> Optional[Ghost]:
        return self._ghosts.get(slot_id)

    def link_player(self, slot_id: int, player_id: str) -> None:
        g = self._ghosts.get(slot_id)
        if g:
            g.player_id = player_id
        self._pid_to_slot[player_id] = slot_id

    def all_active_slots(self) -> list[int]:
        return list(self._tid_slot.values())

    def invisible_ghosts(self) -> list[Ghost]:
        active = set(self._tid_slot.values())
        return [g for g in self._ghosts.values()
                if g.slot_id not in active and g.confirmed]

    # ── OCR re-identification ─────────────────────────────────────────────────

    def _try_ocr_reid(self, frame: np.ndarray, t) -> Optional[int]:
        """
        Run jersey-number OCR when the player's bbox is large enough to read.
        Returns a registered slot_id on a confident digit hit, else None.
        """
        if self._ocr is None or not self._jersey_to_slot:
            return None

        area = (t.x2 - t.x1) * (t.y2 - t.y1)
        if area < self._ocr_area_thresh:
            return None

        fh, fw = frame.shape[:2]
        x1 = int(max(t.x1, 0));  y1 = int(max(t.y1, 0))
        x2 = int(min(t.x2, fw)); y2 = int(min(t.y2, fh))
        bh = y2 - y1;            bw = x2 - x1
        if bw < 8 or bh < 12:
            return None

        # Jersey number lives in the upper chest region
        ny1 = y1 + int(bh * 0.10)
        ny2 = y1 + int(bh * 0.45)
        crop = frame[ny1:ny2, x1:x2]
        if crop.size == 0:
            return None

        # Upscale tiny crops so OCR has enough pixels to work with
        h_c, w_c = crop.shape[:2]
        if h_c < OCR_UPSCALE_MIN_DIM or w_c < OCR_UPSCALE_MIN_DIM:
            scale = max(OCR_UPSCALE_MIN_DIM / h_c, OCR_UPSCALE_MIN_DIM / w_c)
            crop  = cv2.resize(crop, (0, 0), fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        try:
            results = self._ocr.ocr(crop, cls=False)
        except Exception:
            return None

        if not results or not results[0]:
            return None

        for line in results[0]:
            if not line or len(line) < 2:
                continue
            text_conf = line[1]
            if not text_conf or len(text_conf) < 2:
                continue
            text, conf = str(text_conf[0]), float(text_conf[1])
            if conf < OCR_CONF_THRESH:
                continue
            digits = "".join(c for c in text if c.isdigit())
            if len(digits) not in (1, 2):
                continue
            num = int(digits)
            if not 1 <= num <= 99:
                continue
            slot = self._jersey_to_slot.get(num)
            if slot is not None:
                return slot

        return None

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(
        self,
        t,
        feat:   Optional[np.ndarray],
        color:  Optional[np.ndarray],
        prop:   float,
        ghost:  Ghost,
        frame_n: int,
    ) -> float:
        score = 0.0

        # ── Kinematic prediction with adaptive radius ─────────────────────────
        gap = frame_n - ghost.last_seen
        if gap <= MAX_INVISIBLE and ghost.positions:
            pred_x, pred_y = _predict_position(ghost, gap)
            dist = math.hypot(t.cx - pred_x, t.cy - pred_y)

            # Search radius grows beyond the linear horizon to account for
            # increasing prediction uncertainty during long occlusions.
            adaptive_r = (SPATIAL_RADIUS
                          + SPATIAL_RADIUS_GROWTH
                            * max(0, gap - PRED_LINEAR_FRAMES))
            if dist > adaptive_r * 2:
                return 0.0
            spatial_score = max(0.0, 1.0 - dist / adaptive_r)
            score += SPATIAL_W * spatial_score
        else:
            return 0.0

        # ── ReID embedding ───────────────────────────────────────────────────
        if feat is not None and ghost.embedding is not None:
            norm = float(np.linalg.norm(feat))
            if norm >= MIN_EMBED_NORM:
                sim = float(np.dot(feat / norm, ghost.embedding))
                score += REID_W * max(0.0, sim)

        # ── Jersey color ─────────────────────────────────────────────────────
        if color is not None and ghost.color_hist is not None:
            c_score = float(cv2.compareHist(
                color, ghost.color_hist, cv2.HISTCMP_CORREL))
            score += COLOR_W * max(0.0, c_score)

        # ── Body proportion ──────────────────────────────────────────────────
        if ghost.aspect_ratio > 0 and prop > 0:
            diff = abs(prop - ghost.aspect_ratio) / max(prop, ghost.aspect_ratio)
            score += PROP_W * max(0.0, 1.0 - diff * 3.0)

        # ── Team consistency bonus ───────────────────────────────────────────
        if ghost.team == t.team != "unknown":
            score = min(1.0, score + 0.05)

        return score

    # ── Ghost lifecycle ───────────────────────────────────────────────────────

    def _try_faiss_reid(
        self,
        feat:        Optional[np.ndarray],
        active_sids: set,
    ) -> Optional[int]:
        if feat is None or self._player_db is None:
            return None
        try:
            results = self._player_db.search_by_embedding(
                feat, k=16, min_samples=REID_MIN_SAMPLES)
            if not results:
                return None

            by_pid: dict[str, list[float]] = {}
            for pid, sim in results:
                if sim > 0:
                    by_pid.setdefault(pid, []).append(float(sim))
            if not by_pid:
                return None

            def _agg(sims: list[float]) -> float:
                return float(np.mean(sorted(sims, reverse=True)[:REID_DIRECT_TOPK]))

            best_pid   = max(by_pid, key=lambda p: _agg(by_pid[p]))
            best_score = _agg(by_pid[best_pid])
            if best_score < REID_DIRECT_THRESH:
                return None

            slot = self._pid_to_slot.get(best_pid)
            if slot is None or slot in active_sids:
                return None
            return slot
        except Exception:
            return None

    def _new_ghost(self, t, feat, color, frame_n: int) -> int:
        sid   = self._next_slot
        self._next_slot += 1
        ghost = Ghost(slot_id=sid, track_id=t.track_id, last_seen=frame_n)
        self._ghosts[sid] = ghost
        self._refresh_ghost(ghost, t, feat, color,
                            (t.y2 - t.y1) / max(t.x2 - t.x1, 1),
                            frame_n, update_appear=True)
        return sid

    def _refresh_ghost(
        self,
        ghost: Ghost,
        t,
        feat:         Optional[np.ndarray],
        color:        Optional[np.ndarray],
        prop:         float,
        frame_n:      int,
        update_appear: bool = True,
    ) -> None:
        """
        Update ghost state with a new observation.
        update_appear=False skips embedding/color/ratio EMA updates to prevent
        database corruption from low-confidence or occluded detections.
        The initial embedding assignment is always allowed regardless of the flag.
        """
        ghost.track_id   = t.track_id
        ghost.team       = t.team if t.team != "unknown" else ghost.team
        ghost.invisible  = 0
        ghost.last_seen  = frame_n
        ghost.seen_count += 1
        if ghost.seen_count >= self.MIN_CONFIRM:
            ghost.confirmed = True

        # ── Position + velocity (always updated — kinematics are not appearance)
        ghost.positions.append((t.cx, t.cy, frame_n))
        _update_velocity(ghost)

        # ── Embedding ─────────────────────────────────────────────────────────
        if feat is not None:
            norm = float(np.linalg.norm(feat))
            if norm >= MIN_EMBED_NORM:
                v = feat / norm
                if ghost.embedding is None:
                    # Always initialise — having no embedding is worse than a
                    # slightly noisy one.
                    ghost.embedding = v.copy()
                elif update_appear:
                    blended = 0.75 * ghost.embedding + 0.25 * v
                    n_b     = float(np.linalg.norm(blended))
                    ghost.embedding = (blended / max(n_b, 1e-8)).astype(np.float32)

        # ── Colour + aspect ratio (gated by update_appear) ────────────────────
        if not update_appear:
            return

        if color is not None:
            if ghost.color_hist is None:
                ghost.color_hist = color.copy()
            else:
                hist = 0.80 * ghost.color_hist + 0.20 * color
                s    = hist.sum()
                ghost.color_hist = (hist / s if s > 0 else hist).astype(np.float32)

        if prop > 0:
            ghost.aspect_ratio = (0.85 * ghost.aspect_ratio + 0.15 * prop
                                  if ghost.aspect_ratio > 0 else prop)


# ── Kinematic helpers ─────────────────────────────────────────────────────────

def _update_velocity(ghost: Ghost) -> None:
    """
    Recompute ghost.vx / ghost.vy using exponentially-weighted linear
    regression over the full position history.  More recent positions carry
    higher weight, making the estimate stable but responsive.
    """
    n = len(ghost.positions)
    if n < 2:
        return

    positions = list(ghost.positions)

    if n == 2:
        p1, p2 = positions[-2], positions[-1]
        df     = max(p2[2] - p1[2], 1)
        raw_vx = (p2[0] - p1[0]) / df
        raw_vy = (p2[1] - p1[1]) / df
    else:
        times = np.array([p[2] for p in positions], dtype=np.float64)
        xs    = np.array([p[0] for p in positions], dtype=np.float64)
        ys    = np.array([p[1] for p in positions], dtype=np.float64)

        # Exponential weights: index 0 is oldest (weight ≈ e^-2), last is 1.0
        w     = np.exp(np.linspace(-2.0, 0.0, n))
        t_bar = np.average(times, weights=w)
        x_bar = np.average(xs,    weights=w)
        y_bar = np.average(ys,    weights=w)
        dt    = times - t_bar
        wt2   = float(np.dot(w, dt ** 2))

        if wt2 > 1e-8:
            raw_vx = float(np.dot(w * dt, xs - x_bar) / wt2)
            raw_vy = float(np.dot(w * dt, ys - y_bar) / wt2)
        else:
            raw_vx, raw_vy = ghost.vx, ghost.vy

    # EMA smoothing keeps large single-frame spikes from dominating
    ghost.vx = 0.60 * ghost.vx + 0.40 * raw_vx
    ghost.vy = 0.60 * ghost.vy + 0.40 * raw_vy


def _predict_position(ghost: Ghost, gap: int) -> tuple[float, float]:
    """
    Extrapolate ghost position `gap` frames into the future.

    Strategy:
      gap ≤ PRED_LINEAR_FRAMES   → pure linear (constant velocity)
      gap > PRED_LINEAR_FRAMES   → velocity decays by PRED_VELOCITY_DECAY
                                   per frame, so the ghost drifts to a stop
                                   rather than flying off the pitch.

    This models the reality that a player who is invisible for many frames is
    more likely to have slowed down than to have continued at full sprint.
    """
    last_x, last_y, _ = ghost.positions[-1]
    vx, vy = ghost.vx, ghost.vy

    if gap <= PRED_LINEAR_FRAMES:
        return last_x + vx * gap, last_y + vy * gap

    # Linear phase
    x = last_x + vx * PRED_LINEAR_FRAMES
    y = last_y + vy * PRED_LINEAR_FRAMES
    # Decaying phase
    cur_vx, cur_vy = vx, vy
    for _ in range(gap - PRED_LINEAR_FRAMES):
        cur_vx *= PRED_VELOCITY_DECAY
        cur_vy *= PRED_VELOCITY_DECAY
        x += cur_vx
        y += cur_vy
    return x, y


def _is_heavily_occluded(t, all_tracks: list) -> bool:
    """
    Return True if another detection's bounding box covers more than
    EMBED_MAX_IOC of t's area (Intersection-over-Self).
    Used to gate appearance updates for partially-hidden players.
    """
    area_t = max((t.x2 - t.x1) * (t.y2 - t.y1), 1)
    for other in all_tracks:
        if other.track_id == t.track_id:
            continue
        ix1 = max(t.x1, other.x1); iy1 = max(t.y1, other.y1)
        ix2 = min(t.x2, other.x2); iy2 = min(t.y2, other.y2)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / area_t > EMBED_MAX_IOC:
                return True
    return False


# ── Colour signature extraction ───────────────────────────────────────────────

def _extract_color_hist(frame: np.ndarray, t) -> Optional[np.ndarray]:
    """
    16-bin normalised hue histogram from the player's torso region.
    Torso = top 55 % of bbox, centre 20-80 % horizontally.
    """
    try:
        fh, fw = frame.shape[:2]
        x1 = int(max(t.x1, 0)); y1 = int(max(t.y1, 0))
        x2 = int(min(t.x2, fw)); y2 = int(min(t.y2, fh))
        if x2 - x1 < 8 or y2 - y1 < 12:
            return None

        bh = y2 - y1
        bw = x2 - x1
        ty1 = y1 + int(bh * 0.08); ty2 = y1 + int(bh * 0.55)
        tx1 = x1 + int(bw * 0.20); tx2 = x1 + int(bw * 0.80)

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [COLOR_BINS], [0, 180])
        hist = hist.astype(np.float32).flatten()
        s    = hist.sum()
        if s > 0:
            hist /= s
        return hist
    except Exception:
        return None
