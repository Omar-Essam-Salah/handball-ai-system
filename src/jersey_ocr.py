"""
Jersey-Number OCR — primary identity anchor (Phase-2, Urgent Task #1).

Why this exists
---------------
Phase-1's color-histogram ReID collapsed when the broadcast featured a yellow
court + yellow jerseys (Sweden vs Brazil) — every player got fingerprinted as
"yellow" and the system handed out the same ID to four people at once
("Clone Army"). The post-mortem mandates a hard primary key: the printed jersey
number. Numbers are unique within a team, survive every camera angle, and
don't get poisoned by court color bleed.

What this module does
---------------------
For every visible player, every K frames, it crops the player's upper torso /
back, runs EasyOCR restricted to digits, and votes the read into a per-track
cache. After enough confident reads converge on the same number, the track is
*locked* to "<team_letter>#<number>". The HSV histogram then becomes a
fallback — used only when no number can be read for that track.

Performance
-----------
EasyOCR is ~30-80 ms per crop on GPU, ~120-250 ms on CPU. We:
    * read each track at most every ``read_every`` infer-frames (default 10),
    * skip tracks that are already locked,
    * batch all due crops into a single ``readtext`` call per pipeline tick
      (EasyOCR amortises model load + transfer over a batch).

Wiring
------
    ocr = JerseyOCR(device="cuda:0")

    # in pipeline.py, on infer frames:
    ocr.feed_tracks(frame, tracks, frame_n)
    for t in tracks:
        num = ocr.locked_number(t.track_id)
        if num is not None:
            t.shirt_number = num     # used by MultiSessionReID as primary key

API
---
    locked_number(tid)   →  Optional[int]   # 0..99, or None if not yet decided
    candidate_votes(tid) →  dict[int, float] # raw votes for debug
    on_track_remap(old_tid, new_tid)        # transfer cache when ReID merges IDs
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np


# ── Crop strategy constants ─────────────────────────────────────────────────
# Where on the player's bounding box to look for the number. Handball numbers
# are large and printed on chest + back, occupying ~30-50% of the torso width.
_CROP_X_MARGIN = 0.18    # trim 18% from each side
_CROP_Y_TOP    = 0.18    # start 18% from top of bbox
_CROP_Y_BOTTOM = 0.55    # end 55% from top of bbox  (upper-mid torso)

# ── Voting / locking policy ─────────────────────────────────────────────────
_NUMBER_RE        = re.compile(r"^\d{1,2}$")   # 1- or 2-digit jersey numbers
_MIN_OCR_CONF     = 0.55     # below this we ignore the read entirely
_LOCK_VOTE_SCORE  = 2.5      # cumulative confidence sum to lock
_LOCK_MARGIN      = 1.5      # winner must lead the runner-up by this much
_VOTE_TTL_FRAMES  = 600      # forget votes that haven't been refreshed in this many frames
_MAX_NUMBER       = 99


class JerseyOCR:
    def __init__(
        self,
        device:        str  = "cuda:0",
        read_every:    int  = 15,    # read each track every N infer-frames
        min_box_h_px:  int  = 60,    # skip players too small to read
        max_concurrent_reads: int = 3,
        async_worker:  bool = True,  # run OCR in a background thread
    ):
        self._device      = device
        self._read_every  = max(2, int(read_every))
        self._min_box_h   = int(min_box_h_px)
        self._max_concurrent = max(1, int(max_concurrent_reads))
        self._async       = bool(async_worker)

        # Lazy-load the reader (EasyOCR is slow to import + warm up)
        self._reader = None
        self._init_failed = False

        # Per-track state
        self._votes:        dict[int, dict[int, float]] = defaultdict(dict)
        self._last_vote_fn: dict[int, int]              = {}
        self._locked:       dict[int, int]              = {}     # tid → number
        self._next_due_fn:  dict[int, int]              = {}     # tid → frame_n

        # Background worker (optional)
        import threading, queue
        self._jobq:    "queue.Queue" = queue.Queue(maxsize=8) if self._async else None
        self._worker_thread = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Stats
        self._reads = 0
        self._locks = 0

    # ──────────────────────────────────────────────────────────────────
    #  Lazy reader + background worker
    # ──────────────────────────────────────────────────────────────────
    def _ensure_reader(self) -> bool:
        if self._reader is not None:
            return True
        if self._init_failed:
            return False
        try:
            import easyocr
            t0 = time.perf_counter()
            self._reader = easyocr.Reader(
                ["en"],
                gpu=("cuda" in str(self._device).lower()),
                verbose=False,
            )
            print(f"[JerseyOCR] reader ready on device={self._device} "
                  f"({time.perf_counter() - t0:.1f}s warmup)")
            if self._async and self._worker_thread is None:
                import threading
                self._worker_thread = threading.Thread(
                    target=self._worker_loop, name="JerseyOCRWorker", daemon=True)
                self._worker_thread.start()
                print("[JerseyOCR] background worker thread started")
            return True
        except Exception as e:
            print(f"[JerseyOCR] init failed: {type(e).__name__}: {e}")
            self._init_failed = True
            return False

    def _worker_loop(self) -> None:
        """Drains the OCR queue serially in a background thread so the main
        pipeline never blocks on EasyOCR latency."""
        import queue as _q
        while not self._stop_event.is_set():
            try:
                job = self._jobq.get(timeout=0.5)
            except _q.Empty:
                continue
            if job is None:
                break
            crop, tid, frame_n = job
            try:
                results = self._reader.readtext(
                    crop,
                    allowlist="0123456789",
                    text_threshold=0.55,
                    low_text=0.30,
                    paragraph=False,
                    detail=1,
                )
                with self._lock:
                    self._absorb(tid, results, frame_n)
            except Exception as e:
                # Single bad call shouldn't kill the worker
                pass
            finally:
                self._jobq.task_done()

    def shutdown(self) -> None:
        """Signal the worker thread to exit. Safe to call multiple times."""
        self._stop_event.set()
        if self._jobq is not None:
            try:
                self._jobq.put_nowait(None)
            except Exception:
                pass
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)

    # ──────────────────────────────────────────────────────────────────
    #  Public ingestion API
    # ──────────────────────────────────────────────────────────────────
    def feed_tracks(self, frame: np.ndarray, tracks, frame_n: int) -> None:
        """Read jersey numbers for tracks that are due. Cheap & batched."""
        if not self._ensure_reader():
            return
        if frame is None or frame.size == 0:
            return

        H, W = frame.shape[:2]

        # Pick at most ``max_concurrent`` due, non-locked, large-enough tracks.
        # Phase-5: ONLY pass keyframes — tracks where the player is clearly
        # readable. This mirrors the mkoshkina jersey-number-pipeline insight:
        # quality > quantity. A few good reads beat 50 noisy ones.
        due = []
        for t in tracks:
            try:
                cls = int(getattr(t, "class_id", 0))
            except Exception:
                continue
            if cls != 0:                         # only persons
                continue
            tid = int(getattr(t, "track_id", -1))
            if tid < 0:
                continue
            if tid in self._locked:              # already decided
                continue
            bh = float(t.y2) - float(t.y1)
            if bh < self._min_box_h:
                continue
            # KEYFRAME CHECK 1: player fully inside frame (not cut off at edges)
            bw = float(t.x2) - float(t.x1)
            if (t.x1 < 4 or t.y1 < 4
                or t.x2 > W - 4 or t.y2 > H - 4):
                continue
            # KEYFRAME CHECK 2: aspect ratio sane (running players are tall;
            # crawled / fallen players are square or wider — bad OCR signal)
            if bh < bw * 1.3:
                continue
            # KEYFRAME CHECK 3: pose keypoints for shoulders+hips visible
            # → guarantees a clean upper-torso crop
            kp = getattr(t, "keypoints", None)
            if isinstance(kp, np.ndarray) and kp.shape[0] >= 17:
                conf_shoulder = max(float(kp[5][2]), float(kp[6][2]))
                if conf_shoulder < 0.35:
                    continue
            next_due = self._next_due_fn.get(tid, 0)
            if frame_n < next_due:
                continue
            due.append(t)

        if not due:
            self._gc_old_votes(frame_n)
            return

        # Score-by-confidence: read tracks that look biggest first (better OCR)
        due.sort(key=lambda t: -(float(t.y2) - float(t.y1)))
        due = due[:self._max_concurrent]

        # Prepare crops
        crops = []
        crop_to_tid = []
        for t in due:
            crop = self._crop(frame, t, H, W)
            if crop is None:
                # Mark "not due again" briefly so we don't retry instantly
                self._next_due_fn[int(t.track_id)] = frame_n + self._read_every
                continue
            crops.append(crop)
            crop_to_tid.append(int(t.track_id))

        if not crops:
            self._gc_old_votes(frame_n)
            return

        # Cool-down for everyone we're enqueueing (success or not, prevents thrash)
        for tid in crop_to_tid:
            self._next_due_fn[tid] = frame_n + self._read_every

        self._reads += len(crops)
        if self._async and self._jobq is not None:
            # Hand off to background thread — never blocks the main pipeline
            import queue as _q
            for crop, tid in zip(crops, crop_to_tid):
                try:
                    self._jobq.put_nowait((crop, tid, frame_n))
                except _q.Full:
                    # Drop this read — worker is overloaded; we'll retry next cycle
                    break
        else:
            # Synchronous fallback (only used when async_worker=False)
            try:
                for crop, tid in zip(crops, crop_to_tid):
                    results = self._reader.readtext(
                        crop,
                        allowlist="0123456789",
                        text_threshold=0.55,
                        low_text=0.30,
                        paragraph=False,
                        detail=1,
                    )
                    self._absorb(tid, results, frame_n)
            except Exception as e:
                print(f"[JerseyOCR] readtext failed: {type(e).__name__}: {e}")

        with self._lock:
            self._gc_old_votes(frame_n)

    def on_track_remap(self, old_tid: int, new_tid: int) -> None:
        """When PlayerReID merges two track IDs, transfer the OCR votes too."""
        if old_tid == new_tid:
            return
        with self._lock:
            if old_tid in self._locked:
                self._locked.setdefault(new_tid, self._locked.pop(old_tid))
            old_v = self._votes.pop(old_tid, None)
            if old_v:
                tgt = self._votes[new_tid]
                for num, score in old_v.items():
                    tgt[num] = tgt.get(num, 0.0) + score

    # ──────────────────────────────────────────────────────────────────
    #  Read-back API
    # ──────────────────────────────────────────────────────────────────
    def locked_number(self, track_id: int) -> Optional[int]:
        with self._lock:
            return self._locked.get(int(track_id))

    def candidate_votes(self, track_id: int) -> dict[int, float]:
        with self._lock:
            return dict(self._votes.get(int(track_id), {}))

    def stats(self) -> dict:
        with self._lock:
            return {
                "reads":       self._reads,
                "locks":       self._locks,
                "locked":      dict(self._locked),
                "tracked":     len(self._votes),
            }

    # ──────────────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────────────
    def _crop(self, frame: np.ndarray, t, H: int, W: int) -> Optional[np.ndarray]:
        """Cut the upper-mid torso out of a track's bbox.
        Pose keypoints (shoulders + hips) refine the crop when available."""
        bx1, by1 = max(0, int(t.x1)), max(0, int(t.y1))
        bx2, by2 = min(W, int(t.x2)), min(H, int(t.y2))
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 16 or bh < 32:
            return None

        kpts = getattr(t, "keypoints", None)
        if kpts is not None and isinstance(kpts, np.ndarray) and kpts.shape[0] >= 13:
            xs, ys = [], []
            for idx in (5, 6, 11, 12):
                if float(kpts[idx][2]) > 0.40:
                    xs.append(float(kpts[idx][0]))
                    ys.append(float(kpts[idx][1]))
            if len(xs) >= 3:
                tx1, tx2 = int(min(xs)), int(max(xs))
                ty1 = int(min(ys[:2])) if len(ys) >= 2 else int(min(ys))
                ty2 = int(max(ys))
                # widen slightly + clamp
                pad_x = max(8, int((tx2 - tx1) * 0.15))
                tx1 = max(0, tx1 - pad_x)
                tx2 = min(W, tx2 + pad_x)
                # take the upper torso (shoulders → mid hips)
                ty2 = ty1 + int((ty2 - ty1) * 0.65)
                if tx2 - tx1 >= 24 and ty2 - ty1 >= 28:
                    return frame[ty1:ty2, tx1:tx2]

        # bbox-relative fallback
        cx1 = bx1 + int(bw * _CROP_X_MARGIN)
        cx2 = bx2 - int(bw * _CROP_X_MARGIN)
        cy1 = by1 + int(bh * _CROP_Y_TOP)
        cy2 = by1 + int(bh * _CROP_Y_BOTTOM)
        if cx2 - cx1 < 24 or cy2 - cy1 < 28:
            return None
        return frame[cy1:cy2, cx1:cx2]

    def _absorb(self, tid: int, ocr_results, frame_n: int) -> None:
        """``ocr_results`` is the EasyOCR ``[(bbox, text, conf), ...]`` list.
        Each digit reading votes into the per-track tally. After enough
        cumulative confidence on a single number, lock it.

        UNIQUE-NUMBER CONSTRAINT: a jersey number can only be locked to ONE
        track at a time. If two tracks both vote heavily for the same number,
        only the strongest holder gets the lock — the loser is told to look
        for another number. This kills the "Clone Army" effect where multiple
        players ended up labelled as #1.
        """
        if not ocr_results:
            return
        tally = self._votes[tid]
        for entry in ocr_results:
            try:
                _bbox, text, conf = entry
            except Exception:
                continue
            text = (text or "").strip()
            if not _NUMBER_RE.match(text):
                continue
            try:
                num = int(text)
            except ValueError:
                continue
            if num <= 0 or num > _MAX_NUMBER:
                continue
            c = float(conf)
            if c < _MIN_OCR_CONF:
                continue
            tally[num] = tally.get(num, 0.0) + c

        self._last_vote_fn[tid] = frame_n

        # Try to lock
        if not tally:
            return
        ranked = sorted(tally.items(), key=lambda kv: -kv[1])
        top_num, top_score = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_score < _LOCK_VOTE_SCORE or (top_score - runner_score) < _LOCK_MARGIN:
            return

        # ── Unique-number enforcement ──
        # Is another track already locked on this number?
        existing_holder = next(
            (other_tid for other_tid, num in self._locked.items()
             if num == top_num and other_tid != tid),
            None,
        )
        if existing_holder is None:
            self._locked[tid] = top_num
            self._locks += 1
            return

        # Conflict: pick the track with the higher cumulative score for this number
        existing_score = self._votes.get(existing_holder, {}).get(top_num, 0.0)
        if top_score > existing_score * 1.25:
            # We win — kick the existing holder out
            self._locked.pop(existing_holder, None)
            self._locked[tid] = top_num
            self._locks += 1
            # Make the loser stop voting for this number — clear it from their tally
            other_tally = self._votes.get(existing_holder)
            if other_tally is not None:
                other_tally.pop(top_num, None)
        else:
            # We lose — drop this number from our own tally so we look for another
            tally.pop(top_num, None)
            self._locks += 1

    def _gc_old_votes(self, frame_n: int) -> None:
        """Drop track caches that haven't seen activity in a long time."""
        if not self._last_vote_fn:
            return
        cutoff = frame_n - _VOTE_TTL_FRAMES
        stale = [tid for tid, fn in self._last_vote_fn.items() if fn < cutoff]
        for tid in stale:
            self._votes.pop(tid, None)
            self._last_vote_fn.pop(tid, None)
            self._next_due_fn.pop(tid, None)
            # Don't drop locks — they're long-lived identity decisions
