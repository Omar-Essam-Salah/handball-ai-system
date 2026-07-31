"""
VideoEngine — threaded RTSP pipeline for the live demo.

Three threads, producer-consumer:
    1. CAPTURE   : cv2.VideoCapture → queue.Queue(maxsize=30)
                   drops the OLDEST frame when full so live lag never grows.
    2. INFERENCE : pulls frames; runs YOLOv8 + PlayerTracker (BoT-SORT) on
                   GPU every Nth frame; for the in-between frames, propagates
                   the last bounding boxes with sparse Lucas-Kanade optical
                   flow (single CPU call).  Track IDs from BoT-SORT travel
                   with each box during propagation, so identity stays stable
                   across the full display rate.
    3. DISPLAY   : reads a maxlen=1 mailbox at TARGET_FPS and renders the
                   annotated frame plus the bird's-eye minimap.

Oblique-camera notes (camera on a tripod in the stands):
    • Player position on the 2D minimap is computed from the bbox
      BOTTOM-CENTER (the feet), NEVER the bbox center.  See _draw_minimap().
    • Homography is loaded from ./homography_matrix.npy on startup, produced
      by the pre-match calibrate_court.py tool.  Without it the minimap is
      hidden but tracking still works.
    • PlayerTracker (player_tracker.py) is tuned for high inter-player
      occlusion: track_buffer=90 frames, match_thresh=0.85.
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ── Import AI Analyst & Logger ───────────────────────────────────────────────
from match_logger import MatchLogger
from ollama_analyst import OllamaAnalyst


# RTSP over TCP — set before cv2.VideoCapture() opens any stream.
# Hardware decode (NVDEC) is opted in via the --hwaccel CLI flag below; we
# don't enable it by default because some opencv-python wheels ship an
# FFmpeg without CUDA support and would silently fall back anyway.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Bird's-eye minimap dimensions — must match calibrate_court.py.
COURT_W_PX = 400
COURT_H_PX = 200


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float
    seq: int


@dataclass
class TrackedBox:
    """One propagated/detected bounding box with a stable ID."""
    track_id:   int
    x1: float; y1: float; x2: float; y2: float
    conf:       float = 0.0
    class_id:   int   = 0

    @property
    def foot_xy(self) -> tuple[float, float]:
        """Bottom-center pixel — used for homography projection.

        The camera is oblique, so the bbox CENTER projects to a point
        BEHIND the player.  Always anchor on the feet (y_max).
        """
        return ((self.x1 + self.x2) * 0.5, self.y2)


@dataclass
class AnnotatedFrame:
    frame: np.ndarray
    tracks: list[TrackedBox] = field(default_factory=list)
    seq: int = 0
    inferred: bool = False
    yolo_age_frames: int = 0
    timestamp: float = 0.0


# ── engine ───────────────────────────────────────────────────────────────────

class VideoEngine:
    def __init__(
        self,
        source:        str,
        model_path:    str               = "yolov8n.pt",
        device:        str               = "cuda:0",
        conf:          float             = 0.25,
        imgsz:         int                = 640,
        infer_every:   int                = 3,
        target_fps:    int                = 30,
        queue_size:    int                = 30,
        classes:       Optional[list[int]] = None,
        homography:    Optional[Path]    = Path("homography_matrix.npy"),
        track_buffer:  int                = 90,
        match_thresh:  float             = 0.85,
        use_tracker:   bool               = True,
        # ── performance knobs ────────────────────────────────────────────────
        half:          Optional[bool]    = None,    # FP16 inference (auto on GPU)
        display_scale: float             = 1.0,     # render at fraction of frame size
        adaptive:      bool              = True,    # auto-relax infer_every under load
        hwaccel:       bool              = False,   # NVDEC via FFmpeg
        window_name:   str                = "VideoEngine",
        show_window:   bool               = True,
    ):
        self.source       = source
        self.model_path   = model_path
        self.device       = device
        self.conf         = conf
        self.imgsz        = imgsz
        self.infer_every  = max(1, int(infer_every))
        self._infer_every_user = self.infer_every  # keep user-requested floor
        self.target_fps   = max(1, int(target_fps))
        self.classes      = classes
        self.window_name  = window_name
        self.show_window  = show_window
        self.use_tracker  = use_tracker
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.display_scale = max(0.1, float(display_scale))
        self.adaptive      = bool(adaptive)
        # half precision: default ON for GPU, OFF for CPU (FP16 on CPU is slow)
        self.half = (device != "cpu") if half is None else bool(half)

        # NVDEC/hwaccel: opt-in.  Some FFmpeg builds bundled with opencv-python
        # don't have CUDA — in that case FFmpeg silently falls back to software
        # decode, so this flag is safe to set if you're unsure.
        if hwaccel:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|hwaccel;cuda|hwaccel_output_format;cuda"
            )
            print("[VideoEngine] Hardware decode requested (NVDEC via FFmpeg).")

        # ── homography (loaded once at startup) ──────────────────────────────
        self.H: Optional[np.ndarray] = self._load_homography(homography)

        # ── threading ────────────────────────────────────────────────────────
        self._stop = threading.Event()
        self._frame_q: queue.Queue[FramePacket] = queue.Queue(maxsize=queue_size)
        self._display_slot: deque[AnnotatedFrame] = deque(maxlen=1)

        # ── stats ────────────────────────────────────────────────────────────
        self.cap_fps = 0.0
        self.inf_fps = 0.0
        self.disp_fps = 0.0
        self.frames_captured = 0
        self.frames_dropped  = 0
        self.frames_inferred = 0
        self.frames_propagated = 0

        self._threads: list[threading.Thread] = []
        
        # ── Backend AI Analyst & Logger ──────────────────────────────────────
        self.logger = MatchLogger()
        self.analyst = OllamaAnalyst(model="llama3.2")

    # ── homography load ─────────────────────────────────────────────────────
    @staticmethod
    def _load_homography(path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            print(f"[VideoEngine] No homography at {p} — minimap disabled.  "
                  f"Run calibrate_court.py to generate it.")
            return None
        try:
            H = np.load(p)
            if H.shape != (3, 3):
                print(f"[VideoEngine] Bad homography shape {H.shape} — expected (3,3).")
                return None
            print(f"[VideoEngine] Homography loaded from {p}")
            return H.astype(np.float32)
        except Exception as exc:
            print(f"[VideoEngine] Failed to load homography: {exc}")
            return None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        self.analyst.start() # Start Ollama AI Analyst
        for fn, name in (
            (self._capture_loop,   "VE-capture"),
            (self._inference_loop, "VE-inference"),
            (self._display_loop,   "VE-display"),
        ):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=join_timeout)
        self._threads.clear()
        
        # Stop background AI services
        self.analyst.stop()
        self.logger.stop()

    def wait(self) -> None:
        try:
            for t in self._threads:
                t.join()
        except KeyboardInterrupt:
            self.stop()

    # ── 1. CAPTURE ───────────────────────────────────────────────────────────
    def _capture_loop(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        seq = 0
        n_since_log = 0
        last_log = time.perf_counter()

        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                cap = self._open_capture()
                if cap is None:
                    time.sleep(1.0)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release(); cap = None
                continue

            seq += 1
            self.frames_captured += 1
            n_since_log += 1
            pkt = FramePacket(frame=frame, timestamp=time.perf_counter(), seq=seq)

            try:
                self._frame_q.put_nowait(pkt)
            except queue.Full:
                # Drop-oldest → newest frame always wins. This is what keeps
                # latency bounded if YOLO momentarily stalls.
                try:
                    self._frame_q.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
                try:
                    self._frame_q.put_nowait(pkt)
                except queue.Full:
                    pass

            now = time.perf_counter()
            if now - last_log >= 1.0:
                self.cap_fps = n_since_log / (now - last_log)
                n_since_log = 0; last_log = now

        if cap is not None:
            cap.release()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ── 2. INFERENCE ─────────────────────────────────────────────────────────
    def _inference_loop(self) -> None:
        # Lazy load — keeps CUDA confined to this thread and lets the engine
        # initialise even without a GPU available.
        model = YOLO(self.model_path)

        # CUDA warm-up so the first real frame is fast.  The warm-up uses
        # the same FP16/FP32 path as the live loop so kernels are pre-compiled
        # for the right precision.
        warm = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        model.predict(warm, device=self.device, imgsz=self.imgsz,
                      conf=self.conf, half=self.half, verbose=False)
        print(f"[VideoEngine] YOLO ready  device={self.device}  "
              f"half={self.half}  imgsz={self.imgsz}  "
              f"model={Path(self.model_path).name}")

        # Tracker (high-occlusion tuned).  Optional: --no-tracker disables it.
        tracker = None
        if self.use_tracker:
            try:
                from player_tracker import PlayerTracker
                tracker = PlayerTracker(
                    device       = self.device,
                    frame_rate   = self.target_fps,
                    track_buffer = self.track_buffer,
                    match_thresh = self.match_thresh,
                )
            except Exception as exc:
                print(f"[VideoEngine] PlayerTracker unavailable ({exc}). "
                      f"Falling back to detection-only mode (no IDs).")

        last_tracks:  list[TrackedBox]      = []
        last_centers: Optional[np.ndarray]  = None   # (N,1,2) box centers for LK
        last_gray:    Optional[np.ndarray]  = None

        idx = 0; yolo_age = 0
        n_since_log = 0; last_log = time.perf_counter()

        while not self._stop.is_set():
            try:
                pkt = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            do_yolo = (idx % self.infer_every == 0) or not last_tracks
            inferred = False

            if do_yolo:
                last_tracks, last_centers = self._yolo_then_track(
                    model, tracker, pkt.frame,
                )
                last_gray = cv2.cvtColor(pkt.frame, cv2.COLOR_BGR2GRAY)
                yolo_age = 0; inferred = True
                self.frames_inferred += 1
            else:
                last_tracks, last_centers, last_gray = self._propagate(
                    pkt.frame, last_gray, last_tracks, last_centers,
                )
                yolo_age += 1
                self.frames_propagated += 1

            self._display_slot.append(AnnotatedFrame(
                frame=pkt.frame,
                tracks=last_tracks,
                seq=pkt.seq,
                inferred=inferred,
                yolo_age_frames=yolo_age,
                timestamp=pkt.timestamp,
            ))
            
            # --- Auto-log positions to Ollama's MatchLogger ---
            for t in last_tracks:
                fx, fy = t.foot_xy
                # MatchLogger automatically throttles high-frequency position logs
                self.logger.log_position(t.track_id, fx, fy)

            idx += 1; n_since_log += 1
            now = time.perf_counter()
            if now - last_log >= 1.0:
                self.inf_fps = n_since_log / (now - last_log)
                n_since_log = 0; last_log = now

                # ── adaptive frame-skipping ─────────────────────────────────
                # Goal: keep inference roughly matched to the source FPS so
                # the queue doesn't fill and frames don't get dropped.  We
                # nudge infer_every up when we're falling behind, and back
                # toward the user's request when we have headroom.  Only
                # active when --adaptive is on (default).
                if self.adaptive:
                    src_fps = self.cap_fps or self.target_fps
                    target_eff = src_fps / self.infer_every
                    if self.inf_fps < target_eff * 0.85 and self.infer_every < 8:
                        self.infer_every += 1
                        print(f"[VideoEngine] adaptive: infer_every → "
                              f"{self.infer_every} (inf {self.inf_fps:.1f} "
                              f"< target {target_eff:.1f})")
                    elif (self.inf_fps > target_eff * 1.5
                          and self.infer_every > self._infer_every_user):
                        self.infer_every -= 1
                        print(f"[VideoEngine] adaptive: infer_every → "
                              f"{self.infer_every} (headroom available)")

    def _yolo_then_track(
        self,
        model:   YOLO,
        tracker, # Optional[PlayerTracker]
        frame:   np.ndarray,
    ) -> tuple[list[TrackedBox], np.ndarray]:
        """Run YOLO, then BoT-SORT to get stable IDs.  Returns (tracks, centers)."""
        results = model.predict(
            frame,
            device  = self.device,
            conf    = self.conf,
            imgsz   = self.imgsz,
            classes = self.classes,
            half    = self.half,
            verbose = False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return [], np.empty((0, 1, 2), dtype=np.float32)

        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls  = r.boxes.cls.cpu().numpy().astype(np.int32)

        if tracker is not None:
            det_arr = np.hstack([xyxy, conf.reshape(-1, 1),
                                  cls.reshape(-1, 1).astype(np.float32)])
            t_objs = tracker.update(det_arr, frame)
            tracks = [
                TrackedBox(track_id=t.track_id,
                           x1=t.x1, y1=t.y1, x2=t.x2, y2=t.y2,
                           conf=t.confidence, class_id=t.class_id)
                for t in t_objs
            ]
        else:
            # Detection-only fallback — fake IDs from index, unstable across frames
            tracks = [
                TrackedBox(track_id=-(i + 1),
                           x1=float(xyxy[i, 0]), y1=float(xyxy[i, 1]),
                           x2=float(xyxy[i, 2]), y2=float(xyxy[i, 3]),
                           conf=float(conf[i]),  class_id=int(cls[i]))
                for i in range(len(xyxy))
            ]

        # Centers for LK propagation — use BOX center (not feet) because
        # box-center pixels usually have better texture for optical flow.
        centers = np.array(
            [[(t.x1 + t.x2) * 0.5, (t.y1 + t.y2) * 0.5] for t in tracks],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        return tracks, centers

    def _propagate(
        self,
        frame:        np.ndarray,
        last_gray:    Optional[np.ndarray],
        last_tracks:  list[TrackedBox],
        last_centers: Optional[np.ndarray],
    ) -> tuple[list[TrackedBox], np.ndarray, np.ndarray]:
        """Sparse LK on box centers — preserves IDs through intermediate frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (last_gray is None or not last_tracks or last_centers is None
                or len(last_centers) == 0):
            return [], np.empty((0, 1, 2), dtype=np.float32), gray

        new_centers, status, _ = cv2.calcOpticalFlowPyrLK(
            last_gray, gray, last_centers, None,
            winSize=(21, 21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if new_centers is None or status is None:
            return [], np.empty((0, 1, 2), dtype=np.float32), gray

        valid = status.reshape(-1).astype(bool)
        if not valid.any():
            return [], np.empty((0, 1, 2), dtype=np.float32), gray

        shift = (new_centers - last_centers).reshape(-1, 2)
        new_tracks: list[TrackedBox] = []
        for i, t in enumerate(last_tracks):
            if not valid[i]:
                continue
            dx, dy = float(shift[i, 0]), float(shift[i, 1])
            new_tracks.append(TrackedBox(
                track_id = t.track_id,
                x1 = t.x1 + dx, y1 = t.y1 + dy,
                x2 = t.x2 + dx, y2 = t.y2 + dy,
                conf = t.conf, class_id = t.class_id,
            ))
        return new_tracks, new_centers[valid], gray

    # ── 3. DISPLAY ───────────────────────────────────────────────────────────
    def _display_loop(self) -> None:
        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        target_dt = 1.0 / self.target_fps
        n_since_log = 0; last_log = time.perf_counter()

        while not self._stop.is_set():
            t0 = time.perf_counter()

            ann = self._display_slot[-1] if self._display_slot else None
            if ann is not None and self.show_window:
                vis = ann.frame.copy()
                self._draw_boxes(vis, ann.tracks, propagated=not ann.inferred)
                if self.H is not None:
                    self._draw_minimap(vis, ann.tracks)
                self._draw_hud(vis, ann)
                if self.display_scale != 1.0:
                    # Downscale only the displayed copy — inference and
                    # tracking always see the native resolution.
                    new_w = int(vis.shape[1] * self.display_scale)
                    new_h = int(vis.shape[0] * self.display_scale)
                    vis = cv2.resize(vis, (new_w, new_h),
                                     interpolation=cv2.INTER_LINEAR)
                cv2.imshow(self.window_name, vis)

            if self.show_window:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    self._stop.set()
                    break

            n_since_log += 1
            now = time.perf_counter()
            if now - last_log >= 1.0:
                self.disp_fps = n_since_log / (now - last_log)
                n_since_log = 0; last_log = now

            sleep = target_dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        if self.show_window:
            cv2.destroyWindow(self.window_name)

    @staticmethod
    def _draw_boxes(img: np.ndarray, tracks: list[TrackedBox],
                    propagated: bool) -> None:
        for t in tracks:
            x1, y1, x2, y2 = int(t.x1), int(t.y1), int(t.x2), int(t.y2)
            col = (0, 180, 220) if propagated else (0, 220, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

            # Foot anchor — the point we project onto the minimap.  Drawing
            # it makes mis-calibration visually obvious during the demo.
            fx, fy = t.foot_xy
            cv2.circle(img, (int(fx), int(fy)), 4, (255, 255, 255), -1)
            cv2.circle(img, (int(fx), int(fy)), 5, (0, 0, 0), 1)

            label = f"#{t.track_id}" if t.track_id >= 0 else "?"
            cv2.putText(img, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    def _draw_minimap(self, img: np.ndarray,
                      tracks: list[TrackedBox]) -> None:
        """
        Render a 400×200 bird's-eye minimap in the bottom-right corner.

        BOTTOM-CENTER ANCHORING IS MANDATORY HERE.
        ──────────────────────────────────────────
        With the camera in the stands at an oblique angle, the bbox CENTER
        sits at chest height; projecting it through the homography places
        the player ~3-5 m behind their actual position on the court.  The
        feet (y_max) lie on the ground plane that the homography was built
        for, so foot_xy() is the only correct anchor.
        """
        H = self.H
        assert H is not None  # guarded by caller

        # Build the canvas
        mm = np.full((COURT_H_PX, COURT_W_PX, 3), 30, dtype=np.uint8)
        # Court outline + half-line
        cv2.rectangle(mm, (0, 0), (COURT_W_PX - 1, COURT_H_PX - 1),
                      (200, 200, 200), 1)
        cv2.line(mm, (COURT_W_PX // 2, 0), (COURT_W_PX // 2, COURT_H_PX),
                 (200, 200, 200), 1)
        # 6 m goal area (60 px from each end)
        cv2.line(mm, (60, 0), (60, COURT_H_PX), (120, 120, 120), 1)
        cv2.line(mm, (COURT_W_PX - 60, 0),
                 (COURT_W_PX - 60, COURT_H_PX), (120, 120, 120), 1)

        if tracks:
            # Project FEET (bottom-center), not bbox center.
            foot_pts = np.array(
                [[t.foot_xy] for t in tracks],
                dtype=np.float32,
            )                                       # (N, 1, 2)
            court_pts = cv2.perspectiveTransform(foot_pts, H)  # (N, 1, 2)

            for t, cp in zip(tracks, court_pts):
                cx, cy = int(cp[0, 0]), int(cp[0, 1])
                # Skip points that fall outside the court rectangle (off-pitch
                # detections, calibration drift, projection over the horizon).
                if not (0 <= cx < COURT_W_PX and 0 <= cy < COURT_H_PX):
                    continue
                col = (0, 220, 220) if t.track_id == 32 else (0, 220, 0) # Fallback heuristic
                cv2.circle(mm, (cx, cy), 5, col, -1)
                cv2.circle(mm, (cx, cy), 6, (0, 0, 0), 1)
                if t.track_id >= 0 and t.track_id != 32:
                    cv2.putText(mm, str(t.track_id), (cx + 6, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1, cv2.LINE_AA)

        # Paste minimap into bottom-right of the main view, with a thin border
        h, w = img.shape[:2]
        x0, y0 = w - COURT_W_PX - 12, h - COURT_H_PX - 12
        cv2.rectangle(img, (x0 - 2, y0 - 2),
                      (x0 + COURT_W_PX + 1, y0 + COURT_H_PX + 1),
                      (0, 0, 0), 2)
        img[y0:y0 + COURT_H_PX, x0:x0 + COURT_W_PX] = mm

    def _draw_hud(self, img: np.ndarray, ann: AnnotatedFrame) -> None:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 28), (15, 15, 15), -1)
        latency_ms = (time.perf_counter() - ann.timestamp) * 1000.0
        homo_state = "ON" if self.H is not None else "off"
        text = (f"cap {self.cap_fps:5.1f}  "
                f"inf {self.inf_fps:5.1f}  "
                f"disp {self.disp_fps:5.1f}  |  "
                f"q {self._frame_q.qsize():2d}/{self._frame_q.maxsize}  "
                f"lat {latency_ms:4.0f}ms  "
                f"yolo+{ann.yolo_age_frames}f  "
                f"trk {len(ann.tracks):2d}  "
                f"homo:{homo_state}  "
                f"drop {self.frames_dropped}")
        cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 0), 1, cv2.LINE_AA)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_classes(arg: Optional[str]) -> Optional[list[int]]:
    if not arg:
        return None
    return [int(c) for c in arg.split(",") if c.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Threaded RTSP video engine with high-occlusion tracking + minimap")
    parser.add_argument("--source",       required=True)
    parser.add_argument("--model",        default="yolov8n.pt")
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--conf",         type=float, default=0.25)
    parser.add_argument("--imgsz",        type=int,   default=640)
    parser.add_argument("--infer-every",  type=int,   default=3)
    parser.add_argument("--fps",          type=int,   default=30)
    parser.add_argument("--queue",        type=int,   default=30)
    parser.add_argument("--classes",      default="",
                        help="COCO class IDs (e.g. '0' persons, '0,32' person+ball)")
    parser.add_argument("--homography",   default="homography_matrix.npy",
                        help="Path to homography .npy from calibrate_court.py. "
                             "Empty string disables the minimap.")
    parser.add_argument("--track-buffer", type=int, default=90,
                        help="Frames to keep an unmatched track alive (3 s @ 30 fps)")
    parser.add_argument("--match-thresh", type=float, default=0.85,
                        help="BoT-SORT association cost threshold "
                             "(higher = more permissive, recovers more occlusions)")
    parser.add_argument("--no-tracker",   action="store_true",
                        help="Skip BoT-SORT (raw detection mode, no stable IDs)")
    parser.add_argument("--no-display",   action="store_true")

    # ── performance knobs ────────────────────────────────────────────────────
    parser.add_argument("--half", choices=("auto", "on", "off"), default="auto",
                        help="FP16 inference: 'auto' = on for GPU, off for CPU")
    parser.add_argument("--display-scale", type=float, default=1.0,
                        help="Render at fraction of native resolution "
                             "(e.g. 0.5 = half-size window, ~4× cheaper blit)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable auto-tuning of infer_every under load")
    parser.add_argument("--hwaccel", action="store_true",
                        help="Try NVDEC hardware decode via FFmpeg")
    args = parser.parse_args()

    half_arg = None if args.half == "auto" else (args.half == "on")

    homo = Path(args.homography) if args.homography else None

    engine = VideoEngine(
        source        = args.source,
        model_path    = args.model,
        device        = args.device,
        conf          = args.conf,
        imgsz         = args.imgsz,
        infer_every   = args.infer_every,
        target_fps    = args.fps,
        queue_size    = args.queue,
        classes       = _parse_classes(args.classes),
        homography    = homo,
        track_buffer  = args.track_buffer,
        match_thresh  = args.match_thresh,
        use_tracker   = not args.no_tracker,
        half          = half_arg,
        display_scale = args.display_scale,
        adaptive      = not args.no_adaptive,
        hwaccel       = args.hwaccel,
        show_window   = not args.no_display,
    )

    signal.signal(signal.SIGINT, lambda *_: engine.stop())

    print(f"[VideoEngine] source={args.source}  device={args.device}  "
          f"model={args.model}  imgsz={args.imgsz}  "
          f"infer_every={args.infer_every}  target_fps={args.fps}")
    print(f"[VideoEngine] tracker={'on' if not args.no_tracker else 'OFF'}  "
          f"track_buffer={args.track_buffer}  match_thresh={args.match_thresh}")
    print("[VideoEngine] Press Q in the window or Ctrl+C in the terminal to stop.")

    engine.start()
    engine.wait()
    engine.stop()