"""RTSP/HTTP/Webcam capture thread — reads frames into a queue."""
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclass
class FramePacket:
    frame:     object   # numpy ndarray (BGR), already undistorted if configured
    timestamp: float    # time.perf_counter() at read


class CaptureThread(threading.Thread):
    RECONNECT_DELAY = 2.0
    MAX_FAILURES    = 10

    def __init__(
        self,
        url:           str,
        queue_size:    int                    = 2,
        camera_matrix: Optional[np.ndarray]  = None,
        dist_coeffs:   Optional[np.ndarray]  = None,
        loop:          bool                   = False,
    ):
        super().__init__(daemon=True, name="CaptureThread")
        self._loop = bool(loop)   # replay a video FILE from the start on EOF
        self._seek_to = None      # pending seek fraction (file only), 0.0..1.0
        # Live playback position (files only) — read by the GUI transport bar.
        self._pos_frames   = 0
        self._total_frames = 0
        self._fps          = 0.0
        
        # Handle Webcam IDs (e.g., user inputs "0")
        if str(url).isdigit():
            self.url = int(url)
        else:
            self.url = str(url)
            
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=queue_size)
        # FIXED: Changed _stop to _stop_event to avoid Python 3.12+ Threading conflicts
        self._stop_event  = threading.Event()
        self._cap:  Optional[cv2.VideoCapture] = None

        # Undistortion state
        self._K:    Optional[np.ndarray] = (
            camera_matrix.astype(np.float64) if camera_matrix is not None else None
        )
        self._dist: Optional[np.ndarray] = (
            dist_coeffs.astype(np.float64) if dist_coeffs is not None else None
        )
        self._map1: Optional[np.ndarray] = None   
        self._map2: Optional[np.ndarray] = None   

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config/camera.yaml",
        queue_size:  int        = 2,
    ) -> "CaptureThread":
        cfg = yaml.safe_load(Path(config_path).read_text())
        url = (
            f"rtsp://{cfg['user']}:{cfg['password']}"
            f"@{cfg['ip']}:{cfg['rtsp_port']}"
            f"/Streaming/Channels/{cfg['channel']}"
        )
        K = dist = None
        ud = cfg.get("undistort", {})
        if ud.get("enabled", False):
            K    = np.array(ud["camera_matrix"], dtype=np.float64)
            dist = np.array(ud["dist_coeffs"],   dtype=np.float64)

        return cls(url, queue_size, camera_matrix=K, dist_coeffs=dist)

    def stop(self) -> None:
        self._stop_event.set()

    def seek_fraction(self, frac: float) -> None:
        """Request a jump to ``frac`` (0..1) of a video FILE. Applied by the
        capture thread on its next read (thread-safe handoff)."""
        self._seek_to = max(0.0, min(1.0, float(frac)))

    def position(self) -> tuple[int, int]:
        """(current_frame, total_frames) for a file; (0, 0) for a live stream.
        Cached values are updated by the capture thread, so reading them here is
        thread-safe (no concurrent VideoCapture access)."""
        if not self._is_file:
            return (0, 0)
        return (int(self._pos_frames), int(self._total_frames))

    def fps(self) -> float:
        return float(self._fps) if self._fps and self._fps > 0 else 25.0

    def get_frame(self, timeout: float = 0.1) -> Optional[FramePacket]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def _is_file(self) -> bool:
        # FIXED: Don't treat webcams or HTTP streams as static files
        if isinstance(self.url, int):
            return False 
        return not (self.url.startswith("rtsp://") or self.url.startswith("http://"))

    def _open(self) -> bool:
        if not self._is_file and isinstance(self.url, str):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp"
                "|timeout;5000000"
                "|fflags;nobuffer"
                "|flags;low_delay"
                "|framedrop;1"
            )
            
        # Initialize VideoCapture (handles both string URLs and integer Webcams)
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG if isinstance(self.url, str) and not self.url.startswith("http") else cv2.CAP_ANY)
        
        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._is_file:
            _flush_deadline = time.perf_counter() + 1.0   
            while time.perf_counter() < _flush_deadline:
                cap.grab()

        self._cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._is_file:
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._build_undistort_maps(w, h)
        print(f"[Capture] Opened  {w}×{h}  "
              f"({'file' if self._is_file else 'stream'})"
              f"{'  undistort=ON' if self._map1 is not None else ''}")
        return True

    def _build_undistort_maps(self, w: int, h: int) -> None:
        if self._K is None or self._dist is None:
            return
        K_opt, _roi = cv2.getOptimalNewCameraMatrix(
            self._K, self._dist, (w, h), alpha=0, newImgSize=(w, h)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._K, self._dist, None, K_opt, (w, h), cv2.CV_16SC2
        )
        print(f"[Capture] Undistort remap tables built  ({w}×{h})")

    def run(self) -> None:
        while not self._stop_event.is_set():
            if not self._open():
                if self._is_file:
                    print(f"[Capture] Cannot open file: {self.url}")
                    return
                print(f"[Capture] Cannot connect — retry in {self.RECONNECT_DELAY}s")
                time.sleep(self.RECONNECT_DELAY)
                continue

            failures = 0
            while not self._stop_event.is_set():
                # Apply a pending seek (video files only).
                if self._seek_to is not None and self._is_file:
                    total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    if total > 0:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES,
                                      int(self._seek_to * (total - 1)))
                    self._seek_to = None
                ret, frame = self._cap.read()
                if not ret:
                    if self._is_file:
                        if self._loop and not self._stop_event.is_set():
                            # Replay from the first frame — keeps the live view
                            # going for demos / phone monitoring without a model
                            # reload.
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        print("[Capture] Video file ended.")
                        self._stop_event.set()
                        break
                    failures += 1
                    if failures >= self.MAX_FAILURES:
                        print("[Capture] Stream lost — reconnecting…")
                        break
                    continue

                failures = 0

                if self._is_file:                      # cache position for the GUI
                    try:
                        self._pos_frames = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                    except Exception:
                        pass

                if self._map1 is not None:
                    frame = cv2.remap(
                        frame, self._map1, self._map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )

                packet = FramePacket(frame=frame, timestamp=time.perf_counter())

                if self._is_file:
                    self.queue.put(packet)
                else:
                    if self.queue.full():
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.queue.put_nowait(packet)

            if self._cap:
                self._cap.release()
                self._cap = None
                self._map1 = self._map2 = None

            if self._is_file or self._stop_event.is_set():
                break
            time.sleep(self.RECONNECT_DELAY)

        print("[Capture] Thread stopped.")