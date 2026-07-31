"""
calibrate_court.py — standalone court homography calibration tool.

Workflow:
    1. Reads ONE frame from the source (RTSP URL or video file) and displays it.
    2. Click the 4 court corners in this order:
           TL (top-left)  →  TR (top-right)  →  BR (bottom-right)  →  BL (bottom-left)
       A circle + label is drawn on each click.
    3. Press ENTER to compute the perspective transform mapping those 4 points
       to a 400×200 bird's-eye rectangle (10 px = 1 m on a 40×20 m IHF court).
    4. The homography matrix is saved to ./homography_matrix.npy.

Hotkeys:
    Enter   — confirm 4 points and save the matrix
    U       — undo the last clicked point
    R       — reset all points
    ESC / Q — quit without saving

This is a calibration tool only — no YOLO, no tracking, no event logic.

Usage:
    python calibrate_court.py --source "rtsp://admin:PASS@172.16.5.174:554/Streaming/Channels/102"
    python calibrate_court.py --source match.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# RTSP over TCP is significantly more stable than the default UDP transport.
# Must be set before cv2.VideoCapture() opens the stream.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|fflags;nobuffer|flags;low_delay",
)


# ── Bird's-eye target rectangle (10 px/m on a 40 × 20 m IHF court) ───────────
COURT_W_PX = 400
COURT_H_PX = 200

# Order matters: TL, TR, BR, BL.  The homography orientation depends on it.
CORNER_LABELS = ("TL", "TR", "BR", "BL")
DST_POINTS = np.array(
    [
        [0,            0           ],   # TL
        [COURT_W_PX,   0           ],   # TR
        [COURT_W_PX,   COURT_H_PX  ],   # BR
        [0,            COURT_H_PX  ],   # BL
    ],
    dtype=np.float32,
)


# ── First-frame grab ─────────────────────────────────────────────────────────

def grab_first_frame(source: str) -> np.ndarray:
    """Open the source, read exactly one frame, release.  Raise on failure."""
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # On RTSP, the first read sometimes returns a stale buffered frame.
    # Read up to 5 frames and use the last one — guarantees a fresh image.
    frame = None
    for _ in range(5):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()

    if frame is None:
        raise RuntimeError(f"Source opened but no frame could be decoded: {source}")
    return frame


# ── Undistortion (must match src/capture.py so clicks map to live frames) ────

def make_undistorter(camera_config: str | Path):
    """Return fn(frame) -> frame applying the exact undistortion the live
    CaptureThread applies.  Remap tables are built lazily on the first frame
    and cached.  Identity function if config is missing/disabled."""
    cfg_path = Path(camera_config)
    if not cfg_path.exists():
        return lambda f: f
    ud = (yaml.safe_load(cfg_path.read_text()) or {}).get("undistort", {})
    if not ud.get("enabled", False):
        return lambda f: f

    K    = np.array(ud["camera_matrix"], dtype=np.float64)
    dist = np.array(ud["dist_coeffs"],   dtype=np.float64)
    maps: list = []          # [map1, map2] built on first call

    def _apply(frame: np.ndarray) -> np.ndarray:
        if not maps:
            h, w = frame.shape[:2]
            K_opt, _roi = cv2.getOptimalNewCameraMatrix(
                K, dist, (w, h), alpha=0, newImgSize=(w, h)
            )
            maps.extend(cv2.initUndistortRectifyMap(
                K, dist, None, K_opt, (w, h), cv2.CV_16SC2
            ))
            print("[Calibrate] Undistortion active (matching live engine).")
        return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

    return _apply


# ── Digital zoom/pan for pixel-accurate corner clicking ──────────────────────

class ZoomView:
    """Digital zoom + pan over a fixed-size image.

    View coordinates are always the original (w, h) window size; the visible
    region is a (w/zoom, h/zoom) crop at offset (ox, oy) scaled back up.
    """

    MAX_ZOOM = 10.0

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.zoom = 1.0
        self.ox = 0.0
        self.oy = 0.0

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x / self.zoom, self.oy + y / self.zoom

    def to_view(self, px: float, py: float) -> tuple[int, int]:
        return (int(round((px - self.ox) * self.zoom)),
                int(round((py - self.oy) * self.zoom)))

    def _clamp(self) -> None:
        self.ox = min(max(self.ox, 0.0), self.w - self.w / self.zoom)
        self.oy = min(max(self.oy, 0.0), self.h - self.h / self.zoom)

    def wheel(self, x: int, y: int, delta: int) -> None:
        """Zoom in/out keeping the image point under the cursor fixed."""
        px, py = self.to_image(x, y)
        factor = 1.25 if delta > 0 else 1 / 1.25
        self.zoom = min(self.MAX_ZOOM, max(1.0, self.zoom * factor))
        self.ox = px - x / self.zoom
        self.oy = py - y / self.zoom
        self._clamp()

    def pan(self, dx_view: int, dy_view: int) -> None:
        self.ox -= dx_view / self.zoom
        self.oy -= dy_view / self.zoom
        self._clamp()

    def reset(self) -> None:
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0

    def render(self, img: np.ndarray) -> np.ndarray:
        if self.zoom <= 1.0:
            return img.copy()
        vw = max(1, int(round(self.w / self.zoom)))
        vh = max(1, int(round(self.h / self.zoom)))
        x0 = min(int(round(self.ox)), self.w - vw)
        y0 = min(int(round(self.oy)), self.h - vh)
        crop = img[y0:y0 + vh, x0:x0 + vw]
        return cv2.resize(crop, (self.w, self.h),
                          interpolation=cv2.INTER_LINEAR)


# ── Live preview: adjust the camera, then freeze a frame to calibrate ────────

def live_preview(cap: cv2.VideoCapture, undistort) -> np.ndarray | None:
    """Stream frames until SPACE (returns the latest frame, undistorted) or
    ESC/Q (returns None).  Lets the operator physically adjust the camera
    before freezing the calibration frame."""
    win = "LIVE preview  -  SPACE = freeze & calibrate   ESC = quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    view: ZoomView | None = None
    drag: list = [None]                       # mutable box for the callback

    def _on_mouse(event, x, y, flags, _):
        if view is None:
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            view.wheel(x, y, cv2.getMouseWheelDelta(flags))
        elif event == cv2.EVENT_RBUTTONDOWN:
            drag[0] = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            drag[0] = None
        elif event == cv2.EVENT_MOUSEMOVE and drag[0] is not None:
            view.pan(x - drag[0][0], y - drag[0][1])
            drag[0] = (x, y)

    cv2.setMouseCallback(win, _on_mouse)

    last: np.ndarray | None = None
    try:
        while True:
            ok, f = cap.read()
            if ok and f is not None:
                last = undistort(f)
            if last is None:
                if (cv2.waitKey(50) & 0xFF) in (27, ord("q")):
                    return None
                continue

            if view is None:
                h0, w0 = last.shape[:2]
                view = ZoomView(w0, h0)

            disp = view.render(last)
            h, w = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (w, 30), (15, 15, 15), -1)
            cv2.putText(disp,
                        f"LIVE - adjust camera  |  SPACE=freeze & calibrate  |  "
                        f"zoom {view.zoom:.1f}x (wheel, right-drag=pan, 0=reset)  |  ESC=quit",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 200, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return None
            if key == 32:                        # SPACE → freeze (full frame)
                return last
            if key in (ord("+"), ord("=")):
                view.wheel(view.w // 2, view.h // 2, +1)
            if key in (ord("-"), ord("_")):
                view.wheel(view.w // 2, view.h // 2, -1)
            if key == ord("0"):
                view.reset()
    finally:
        cv2.destroyWindow(win)


# ── Calibration UI ───────────────────────────────────────────────────────────

class CourtCalibrator:
    WINDOW = "Court Calibration  (TL → TR → BR → BL)"

    def __init__(self, frame: np.ndarray):
        self.base   = frame.copy()         # untouched copy for redraws
        self.points: list[tuple[int, int]] = []   # in IMAGE coordinates
        h, w = frame.shape[:2]
        self.view = ZoomView(w, h)
        self._drag: tuple[int, int] | None = None

    # mouse callback ----------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_MOUSEWHEEL:
            self.view.wheel(x, y, cv2.getMouseWheelDelta(flags))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._drag = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            self.view.pan(x - self._drag[0], y - self._drag[1])
            self._drag = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            px, py = self.view.to_image(x, y)
            self.points.append((int(round(px)), int(round(py))))

    # rendering ---------------------------------------------------------------
    def _render(self) -> np.ndarray:
        img  = self.view.render(self.base)
        vpts = [self.view.to_view(px, py) for (px, py) in self.points]

        # connect the points so the user can see the polygon shape
        if len(vpts) >= 2:
            for i in range(len(vpts) - 1):
                cv2.line(img, vpts[i], vpts[i + 1],
                         (0, 200, 255), 1, cv2.LINE_AA)
            if len(vpts) == 4:
                cv2.line(img, vpts[3], vpts[0],
                         (0, 200, 255), 1, cv2.LINE_AA)

        # crosshair markers + labels (small dot stays pixel-accurate at zoom)
        for i, (vx, vy) in enumerate(vpts):
            cv2.drawMarker(img, (vx, vy), (0, 255, 0),
                           cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
            cv2.circle(img, (vx, vy), 3, (0, 255, 0), -1, cv2.LINE_AA)
            label = f"{i + 1}:{CORNER_LABELS[i]}"
            cv2.putText(img, label, (vx + 12, vy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0),   3, cv2.LINE_AA)
            cv2.putText(img, label, (vx + 12, vy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)

        # status bar
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 30), (15, 15, 15), -1)
        zoom_hint = f"zoom {self.view.zoom:.1f}x (wheel, right-drag=pan, 0=reset)"
        if len(self.points) < 4:
            msg = (f"Click corner {len(self.points) + 1} of 4 "
                   f"({CORNER_LABELS[len(self.points)]})   "
                   f"|  {zoom_hint}  |  U=undo  R=reset  ESC=back")
            color = (0, 200, 255)
        else:
            msg = f"4 corners set: ENTER=save  |  {zoom_hint}  |  U=undo  R=reset"
            color = (0, 255, 0)
        cv2.putText(img, msg, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return img

    # main loop ---------------------------------------------------------------
    def run(self) -> np.ndarray | None:
        """Returns the 4 source points as float32 array, or None if cancelled."""
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)

        try:
            while True:
                cv2.imshow(self.WINDOW, self._render())
                key = cv2.waitKey(20) & 0xFF

                if key in (27, ord("q")):                # ESC / Q → cancel
                    return None

                if key in (ord("u"), 8):                 # U / Backspace → undo
                    if self.points:
                        self.points.pop()

                if key == ord("r"):                      # R → reset
                    self.points.clear()

                if key in (ord("+"), ord("=")):          # keyboard zoom in
                    self.view.wheel(self.view.w // 2, self.view.h // 2, +1)
                if key in (ord("-"), ord("_")):          # keyboard zoom out
                    self.view.wheel(self.view.w // 2, self.view.h // 2, -1)
                if key == ord("0"):                      # reset zoom
                    self.view.reset()

                if key in (13, 10) and len(self.points) == 4:   # Enter → confirm
                    return np.asarray(self.points, dtype=np.float32)
        finally:
            cv2.destroyWindow(self.WINDOW)


# ── Preview the warped bird's-eye view ───────────────────────────────────────

def show_birdseye_preview(frame: np.ndarray, H: np.ndarray) -> None:
    warped = cv2.warpPerspective(frame, H, (COURT_W_PX, COURT_H_PX))
    win = "Bird's-eye preview  (any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, warped)
    cv2.waitKey(0)
    cv2.destroyWindow(win)


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Click 4 court corners and save the homography matrix.")
    parser.add_argument("--source", required=True,
                        help="RTSP URL or video file path")
    parser.add_argument("--out", default="homography_matrix.npy",
                        help="Output .npy path (default: ./homography_matrix.npy)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the bird's-eye preview after saving")
    parser.add_argument("--camera-config", default="config/camera.yaml",
                        help="camera.yaml used to mirror the live engine's "
                             "undistortion (default: config/camera.yaml)")
    parser.add_argument("--live", action="store_true",
                        help="Show a live preview first; SPACE freezes the "
                             "frame for calibration (auto-on for rtsp:// sources)")
    args = parser.parse_args()

    undistort = make_undistorter(args.camera_config)
    is_live   = args.live or args.source.startswith("rtsp")

    if is_live:
        print(f"[Calibrate] LIVE mode — opening stream: {args.source}")
        cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[Calibrate] ERROR: could not open source: {args.source}")
            return 1

        # Loop: live preview → freeze → click corners.  ESC while clicking
        # returns to the live preview; ESC in the preview quits for real.
        points = frame = None
        try:
            while True:
                frame = live_preview(cap, undistort)
                if frame is None:
                    print("[Calibrate] Cancelled — nothing saved.")
                    return 1
                print(f"[Calibrate] Frame frozen ({frame.shape[1]}x{frame.shape[0]}). "
                      f"Click corners: TL, TR, BR, BL — ESC returns to live view.")
                points = CourtCalibrator(frame).run()
                if points is not None:
                    break
                print("[Calibrate] Back to live view — adjust camera, then SPACE again.")
        finally:
            cap.release()
    else:
        print(f"[Calibrate] Grabbing first frame from: {args.source}")
        try:
            frame = undistort(grab_first_frame(args.source))
        except RuntimeError as exc:
            print(f"[Calibrate] ERROR: {exc}")
            return 1
        print(f"[Calibrate] Got frame {frame.shape[1]}x{frame.shape[0]}.  "
              f"Click corners in order: TL, TR, BR, BL")
        points = CourtCalibrator(frame).run()
        if points is None:
            print("[Calibrate] Cancelled — nothing saved.")
            return 1

    H = cv2.getPerspectiveTransform(points, DST_POINTS)

    # Both consumers must see the same matrix:
    #   src/video_engine.py reads ./homography_matrix.npy
    #   src/pipeline.py     reads ./config/homography_matrix.npy
    out_path = Path(args.out).resolve()
    np.save(out_path, H)
    print(f"[Calibrate] Saved homography → {out_path}")
    config_copy = Path("config") / "homography_matrix.npy"
    if config_copy.resolve() != out_path and config_copy.parent.is_dir():
        np.save(config_copy, H)
        print(f"[Calibrate] Saved homography → {config_copy.resolve()}")
    print(f"[Calibrate] Source corners (TL,TR,BR,BL):\n{points}")
    print(f"[Calibrate] Homography matrix:\n{H}")

    if not args.no_preview:
        print("[Calibrate] Showing bird's-eye preview — press any key to close.")
        show_birdseye_preview(frame, H)

    return 0


if __name__ == "__main__":
    sys.exit(main())
