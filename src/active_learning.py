"""
Active Learning Corrector — pause-and-teach during live playback.

User flow
─────────
1. While the pipeline runs, the user spots the ball being missed (or a head
   being mis-detected as a ball).
2. User presses 'P' → pipeline pauses, this module enters correction mode.
3. The cv2 window stays on the current frame and listens for mouse clicks:
       * LEFT-click on the ball       → positive label
       * RIGHT-click on a false ball  → negative point (zone-of-rejection)
       * U                            → undo last click on this frame
       * +/- (or [ / ])               → adjust bbox size
       * C  or SPACE                  → save + resume playback
       * X                            → discard corrections for this frame
                                          (don't save, just resume)
4. On save, the frame image goes to
       training/dataset/images/train/<video>_f<NNNNNN>.jpg
   and a YOLO label file with class 0 = ball goes to
       training/dataset/labels/train/<video>_f<NNNNNN>.txt
5. Corrections accumulate into the training dataset. The user can press 'T'
   from the live pipeline (any time) to trigger a background re-train of the
   ball model on the updated dataset.

Negative labels
───────────────
YOLO format doesn't have explicit "negative point" labels — you simply
exclude that region from positives. We achieve the same effect by saving the
frame with ONLY the corrected positive label (if any). Frames where the user
right-clicks a wrong detection but didn't supply a positive get saved as
"empty label" files — these are *hard negatives* (the model gets a strong
"there's NO ball anywhere here" signal during training).
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ActiveLearningCorrector:
    """State machine for one correction session (= one paused frame)."""

    DEFAULT_BBOX_PX = 26

    def __init__(
        self,
        dataset_root: str | Path = "training/dataset",
        video_stem:   str = "session",
    ):
        self.dataset_root = Path(dataset_root)
        self.images_dir   = self.dataset_root / "images" / "train"
        self.labels_dir   = self.dataset_root / "labels" / "train"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.video_stem   = video_stem

        # Track which (video, frame_n) the user has manually corrected — these
        # are PROTECTED from being overwritten by future auto_label runs.
        self._protected_index = self.dataset_root / "_user_corrected.json"
        self._protected: set[str] = self._load_protected()

        # Current correction session state (only valid while paused)
        self.active = False
        self._frame: Optional[np.ndarray] = None
        self._frame_n: int = -1
        self._fw = 0
        self._fh = 0
        self._positives: list[tuple[float, float]] = []   # (cx, cy) in pixel coords
        self._negatives: list[tuple[float, float]] = []   # marked FPs
        self._bbox_px = self.DEFAULT_BBOX_PX
        self.last_status_msg = ""
        self.total_saved = 0

    # ────────────────────────────────────────────────────────────────
    #  Session control
    # ────────────────────────────────────────────────────────────────
    def begin(self, frame: np.ndarray, frame_n: int) -> None:
        """Enter correction mode for a single paused frame."""
        if self.active:
            return
        self.active     = True
        self._frame     = frame.copy()
        self._frame_n   = int(frame_n)
        self._fh, self._fw = frame.shape[:2]
        self._positives = []
        self._negatives = []
        self._bbox_px   = self.DEFAULT_BBOX_PX
        self.last_status_msg = "Correction mode — L-click=ball  R-click=NO ball  U=undo  C/SPACE=save  X=discard  +/-=size"

    def end_save(self) -> bool:
        """Save the current corrections to disk and exit correction mode.
        Returns True if a frame+label was written."""
        if not self.active or self._frame is None:
            return False
        wrote = self._write_frame_and_label()
        if wrote:
            self.total_saved += 1
            self._protected.add(self._stem_for_current())
            self._save_protected()
        self._cleanup_session()
        return wrote

    def end_discard(self) -> None:
        """Exit without saving."""
        self.last_status_msg = "Correction discarded"
        self._cleanup_session()

    def _cleanup_session(self) -> None:
        self.active = False
        self._frame = None
        self._frame_n = -1
        self._positives = []
        self._negatives = []

    # ────────────────────────────────────────────────────────────────
    #  Click + key handling
    # ────────────────────────────────────────────────────────────────
    def on_mouse(self, event: int, x: int, y: int, flags: int, _userdata) -> None:
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._positives.append((float(x), float(y)))
            self.last_status_msg = f"Added BALL at ({x},{y})  size={self._bbox_px}px"
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._negatives.append((float(x), float(y)))
            self.last_status_msg = f"Marked NO-BALL at ({x},{y})"

    def handle_key(self, key: int) -> Optional[str]:
        """Returns an action: 'save', 'discard', or None (consumed)."""
        if not self.active:
            return None
        if key in (ord('c'), ord('C'), ord(' '), 13):     # C or SPACE or Enter → save
            return 'save'
        if key in (ord('x'), ord('X'), 27):                # X or ESC → discard
            return 'discard'
        if key in (ord('u'), ord('U')):                    # U → undo
            if self._negatives:
                p = self._negatives.pop()
                self.last_status_msg = f"Undid NO-BALL at ({int(p[0])},{int(p[1])})"
            elif self._positives:
                p = self._positives.pop()
                self.last_status_msg = f"Undid BALL at ({int(p[0])},{int(p[1])})"
            return None
        if key in (ord('+'), ord('='), ord(']')):
            self._bbox_px = min(120, self._bbox_px + 4)
            self.last_status_msg = f"bbox size = {self._bbox_px}px"
            return None
        if key in (ord('-'), ord('_'), ord('[')):
            self._bbox_px = max(8, self._bbox_px - 4)
            self.last_status_msg = f"bbox size = {self._bbox_px}px"
            return None
        return None

    # ────────────────────────────────────────────────────────────────
    #  Overlay rendering
    # ────────────────────────────────────────────────────────────────
    def render_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draw current annotations + HUD on top of a frame copy. Caller
        owns the returned image (it's a copy)."""
        if not self.active or self._frame is None:
            return img
        out = img.copy()
        half = self._bbox_px // 2
        # Positives — yellow boxes
        for x, y in self._positives:
            cv2.rectangle(out,
                          (int(x) - half, int(y) - half),
                          (int(x) + half, int(y) + half),
                          (0, 230, 255), 2)
            cv2.circle(out, (int(x), int(y)), 3, (0, 200, 255), -1)
        # Negatives — red X
        for x, y in self._negatives:
            ix, iy = int(x), int(y)
            cv2.line(out, (ix - 10, iy - 10), (ix + 10, iy + 10), (0, 0, 255), 2)
            cv2.line(out, (ix - 10, iy + 10), (ix + 10, iy - 10), (0, 0, 255), 2)
            cv2.circle(out, (ix, iy), 14, (0, 0, 255), 1, cv2.LINE_AA)

        # HUD
        h = out.shape[0]
        cv2.rectangle(out, (0, h - 60), (out.shape[1], h), (15, 15, 15), -1)
        cv2.putText(out, "[PAUSED — CORRECTION MODE]",
                    (12, h - 38), cv2.FONT_HERSHEY_DUPLEX, 0.65,
                    (0, 230, 255), 2, cv2.LINE_AA)
        msg = self.last_status_msg or "L-click=ball  R-click=NO-ball  U=undo  C/SPACE=save  X=discard"
        cv2.putText(out, msg, (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 230, 255), 1, cv2.LINE_AA)
        return out

    # ────────────────────────────────────────────────────────────────
    #  File I/O
    # ────────────────────────────────────────────────────────────────
    def _stem_for_current(self) -> str:
        return f"{self.video_stem}_f{self._frame_n:06d}_user"

    def _write_frame_and_label(self) -> bool:
        if self._frame is None or self._frame_n < 0:
            return False
        # Only save if we have at least one annotation (positive OR negative)
        if not self._positives and not self._negatives:
            self.last_status_msg = "Nothing to save (no clicks)"
            return False

        stem = self._stem_for_current()
        img_path = self.images_dir / f"{stem}.jpg"
        lbl_path = self.labels_dir / f"{stem}.txt"
        try:
            cv2.imwrite(str(img_path), self._frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception as e:
            self.last_status_msg = f"image write failed: {e}"
            return False

        # YOLO label format: "class cx cy w h" (all normalised 0..1).
        # Class 0 = ball (matches our fine-tuned model).
        lines = []
        bbox_norm_w = self._bbox_px / max(self._fw, 1)
        bbox_norm_h = self._bbox_px / max(self._fh, 1)
        for x, y in self._positives:
            cxn = max(0.001, min(0.999, x / self._fw))
            cyn = max(0.001, min(0.999, y / self._fh))
            lines.append(f"0 {cxn:.6f} {cyn:.6f} {bbox_norm_w:.6f} {bbox_norm_h:.6f}")
        lbl_path.write_text("\n".join(lines), encoding="utf-8")

        kind = "positive" if self._positives else "hard-negative (no-ball frame)"
        self.last_status_msg = (f"Saved {stem}  ({len(self._positives)} pos, "
                                 f"{len(self._negatives)} neg) — {kind}")
        return True

    # ────────────────────────────────────────────────────────────────
    #  Protected index — auto_label.py must not overwrite these
    # ────────────────────────────────────────────────────────────────
    def is_protected(self, stem: str) -> bool:
        return stem in self._protected

    def _load_protected(self) -> set[str]:
        if not self._protected_index.exists():
            return set()
        try:
            return set(json.loads(self._protected_index.read_text(encoding="utf-8")))
        except Exception:
            return set()

    def _save_protected(self) -> None:
        try:
            self._protected_index.parent.mkdir(parents=True, exist_ok=True)
            self._protected_index.write_text(
                json.dumps(sorted(self._protected), indent=1), encoding="utf-8")
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "active":              self.active,
            "positives_pending":   len(self._positives),
            "negatives_pending":   len(self._negatives),
            "total_saved_session": self.total_saved,
            "total_protected":     len(self._protected),
            "bbox_px":             self._bbox_px,
        }


# ────────────────────────────────────────────────────────────────────
#  Background retrain trigger
# ────────────────────────────────────────────────────────────────────
def trigger_retrain_background(dataset_root: str | Path = "training/dataset",
                                 epochs: int = 30) -> bool:
    """Spawn training/train_ball.py in a separate process. Non-blocking.
    Returns True if launch succeeded.

    IMPORTANT: passes --base handball_ball.pt when it exists, so each retrain
    INCREMENTAL fine-tunes the existing model instead of restarting from COCO.
    This preserves earlier user-correction learning."""
    import subprocess, sys
    project_root = Path(__file__).parent.parent
    train_script = project_root / "training" / "train_ball.py"
    if not train_script.exists():
        print(f"[ActiveLearning] retrain script not found at {train_script}")
        return False
    # Continue training from the current fine-tuned model if it exists.
    existing_weights = project_root / "handball_ball.pt"
    base = str(existing_weights) if existing_weights.exists() else "yolov8n.pt"
    cmd = [
        sys.executable, str(train_script),
        "--root",   str(project_root / dataset_root),
        "--base",   base,
        "--epochs", str(epochs),
        "--name",   f"ball_user_{int(time.time())}",
    ]
    print(f"[ActiveLearning] retrain base = {Path(base).name}")
    try:
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE
                          if hasattr(subprocess, "CREATE_NEW_CONSOLE") else 0)
        print(f"[ActiveLearning] retrain launched: {' '.join(cmd)}")
        return True
    except Exception as e:
        print(f"[ActiveLearning] retrain launch failed: {e}")
        return False
