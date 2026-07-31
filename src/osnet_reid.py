"""
OSNet-Based Player ReID — replaces HSV fingerprint with deep features.

Why this exists
───────────────
The HSV color-histogram fingerprint (in pipeline._color_fingerprint) collapses
when the court colour matches the jersey colour — the entire "Clone Army"
issue from the Sweden vs Brazil broadcast. Deep ReID embeddings (OSNet trained
on MSMT17) are robust to:
    • Court colour bleed
    • Lighting changes
    • Camera angle shifts
    • Partial occlusions

OSNet x0.25 is the SMALLEST variant (~2.2 MB) — ~5 ms per crop on GPU.
For a typical handball broadcast (10-14 visible players, infer every 2 frames),
total ReID cost is ~50-70 ms/sec — well within budget.

API
───
    reid = OSNetReID(device="cuda:0")
    feat = reid.embed_crop(player_bgr_crop)   # → (512,) float32, L2-normalised
    feats = reid.embed_batch(list_of_crops)   # → (N, 512) float32

The features are cosine-comparable: closer to 1 = same player.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


class OSNetReID:
    """Lightweight OSNet-x0.25 wrapper for player ReID embeddings."""

    def __init__(
        self,
        weights: str  = "osnet_x0_25_msmt17.pt",
        device:  str  = "cuda:0",
        half:    bool = True,
    ):
        self.device = device
        self._reid  = None
        self._half  = bool(half) and ("cuda" in str(device).lower())
        self._weights_path = weights
        self._init_failed  = False

    def _ensure_loaded(self) -> bool:
        if self._reid is not None:
            return True
        if self._init_failed:
            return False
        try:
            from boxmot.reid import ReID
            self._reid = ReID(path=self._weights_path,
                              device=self.device,
                              half=self._half)
            print(f"[OSNetReID] loaded {self._weights_path} on {self.device}")
            return True
        except Exception as e:
            print(f"[OSNetReID] init failed: {type(e).__name__}: {e}")
            self._init_failed = True
            return False

    # ────────────────────────────────────────────────────────────────
    def embed_crop(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Single-crop convenience. Returns 512-d L2-normalised vector or None."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        feats = self.embed_batch([crop_bgr])
        if feats is None or feats.size == 0:
            return None
        return feats[0]

    def embed_batch(self, crops: list[np.ndarray]) -> Optional[np.ndarray]:
        """Batch embed several crops. Returns (N, 512) float32 or None on failure."""
        if not crops:
            return None
        if not self._ensure_loaded():
            return None
        try:
            feats = self._reid.process({"mode": "crops", "crops": crops})
            return feats if feats.size else None
        except Exception as e:
            if not getattr(self, "_err_logged", False):
                print(f"[OSNetReID] embed_batch failed: {type(e).__name__}: {e}")
                self._err_logged = True
            return None

    def embed_image_boxes(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Embed multiple boxes in a single image (fastest path). Returns
        (N, 512) float32 of L2-normalised features. Boxes are xyxy float32."""
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return None
        if not self._ensure_loaded():
            return None
        try:
            feats = self._reid.process({
                "mode":   "image_boxes",
                "image":  image_bgr,
                "boxes":  np.asarray(boxes_xyxy, dtype=np.float32),
            })
            if feats is None or feats.size == 0:
                return None
            # boxmot may not L2-normalise the image_boxes path; do it ourselves
            norms = np.linalg.norm(feats, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            return (feats / norms).astype(np.float32)
        except Exception as e:
            if not getattr(self, "_err_img_logged", False):
                print(f"[OSNetReID] embed_image_boxes failed: "
                      f"{type(e).__name__}: {e}")
                self._err_img_logged = True
            return None
