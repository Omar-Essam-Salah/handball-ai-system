"""
player_reid.py — Occlusion-robust ID recovery for BoT-SORT.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn

torch.backends.cudnn.benchmark = True


class FeatureGallery:
    def __init__(self, maxlen: int = 10):
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, feat: np.ndarray) -> None:
        n = np.linalg.norm(feat)
        self._buf.append(feat / n if n > 1e-6 else feat.copy())

    def mean_feat(self) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        m = np.mean(self._buf, axis=0).astype(np.float32)
        n = np.linalg.norm(m)
        return m / n if n > 1e-6 else m

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class GalleryEntry:
    track_id:   int
    mean_feat:  np.ndarray          
    last_cx:    float               
    last_cy:    float
    vx:         float = 0.0         
    vy:         float = 0.0
    lost_frame: int   = 0
    lost_at:    float = field(default_factory=time.monotonic)

    def predict_pos(self, frames_elapsed: int) -> tuple[float, float]:
        return (self.last_cx + self.vx * frames_elapsed,
                self.last_cy + self.vy * frames_elapsed)


class PlayerReID:
    MAX_LOST_FRAMES  = 100     
    SPATIAL_SIGMA    = 150.0   
    MATCH_THRESHOLD  = 0.85    
    
    W_VISUAL         = 0.70
    W_SPATIAL        = 0.30
    
    GALLERY_MAXLEN   = 10
    
    def __init__(self, tracker, device: str = "cuda:0"):
        self._tracker  = tracker
        self._device   = torch.device(device)

        self._galleries: dict[int, FeatureGallery]       = defaultdict(
            lambda: FeatureGallery(self.GALLERY_MAXLEN))
        self._positions: dict[int, deque[tuple[float,float]]] = defaultdict(
            lambda: deque(maxlen=5))
        self._lost:     dict[int, GalleryEntry]           = {}
        self._id_remap: dict[int, int]                    = {}
        self._prev_active: set[int]                       = set()
        self._frame_n: int                                = 0

    def update(self, detections, frame: np.ndarray) -> list:
        self._frame_n += 1
        tracks = self._tracker.update(detections, frame)
        feats  = self._tracker.get_track_features()

        active_now: set[int] = set()
        new_tracks = []

        for t in tracks:
            if t.is_ball:
                continue
            tid = t.track_id
            active_now.add(tid)

            if tid in feats:
                self._galleries[tid].push(feats[tid])
            self._positions[tid].append((t.cx, t.cy))

            if tid not in self._prev_active:
                new_tracks.append(t)

        just_lost = self._prev_active - active_now
        for lid in just_lost:
            if lid not in self._lost:
                self._freeze_lost(lid)

        stale = [lid for lid, e in self._lost.items()
                 if self._frame_n - e.lost_frame > self.MAX_LOST_FRAMES]
        for lid in stale:
            self._lost.pop(lid)
            self._galleries.pop(lid, None)

        if new_tracks and self._lost:
            self._match_new_tracks(new_tracks, feats, frame)

        for t in tracks:
            canonical = self._id_remap.get(t.track_id)
            if canonical is not None:
                t.track_id = canonical

        self._prev_active = active_now
        return tracks

    def extract_crop_embedding(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        model = self._tracker.reid_model
        if model is None:
            return None

        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
        if crop.size == 0:
            return None
        return self._embed_crops([crop])[0]

    def batch_embed_crops(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[Optional[np.ndarray]]:
        h, w = frame.shape[:2]
        crops, valid_idx = [], []
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            crop = frame[max(int(y1),0):min(int(y2),h), max(int(x1),0):min(int(x2),w)]
            if crop.size > 0:
                crops.append(crop)
                valid_idx.append(i)

        out: list[Optional[np.ndarray]] = [None] * len(bboxes)
        if not crops:
            return out
        embeddings = self._embed_crops(crops)
        for idx, emb in zip(valid_idx, embeddings):
            out[idx] = emb
        return out

    def _freeze_lost(self, lid: int) -> None:
        gal   = self._galleries.get(lid)
        mfeat = gal.mean_feat() if gal else None
        if mfeat is None:
            return
        pos  = list(self._positions[lid])
        vx, vy = self._velocity(pos)
        lx, ly = pos[-1] if pos else (0.0, 0.0)
        self._lost[lid] = GalleryEntry(
            track_id=lid, mean_feat=mfeat,
            last_cx=lx, last_cy=ly, vx=vx, vy=vy,
            lost_frame=self._frame_n,
        )

    def _match_new_tracks(self, new_tracks: list, feats: dict, frame: np.ndarray) -> None:
        lost_ids  = list(self._lost.keys())
        lost_mat  = np.stack([self._lost[lid].mean_feat for lid in lost_ids])  

        for t in new_tracks:
            tid    = t.track_id
            q_feat = feats.get(tid)
            if q_feat is None:
                q_feat = self._galleries[tid].mean_feat()
            if q_feat is None:
                continue

            q_norm = q_feat / (np.linalg.norm(q_feat) + 1e-6)
            cos_sims = lost_mat @ q_norm                           

            dt       = np.array([self._frame_n - self._lost[lid].lost_frame
                                 for lid in lost_ids], np.float32)
            pred_cx  = np.array([self._lost[lid].last_cx + self._lost[lid].vx * d
                                 for lid, d in zip(lost_ids, dt)], np.float32)
            pred_cy  = np.array([self._lost[lid].last_cy + self._lost[lid].vy * d
                                 for lid, d in zip(lost_ids, dt)], np.float32)
            dists    = np.hypot(pred_cx - t.cx, pred_cy - t.cy)
            spatial  = np.exp(-(dists ** 2) / (2 * self.SPATIAL_SIGMA ** 2))

            combined  = self.W_VISUAL * cos_sims + self.W_SPATIAL * spatial
            best_i    = int(np.argmax(combined))
            if combined[best_i] < self.MATCH_THRESHOLD:
                continue

            orig_id = lost_ids[best_i]
            self._id_remap[tid] = orig_id
            self._galleries[orig_id] = self._galleries.pop(tid, FeatureGallery(self.GALLERY_MAXLEN))
            self._positions[orig_id] = self._positions.pop(tid, deque(maxlen=5))
            self._lost.pop(orig_id)

    def _embed_crops(self, crops: list[np.ndarray]) -> list[Optional[np.ndarray]]:
        model = self._tracker.reid_model
        if model is None:
            return [None] * len(crops)
        try:
            import torchvision.transforms.functional as TF
            from PIL import Image

            tensors = []
            for bgr in crops:
                pil = Image.fromarray(bgr[:, :, ::-1])
                t   = TF.to_tensor(TF.resize(pil, [256, 128]))
                tensors.append(t)

            batch = torch.stack(tensors).to(self._device)
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=(self._device.type == "cuda")):
                    feats = model(batch)                           
            return [f.cpu().float().numpy() for f in feats]
        except Exception:
            return [None] * len(crops)

    @staticmethod
    def _velocity(positions: list[tuple[float, float]]) -> tuple[float, float]:
        if len(positions) < 2:
            return 0.0, 0.0
        dxs = [positions[i+1][0] - positions[i][0] for i in range(len(positions) - 1)]
        dys = [positions[i+1][1] - positions[i][1] for i in range(len(positions) - 1)]
        return float(np.mean(dxs)), float(np.mean(dys))

    def diagnostics(self) -> dict:
        return {
            "frame":        self._frame_n,
            "active_tracks": len(self._prev_active),
            "lost_count":   len(self._lost),
            "remaps":       dict(self._id_remap),
            "lost_ids":     list(self._lost.keys()),
        }