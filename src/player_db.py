"""
Player Recognition Database — persistent, appearance-robust gallery.

Speed strategy
──────────────
  BotSort already computes 512-dim ReID embeddings (osnet_x0_25_msmt17) for
  every person track each frame.  We reuse those at zero extra GPU cost.

  Per-frame cost: ~0.5 ms total (FAISS cosine search; numpy fallback also fast).

Robustness strategy
───────────────────
  Hard conditions (different shirt, hair, angle) are handled by:
    1. VectorStore + FAISS    — cosine search over all stored embeddings (up to
                                 MAX_SAMPLES per player). IndexFlatIP gives exact
                                 top-K in < 0.5 ms for a 50-player gallery.
    2. FewShotRecognizer       — prototypical top-K re-ranking per player.
       Averaging the best-K hits per player is far more stable than mean-only.
    3. Body-proportion cue     — height/width aspect ratio stored per player.
       Colour-blind second channel (tall players stay tall).
    4. Temporal lock           — once matched, hold for LOCK_FRAMES unless a
       much better candidate appears (hysteresis).
    5. Auto-enrollment         — ALL stable tracks (seen 45+ frames) are enrolled
       automatically, not just manually labelled ones.
    6. Cross-session identity  — before creating a new "Player N" slot, the
       historical gallery is checked at a high threshold (0.62).  Returning
       players are merged into their existing record instead of creating a
       duplicate "Player N" entry.

Database layout
───────────────
  data/player_db/players.json      player metadata
  data/player_db/vectors.npy       all embeddings (FAISS / numpy source of truth)
  data/player_db/pids.pkl          player_id per embedding row
  data/player_db/<pid>.npy         legacy files — auto-migrated on first load
"""

import base64
import json
import uuid
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

try:
    from vector_store import VectorStore
    from few_shot_recognizer import FewShotRecognizer
except ImportError:
    from src.vector_store import VectorStore
    from src.few_shot_recognizer import FewShotRecognizer


# ── Tuning constants ──────────────────────────────────────────────────────────
RECOG_EVERY    = 5      # run full matching every N frames
CACHE_FRAMES   = 120    # matched result trusted for this many frames
LOCK_FRAMES    = 150    # once locked, only override if rival conf > cur+HYSTERESIS
HYSTERESIS     = 0.12   # minimum improvement required to break a temporal lock
MIN_CONF       = 0.50   # base cosine similarity to accept a match
MIN_ENROLL     = 5      # min stored embeddings before a player is queryable
MAX_SAMPLES    = 256    # rolling window of stored embeddings per player
TOPK           = 8      # average top-K gallery scores (re-ranking)
MIN_FEAT_NORM  = 0.1    # reject near-zero / bad feature vectors
PROP_WEIGHT    = 0.15   # body-proportion cue weight (0 = disabled)
AUTO_ENROLL_FRAMES = 45 # frames a track must be stable before auto-enroll


@dataclass
class PlayerProfile:
    player_id:    str
    name:         str
    shirt_number: int   = 0
    team:         str   = "unknown"
    role:         str   = ""
    sample_count: int   = 0
    mean_emb:     np.ndarray = field(default_factory=lambda: np.zeros(512, np.float32))
    aspect_ratio: float = 0.0   # median height/width from enrollment crops
    thumb_jpg:    bytes = b""


class PlayerDatabase:
    """
    Persistent, appearance-robust player gallery backed by FAISS VectorStore.

    Quick reference
    ───────────────
        db = PlayerDatabase(ROOT / "data" / "player_db")

        # each frame
        features = tracker.get_track_features()      # free BotSort embeddings
        proportions = {tid: (h/w) for tid, (h,w) in track_sizes.items()}

        matches = db.recognize(features, frame_n, proportions)
        # → {track_id: (player_id, confidence)}

        # auto-enroll stable new tracks
        db.auto_enroll(track_id, team, features, frame_n, proportion, frame_crop)

        # manual enroll when coach labels a player
        pid = db.get_or_create(name, team, shirt_number)
        db.enroll_features(pid, features[track_id], proportion)
    """

    def __init__(self, save_dir: str | Path):
        self._dir        = Path(save_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._profiles:  dict[str, PlayerProfile] = {}
        self._props:     dict[str, list[float]]    = {}   # pid → aspect ratios
        self._cache:     dict[int, tuple]          = {}   # tid → (pid, conf, frame_n, locked_until)
        self._obs:       dict[int, int]            = {}   # tid → frames seen
        self._auto_nums: dict[str, int]            = {}   # team → next sequential number
        self._store      = VectorStore(dim=512, save_dir=self._dir)
        self._recognizer = FewShotRecognizer(min_conf=MIN_CONF)
        self._load_json()

    # ── Auto-enrollment ───────────────────────────────────────────────────────

    def tick_observation(self, track_id: int) -> int:
        """Increment observation counter. Returns total frames seen for this track."""
        self._obs[track_id] = self._obs.get(track_id, 0) + 1
        return self._obs[track_id]

    def clear_observation(self, track_id: int) -> None:
        self._obs.pop(track_id, None)

    def next_shirt_number(self, team: str) -> int:
        """Return and increment the next auto-assigned shirt number for a team."""
        n = self._auto_nums.get(team, 1)
        self._auto_nums[team] = n + 1
        return n

    def auto_enroll(
        self,
        track_id:   int,
        team:       str,
        features:   dict[int, np.ndarray],
        frame_n:    int,
        proportion: float = 0.0,
        crop_bgr:   np.ndarray | None = None,
    ) -> tuple[str, bool]:
        """
        Called every frame for each non-ball track.
        Returns (player_id, is_new) once stable enough to enroll.
        Returns ("", False) if track is not yet stable.

        If the track is already cached (recognised), enroll more features for it.
        If not, once it reaches AUTO_ENROLL_FRAMES:
          1. Try to match against existing gallery.
          2. Try cross-session identity check (high threshold) to prevent duplicates.
          3. If both fail, create a new "Player N" slot.
        """
        count = self.tick_observation(track_id)
        feat  = features.get(track_id)

        # Already matched → keep enrolling features
        cached = self._cache.get(track_id)
        if cached:
            pid = cached[0]
            if feat is not None:
                self.enroll_features(pid, feat, proportion)
            if crop_bgr is not None and not self._profiles[pid].thumb_jpg:
                self.update_thumbnail(pid, crop_bgr)
            return pid, False

        if count < AUTO_ENROLL_FRAMES:
            return "", False

        # Stable enough — try gallery match first
        if feat is not None and self._store.total > 0:
            norm = float(np.linalg.norm(feat))
            if norm >= MIN_FEAT_NORM:
                result = self._recognizer.match(
                    feat, self._store, min_samples=MIN_ENROLL)
                if result:
                    pid = result.player_id
                    self._cache[track_id] = (pid, result.score, frame_n,
                                             frame_n + LOCK_FRAMES)
                    self.enroll_features(pid, feat, proportion)
                    return pid, False

        # Cross-session identity check — prevent duplicate "Player N"
        if feat is not None and self._store.total > 0:
            norm = float(np.linalg.norm(feat))
            if norm >= MIN_FEAT_NORM:
                existing = self._recognizer.cross_session_match(
                    feat, self._store, min_samples=MIN_ENROLL)
                if existing:
                    self._cache[track_id] = (existing, 1.0, frame_n,
                                             frame_n + LOCK_FRAMES)
                    self.enroll_features(existing, feat, proportion)
                    print(f"[PlayerDB] Cross-session match: track={track_id} → "
                          f"{existing} ({self._profiles[existing].name})")
                    return existing, False

        # Brand new player
        num  = self.next_shirt_number(team)
        name = f"Player {num}"
        pid  = self.add_player(name, team, shirt_number=num)
        if feat is not None:
            self.enroll_features(pid, feat, proportion)
        if crop_bgr is not None:
            self.update_thumbnail(pid, crop_bgr)
        self._cache[track_id] = (pid, 1.0, frame_n, frame_n + LOCK_FRAMES)
        print(f"[PlayerDB] Auto-enrolled {name} (team={team}) track={track_id}")
        return pid, True

    # ── Recognition ───────────────────────────────────────────────────────────

    def recognize(
        self,
        features:    dict[int, np.ndarray],
        frame_n:     int,
        proportions: dict[int, float] | None = None,
    ) -> dict[int, tuple]:
        """
        features    : {track_id: embedding}   from tracker.get_track_features()
        proportions : {track_id: h/w ratio}   optional body-proportion cue
        Returns     : {track_id: (player_id, confidence)}

        Full gallery scan runs every RECOG_EVERY frames.
        Cached results are returned on intermediate frames.
        Temporal lock: a previously matched track keeps its identity unless a
        much stronger rival appears.
        """
        if proportions is None:
            proportions = {}

        result: dict[int, tuple] = {}

        # Return still-valid cached results
        for tid, (pid, conf, fn, locked_until) in list(self._cache.items()):
            if tid in features and (frame_n - fn) < CACHE_FRAMES:
                result[tid] = (pid, conf)

        if frame_n % RECOG_EVERY != 0:
            return result

        if self._store.total == 0:
            return result

        to_match = [
            tid for tid in features
            if tid not in self._cache
            or (frame_n - self._cache[tid][2]) >= CACHE_FRAMES
        ]
        if not to_match:
            return result

        k = min(64, self._store.total)

        for tid in to_match:
            v = features[tid]
            norm = float(np.linalg.norm(v))
            if norm < MIN_FEAT_NORM:
                continue

            raw = self._store.search(v, k=k)
            if not raw:
                continue

            scores = self._scores_from_search(raw)

            # Optional body-proportion penalty
            if PROP_WEIGHT > 0 and proportions.get(tid, 0.0) > 0:
                qp = proportions[tid]
                for gid in list(scores):
                    gp = self._profiles[gid].aspect_ratio if gid in self._profiles else 0.0
                    if gp > 0:
                        prop_sim = 1.0 - min(abs(qp - gp) / max(qp, gp), 1.0)
                        scores[gid] = (1 - PROP_WEIGHT) * scores[gid] + PROP_WEIGHT * prop_sim

            if not scores:
                self._cache.pop(tid, None)
                continue

            best_pid  = max(scores, key=scores.get)
            best_conf = scores[best_pid]

            # Temporal lock: require hysteresis to override a locked identity
            prev = self._cache.get(tid)
            if prev and frame_n < prev[3]:
                prev_pid, prev_conf = prev[0], prev[1]
                if best_pid != prev_pid and (best_conf - prev_conf) < HYSTERESIS:
                    result[tid] = (prev_pid, prev_conf)
                    continue

            if best_conf >= MIN_CONF:
                result[tid]      = (best_pid, best_conf)
                self._cache[tid] = (best_pid, best_conf, frame_n,
                                    frame_n + LOCK_FRAMES)
            else:
                self._cache.pop(tid, None)

        return result

    def invalidate(self, track_id: int) -> None:
        """Call when a track is lost — clears cache and observation counter."""
        self._cache.pop(track_id, None)
        self._obs.pop(track_id, None)

    def search_by_embedding(
        self,
        feat: np.ndarray,
        k: int = 16,
        min_samples: int = 5,
    ) -> list[tuple[str, float]]:
        """
        FAISS cosine search. Returns [(player_id, similarity), ...] for
        players with >= min_samples enrolled embeddings.
        Intended for TrackBridge direct re-identification.
        """
        if self._store.total == 0:
            return []
        norm = float(np.linalg.norm(feat))
        if norm < MIN_FEAT_NORM:
            return []
        raw = self._store.search(feat / norm, k=k)
        return [(pid, sim) for pid, sim in raw
                if self._store.player_count(pid) >= min_samples]

    # ── Enrollment ────────────────────────────────────────────────────────────

    def enroll_features(
        self,
        player_id:  str,
        feat:       np.ndarray | None,
        proportion: float = 0.0,
    ) -> bool:
        """Add one embedding to a player's gallery. Returns True on success."""
        if feat is None or player_id not in self._profiles:
            return False
        norm = float(np.linalg.norm(feat))
        if norm < MIN_FEAT_NORM:
            return False

        self._store.add(player_id, feat)   # handles normalisation + rolling window

        prof = self._profiles[player_id]
        prof.sample_count = self._store.player_count(player_id)

        # Update prototype stored in profile (used by API / analytics)
        embs = self._store.get_embeddings(player_id)
        if embs.shape[0] > 0:
            prof.mean_emb = FewShotRecognizer.build_prototype(embs)

        if proportion > 0:
            lst = self._props.setdefault(player_id, [])
            lst.append(proportion)
            if len(lst) > 64:
                lst.pop(0)
            prof.aspect_ratio = float(np.median(lst))

        return True

    def enroll_crop(self, player_id: str, crop_bgr: np.ndarray,
                    reid_model) -> bool:
        """Fallback: extract embedding from a crop and enroll (pre-match photos)."""
        feat = _extract_crop_feature(crop_bgr, reid_model)
        if feat is None:
            return False
        ok = self.enroll_features(player_id, feat)
        if ok and not self._profiles[player_id].thumb_jpg:
            self.update_thumbnail(player_id, crop_bgr)
        return ok

    def update_thumbnail(self, player_id: str, crop_bgr: np.ndarray) -> None:
        if player_id not in self._profiles:
            return
        thumb = cv2.resize(crop_bgr, (64, 128))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 65])
        self._profiles[player_id].thumb_jpg = buf.tobytes()

    # ── Player management ─────────────────────────────────────────────────────

    def get_or_create(self, name: str, team: str,
                      shirt_number: int = 0, role: str = "") -> str:
        for pid, p in self._profiles.items():
            if p.name.lower() == name.lower() and p.team == team:
                return pid
        return self.add_player(name, team, shirt_number, role)

    def add_player(self, name: str, team: str,
                   shirt_number: int = 0, role: str = "") -> str:
        pid = str(uuid.uuid4())[:8]
        self._profiles[pid] = PlayerProfile(
            player_id=pid, name=name, team=team,
            shirt_number=shirt_number, role=role,
        )
        self.save()
        return pid

    def rename_player(self, player_id: str, name: str,
                      shirt_number: int = 0) -> bool:
        """Update name / number after auto-enrollment."""
        p = self._profiles.get(player_id)
        if p is None:
            return False
        if name:
            p.name = name
        if shirt_number:
            p.shirt_number = shirt_number
        self.save()
        return True

    def delete_player(self, player_id: str) -> bool:
        if player_id not in self._profiles:
            return False
        del self._profiles[player_id]
        self._props.pop(player_id, None)
        self._store.remove_player(player_id)
        npy = self._dir / f"{player_id}.npy"   # remove legacy file if present
        if npy.exists():
            npy.unlink()
        self.save()
        self._cache = {t: v for t, v in self._cache.items() if v[0] != player_id}
        return True

    def get(self, player_id: str) -> PlayerProfile | None:
        return self._profiles.get(player_id)

    def all_profiles(self) -> list[PlayerProfile]:
        return sorted(self._profiles.values(),
                      key=lambda p: (p.team, p.shirt_number, p.name))

    def stats(self) -> dict:
        enrolled = sum(1 for p in self._profiles.values()
                       if p.sample_count >= MIN_ENROLL)
        return {
            "total":    len(self._profiles),
            "enrolled": enrolled,
            "ready":    enrolled > 0,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        data = {}
        for pid, p in self._profiles.items():
            data[pid] = {
                "name":         p.name,
                "shirt_number": p.shirt_number,
                "team":         p.team,
                "role":         p.role,
                "sample_count": p.sample_count,
                "aspect_ratio": round(p.aspect_ratio, 4),
                "mean_emb":     p.mean_emb.tolist(),
                "thumb_b64":    base64.b64encode(p.thumb_jpg).decode() if p.thumb_jpg else "",
            }
        data["__auto_nums__"] = self._auto_nums
        (self._dir / "players.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        self._store.save()

    def load(self) -> None:
        """Reload everything from disk (re-initialises VectorStore + profiles)."""
        self._store = VectorStore(dim=512, save_dir=self._dir)
        self._load_json()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_json(self) -> None:
        """Load players.json and migrate legacy per-player .npy files if needed."""
        path = self._dir / "players.json"
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._auto_nums = raw.pop("__auto_nums__", {})
            for pid, d in raw.items():
                mean  = np.array(d.get("mean_emb", [0] * 512), np.float32)
                thumb = base64.b64decode(d["thumb_b64"]) if d.get("thumb_b64") else b""
                self._profiles[pid] = PlayerProfile(
                    player_id=pid,
                    name=d["name"],
                    shirt_number=d.get("shirt_number", 0),
                    team=d.get("team", "unknown"),
                    role=d.get("role", ""),
                    sample_count=d.get("sample_count", 0),
                    aspect_ratio=d.get("aspect_ratio", 0.0),
                    mean_emb=mean,
                    thumb_jpg=thumb,
                )
                self._props.setdefault(pid, [])

            # One-time migration: load legacy <pid>.npy files → VectorStore
            if self._store.total == 0:
                migrated = 0
                for pid in self._profiles:
                    npy = self._dir / f"{pid}.npy"
                    if npy.exists():
                        arr = np.load(str(npy))
                        for row in arr:
                            self._store.add(pid, row)
                        migrated += arr.shape[0]
                if migrated:
                    self._store.save()
                    print(f"[PlayerDB] Migrated {migrated} legacy embeddings → VectorStore")

            # Sync sample_count from actual store (authoritative after migration)
            for pid in self._profiles:
                sc = self._store.player_count(pid)
                if sc > 0:
                    self._profiles[pid].sample_count = sc

            print(f"[PlayerDB] {len(self._profiles)} profiles loaded "
                  f"({self.stats()['enrolled']} ready, "
                  f"{self._store.total} total embeddings)")
        except Exception as e:
            print(f"[PlayerDB] Load failed: {e}")

    def _scores_from_search(
        self,
        raw: list[tuple[str, float]],
    ) -> dict[str, float]:
        """
        Convert VectorStore search hits to per-player top-K scores.
        Filters out players below MIN_ENROLL threshold.
        """
        by_pid: dict[str, list[float]] = {}
        for pid, sim in raw:
            if self._store.player_count(pid) < MIN_ENROLL:
                continue
            by_pid.setdefault(pid, []).append(float(sim))

        result: dict[str, float] = {}
        for pid, sims in by_pid.items():
            top = sorted(sims, reverse=True)[:TOPK]
            result[pid] = float(np.mean(top))
        return result


# ── Standalone crop feature extraction ───────────────────────────────────────

def _extract_crop_feature(crop_bgr: np.ndarray,
                           reid_model) -> np.ndarray | None:
    """
    Extract one ReID embedding from a BGR crop.
    Only for pre-match photo enrollment — live tracking reuses BotSort features.
    """
    if reid_model is None or crop_bgr is None:
        return None
    try:
        import torch
        crop = cv2.resize(crop_bgr, (128, 256))
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        rgb  = (rgb - mean) / std
        t    = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        try:
            dev = next(reid_model.parameters()).device
        except Exception:
            dev = torch.device("cpu")
        t = t.to(dev)
        with torch.no_grad():
            feat = reid_model(t)
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()
        feat = feat.flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        return feat / max(norm, 1e-8)
    except Exception as e:
        print(f"[PlayerDB] Crop feature extraction failed: {e}")
        return None
