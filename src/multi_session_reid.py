"""
Multi-Session Player ReID — persistent player identity across pipeline runs.

Lightweight cross-session fingerprinting using the HSV histogram that
pipeline.py already computes per track via _color_fingerprint(). No OSNet,
no FAISS — just stable 512-dim L1-normalized histograms persisted as JSON +
base64. Same player, same jersey, next match → same player_id.

The heavyweight OSNet/FAISS gallery in player_db.py remains the alternative;
this module exists for fast deploys where jersey color is already a strong
cross-session identifier (which it is in handball — uniforms are distinctive).

API
───
    msr = MultiSessionReID(ROOT / "data" / "multi_session_reid.json")

    # Per stable track each frame:
    pid, is_new = msr.match_or_enroll(track_id, fingerprint, team)
    # Returns (None, False) when the track is too new to enroll yet,
    # ("P_a3f9", True)  on first enrollment,
    # ("P_a3f9", False) on subsequent matches.

    # Coach renames a player from the dashboard:
    msr.rename("P_a3f9", "Karabatic", number=7)

    # Periodic / on-exit save:
    msr.maybe_save()   # auto-saves every SAVE_EVERY_S
    msr.save()         # force
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np


# ── Tunables ──
# We now match against a multi-sample gallery (best-of-N), so the threshold can
# be slightly higher: a player only needs to look like ANY of their previous
# angles to count as a match.
MATCH_THRESHOLD     = 0.68    # cosine similarity vs the BEST gallery sample
MIN_STABLE_FRAMES   = 24      # frames the same track_id must be observed before we attempt enrollment
SAVE_EVERY_S        = 30.0    # auto-save cadence
GALLERY_PER_PLAYER  = 12      # max stored fingerprints per player (more views = better cross-angle)
GALLERY_DIVERSITY   = 0.88    # add a new sample if it's <0.88 cosine similar to ALL existing ones (more permissive)
TEAMS               = ("team_a", "team_b", "goalkeeper")


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    # Guard against fingerprint dimensionality changes between sessions
    # (e.g., upgrading from HSV-3D to HS-2D+LAB anchor format). Mismatch → 0.
    if a.shape != b.shape:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b)) / (na * nb)


def _short_id() -> str:
    return "P_" + uuid.uuid4().hex[:6]


class MultiSessionReID:
    """Persistent player gallery keyed on jersey-color fingerprint."""

    def __init__(self, json_path: str | Path):
        self.path = Path(json_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.players: dict[str, dict] = {}            # pid → record (incl. fingerprint ndarray)
        self._auto_num: dict[str, int] = {"team_a": 1, "team_b": 1, "goalkeeper": 1}

        # session-local state (does NOT persist across runs)
        self._track_obs:    dict[int, int] = {}       # track_id → frames seen this session
        self._track_to_pid: dict[int, str] = {}       # track_id → pid (this session)

        self._last_save_t = time.monotonic()
        self._load()

    # ──────────────────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            print(f"[MultiSessionReID] no DB at {self.path}, starting fresh")
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for pid, rec in data.get("players", {}).items():
                # Support both the legacy single-fingerprint format AND the new gallery format.
                gallery: list[np.ndarray] = []
                gallery_b64 = rec.get("gallery_b64") or []
                if isinstance(gallery_b64, list):
                    for sample_b64 in gallery_b64:
                        if not sample_b64: continue
                        raw = base64.b64decode(sample_b64.encode("ascii"))
                        gallery.append(np.frombuffer(raw, dtype=np.float32).copy())
                # Legacy migration: if there's a single fingerprint_b64, lift it as the only sample.
                if not gallery and rec.get("fingerprint_b64"):
                    raw = base64.b64decode(rec["fingerprint_b64"].encode("ascii"))
                    gallery.append(np.frombuffer(raw, dtype=np.float32).copy())
                rec["gallery"] = gallery
                self.players[pid] = rec
            self._auto_num.update(data.get("auto_num", {}))
            print(f"[MultiSessionReID] loaded {len(self.players)} players (gallery sizes: "
                  f"{[len(p.get('gallery', [])) for p in self.players.values()]})")
        except Exception as e:
            print(f"[MultiSessionReID] load failed: {e}")

    def save(self) -> None:
        """Atomic write: tmp file + replace, so a concurrent reader never sees half a file."""
        try:
            payload = {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "auto_num": self._auto_num,
                "players":  {},
            }
            for pid, rec in self.players.items():
                gallery = rec.get("gallery") or []
                gallery_b64 = [
                    base64.b64encode(s.astype(np.float32).tobytes()).decode("ascii")
                    for s in gallery if s is not None
                ]
                payload["players"][pid] = {
                    "name":          rec.get("name", pid),
                    "team":          rec.get("team", "unknown"),
                    "shirt_number":  int(rec.get("shirt_number", 0)),
                    "samples":       int(rec.get("samples", 0)),
                    "first_seen":    rec.get("first_seen"),
                    "last_seen":     time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "gallery_b64":   gallery_b64,           # NEW multi-sample format
                }
            tmp = self.path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.path)
            self._last_save_t = time.monotonic()
        except Exception as e:
            print(f"[MultiSessionReID] save failed: {e}")

    def maybe_save(self) -> None:
        if (time.monotonic() - self._last_save_t) >= SAVE_EVERY_S:
            self.save()

    # ──────────────────────────────────────────────────────────────────────
    #  Core API
    # ──────────────────────────────────────────────────────────────────────
    def match_or_enroll(self, track_id: int, fingerprint: np.ndarray | None,
                        team: str) -> tuple[Optional[str], bool]:
        """Returns (player_id, is_new). (None, False) means track is not yet stable."""
        # Already mapped this session — fast path
        cached = self._track_to_pid.get(track_id)
        if cached is not None:
            if fingerprint is not None and cached in self.players:
                self._absorb_sample(cached, fingerprint)
            return cached, False

        # Stability gating
        self._track_obs[track_id] = self._track_obs.get(track_id, 0) + 1
        if self._track_obs[track_id] < MIN_STABLE_FRAMES:
            return None, False
        if fingerprint is None:
            return None, False

        # Search the persistent gallery — match against the BEST sample per player.
        # Same-team filter prevents accidental cross-team matches when uniforms
        # have similar accents.
        best_pid, best_sim = None, MATCH_THRESHOLD
        for pid, rec in self.players.items():
            if team in TEAMS and rec.get("team") != team:
                continue
            gallery = rec.get("gallery") or []
            if not gallery:
                continue
            # Best-of-N — robust to angle / lighting variation
            sim = max((_cosine(fingerprint, s) for s in gallery), default=0.0)
            if sim > best_sim:
                best_sim, best_pid = sim, pid

        if best_pid is not None:
            self._track_to_pid[track_id] = best_pid
            self._absorb_sample(best_pid, fingerprint)
            return best_pid, False

        # Enroll new player
        pid = _short_id()
        num = self._auto_num.get(team, 1)
        self._auto_num[team] = num + 1
        default_name = (
            f"{('A' if team == 'team_a' else 'B')}#{num}"
            if team in ("team_a", "team_b") else f"GK#{num}"
        )
        self.players[pid] = {
            "name":         default_name,
            "team":         team,
            "shirt_number": num,
            "samples":      1,
            "first_seen":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gallery":      [fingerprint.copy().astype(np.float32)],
        }
        self._track_to_pid[track_id] = pid
        return pid, True

    def _absorb_sample(self, pid: str, new_fp: np.ndarray) -> None:
        """Add the new fingerprint to the player's gallery — but only if it's
           DIFFERENT enough from existing samples (otherwise it's redundant)."""
        rec = self.players.get(pid)
        if rec is None:
            return
        gallery: list[np.ndarray] = rec.setdefault("gallery", [])
        rec["samples"] = int(rec.get("samples", 0)) + 1

        if not gallery:
            gallery.append(new_fp.copy().astype(np.float32))
            return

        # Diversity gate — skip if too similar to any existing sample
        max_sim = max(_cosine(new_fp, s) for s in gallery)
        if max_sim >= GALLERY_DIVERSITY:
            return  # already represented in the gallery

        if len(gallery) < GALLERY_PER_PLAYER:
            gallery.append(new_fp.copy().astype(np.float32))
        else:
            # Replace the sample most similar to its neighbors (= the most
            # redundant one) so we keep maximum diversity.
            redundancy = []
            for i, s in enumerate(gallery):
                others_max = max(
                    (_cosine(s, t) for j, t in enumerate(gallery) if j != i),
                    default=0.0,
                )
                redundancy.append((others_max, i))
            redundancy.sort(reverse=True)
            replace_idx = redundancy[0][1]
            gallery[replace_idx] = new_fp.copy().astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    #  Coach actions
    # ──────────────────────────────────────────────────────────────────────
    def rename(self, pid: str, name: str, number: Optional[int] = None) -> bool:
        rec = self.players.get(pid)
        if rec is None:
            return False
        rec["name"] = str(name)
        if number is not None:
            rec["shirt_number"] = int(number)
        return True

    def get(self, pid: str) -> Optional[dict]:
        """Public read-only view of a player record (no raw fingerprint)."""
        rec = self.players.get(pid)
        if rec is None:
            return None
        return {k: v for k, v in rec.items() if k != "fingerprint"}

    def label(self, pid: str) -> str:
        """Short display label: name (#number)."""
        rec = self.players.get(pid)
        if rec is None:
            return pid
        n = rec.get("name", pid)
        s = rec.get("shirt_number", 0)
        return f"{n}" if not s else f"{n} #{s}"

    def __len__(self) -> int:
        return len(self.players)
