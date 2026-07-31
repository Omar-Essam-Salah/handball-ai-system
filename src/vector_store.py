"""
VectorStore — FAISS-backed persistent embedding store.

Stores L2-normalised ReID embeddings for all players across all sessions.
Inner-product search on normalised vectors == cosine similarity.

Falls back to pure-numpy if FAISS is not installed:
    pip install faiss-cpu          # CPU-only (fast on any machine)
    pip install faiss-gpu          # GPU version (faster with NVIDIA)

Layout on disk (inside data/player_db/):
    vectors.npy   — (N, dim) float32 matrix of ALL stored embeddings
    pids.pkl      — list[str] of player_id per row (len == N)
"""

import pickle
import numpy as np
from pathlib import Path

# ── Optional FAISS ────────────────────────────────────────────────────────────
try:
    import faiss as _faiss
    _FAISS_OK = True
except ImportError:
    _faiss    = None
    _FAISS_OK = False
    print("[VectorStore] faiss not found — using numpy (pip install faiss-cpu).")


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = v.flatten().astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


class VectorStore:
    """
    Persistent gallery of L2-normalised ReID embeddings.

    Key design choices
    ──────────────────
    • numpy array is the **source of truth** (vectors.npy on disk).
    • FAISS index is a **hot cache** rebuilt only when the underlying data changes.
      On a 50-player × 256-sample gallery (12 800 vectors × 512 dim),
      index rebuild takes ~2 ms and search takes <0.5 ms — far faster than
      the pure-numpy fallback (≈4 ms matmul).
    • Rolling window MAX_PER_PLAYER: oldest samples are evicted so the
      gallery stays fresh without unbounded memory growth.
    • Works as a single cross-session store: calling save() / load() carries
      all historical samples forward into the next session.

    Usage
    ─────
        store = VectorStore(dim=512, save_dir=Path("data/player_db"))

        # every time a new ReID embedding arrives for a known/new player:
        store.add(player_id, embedding)

        # find the closest players to a query embedding:
        results = store.search(query, k=32)
        # → [(player_id, cosine_sim), ...]  sorted best-first

        # persist at end of session:
        store.save()
    """

    MAX_PER_PLAYER = 256   # rolling window per player

    def __init__(self, dim: int = 512, save_dir: Path | None = None):
        self._dim  = dim
        self._dir  = save_dir

        # Source-of-truth arrays
        self._vecs: np.ndarray = np.empty((0, dim), np.float32)
        self._pids: list[str]  = []

        # FAISS hot-cache — None means "needs rebuild"
        self._faiss_idx = None

        if save_dir:
            self._load()

    # ── Core interface ────────────────────────────────────────────────────────

    def add(self, player_id: str, embedding: np.ndarray) -> None:
        """
        Add one embedding for a player.
        Enforces rolling window: if the player already has MAX_PER_PLAYER
        samples, the oldest one is dropped.
        """
        v = _l2norm(embedding).reshape(1, self._dim)

        # Enforce rolling window
        player_rows = [i for i, p in enumerate(self._pids) if p == player_id]
        if len(player_rows) >= self.MAX_PER_PLAYER:
            oldest = player_rows[0]
            keep   = [i for i in range(len(self._pids)) if i != oldest]
            self._vecs = self._vecs[keep]
            self._pids = [self._pids[i] for i in keep]

        self._vecs = np.vstack([self._vecs, v]) if self._vecs.shape[0] else v
        self._pids.append(player_id)
        self._faiss_idx = None   # invalidate cache

    def search(
        self,
        query: np.ndarray,
        k: int = 32,
    ) -> list[tuple[str, float]]:
        """
        Return up to k nearest (player_id, cosine_similarity) pairs,
        best-first.  Cosine similarities are in [−1, 1] for raw vectors;
        because we normalise, practical range is roughly [0.3, 1.0].
        """
        n = self._vecs.shape[0]
        if n == 0:
            return []

        q = _l2norm(query).reshape(1, self._dim)
        k = min(k, n)

        if _FAISS_OK:
            idx = self._get_faiss_index()
            scores, ids = idx.search(q, k)
            return [
                (self._pids[int(i)], float(s))
                for s, i in zip(scores[0], ids[0])
                if i >= 0
            ]
        else:
            # Numpy fallback: O(N·dim) — still fast for <1000 vectors
            sims    = (q @ self._vecs.T).flatten()
            top_idx = np.argsort(sims)[::-1][:k]
            return [(self._pids[int(i)], float(sims[i])) for i in top_idx]

    def remove_player(self, player_id: str) -> int:
        """
        Remove ALL stored embeddings for a player.
        Returns the number of vectors removed.
        """
        before = len(self._pids)
        keep   = [i for i, p in enumerate(self._pids) if p != player_id]
        self._vecs      = self._vecs[keep] if keep else np.empty((0, self._dim), np.float32)
        self._pids      = [self._pids[i] for i in keep]
        self._faiss_idx = None
        return before - len(self._pids)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def player_count(self, player_id: str) -> int:
        """Number of stored embeddings for one player."""
        return sum(1 for p in self._pids if p == player_id)

    def unique_players(self) -> list[str]:
        """All player_ids that have at least one stored embedding."""
        seen: set[str] = set()
        result = []
        for p in self._pids:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result

    def get_embeddings(self, player_id: str) -> np.ndarray:
        """Return all stored embeddings for one player as (N, dim) array."""
        rows = [i for i, p in enumerate(self._pids) if p == player_id]
        if not rows:
            return np.empty((0, self._dim), np.float32)
        return self._vecs[rows]

    @property
    def total(self) -> int:
        return self._vecs.shape[0]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if not self._dir:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            np.save(str(self._dir / "vectors.npy"), self._vecs)
            with open(self._dir / "pids.pkl", "wb") as f:
                pickle.dump(self._pids, f)
        except Exception as e:
            print(f"[VectorStore] Save failed: {e}")

    def _load(self) -> None:
        vecs_p = self._dir / "vectors.npy"
        pids_p = self._dir / "pids.pkl"
        if not vecs_p.exists():
            return
        try:
            self._vecs = np.load(str(vecs_p))
            with open(pids_p, "rb") as f:
                self._pids = pickle.load(f)
            self._faiss_idx = None
            n_players = len(set(self._pids))
            print(f"[VectorStore] Loaded {self.total} vectors "
                  f"from {n_players} players across all sessions.")
        except Exception as e:
            print(f"[VectorStore] Load failed: {e}")

    # ── FAISS cache ───────────────────────────────────────────────────────────

    def _get_faiss_index(self):
        """Build (or return cached) FAISS IndexFlatIP over current _vecs."""
        if self._faiss_idx is None:
            idx = _faiss.IndexFlatIP(self._dim)
            if self._vecs.shape[0] > 0:
                idx.add(self._vecs)
            self._faiss_idx = idx
        return self._faiss_idx
