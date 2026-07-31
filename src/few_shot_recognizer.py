"""
FewShotRecognizer — Prototypical player matching.

Concept (Prototypical Networks, Snell et al. 2017)
──────────────────────────────────────────────────
Each player in the gallery is represented by a **prototype** — the L2-normalised
mean of all their stored support embeddings.  A query is matched to the nearest
prototype in cosine space.

Why this is better than mean-only matching
──────────────────────────────────────────
• "Mean of normalised" ≠ "normalise the mean".  Computing the prototype by
  normalising each support embedding first and then averaging — and finally
  re-normalising the mean — puts all players on the unit hypersphere.  Distances
  are then directly comparable regardless of how many samples each player has.

• Top-K re-ranking: rather than comparing the query against one prototype,
  we compare it against ALL stored embeddings and average the top-K scores
  per player.  This makes matching robust to outlier frames.

• Confidence via margin: confidence = f(score_1 − score_2).  A large gap
  between best and second-best means the network is "sure".  A small gap
  triggers hysteresis (keep old identity).

Cross-session identity resolution
──────────────────────────────────
When a NEW player is being enrolled in a session, `cross_session_match()`
runs a high-threshold check against ALL historical players.  If it finds a
strong match (≥ CROSS_SESSION_THRESH = 0.62) the new slot is merged into the
existing player record instead of creating a duplicate "Player N" entry.

This means the same person accumulates heatmaps, stats, and performance
grades continuously across matches — even if they were auto-numbered
differently in a previous session.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_SAMPLES          = 3     # minimum stored embeddings before a player is queryable
MIN_CONF             = 0.50  # minimum per-player score to accept a match
CROSS_SESSION_THRESH = 0.62  # higher bar for cross-session merging
TOPK_PER_PLAYER      = 8     # top-K hits averaged per player during re-ranking
MARGIN_SCALE         = 4.0   # confidence = min(1, margin × MARGIN_SCALE)


@dataclass
class MatchResult:
    """Result of a single recognition query."""
    player_id:  str
    score:      float   # 0-1 cosine similarity
    confidence: float   # 0-1 margin-based confidence (how certain the match is)


class FewShotRecognizer:
    """
    Offline few-shot player recognition using prototypical matching.

    Works in as few as MIN_SAMPLES (3) frames — ideal for the first seconds
    after a player enters the frame.

    Usage
    ─────
        recognizer = FewShotRecognizer()

        # per-frame recognition:
        result = recognizer.match(query_embedding, vector_store)
        if result:
            pid, score, conf = result.player_id, result.score, result.confidence

        # at auto-enroll time (check for existing identity first):
        existing_pid = recognizer.cross_session_match(query_embedding, vector_store)
        if existing_pid:
            # reuse existing player — don't create "Player N"
            pass
    """

    def __init__(
        self,
        min_conf:             float = MIN_CONF,
        cross_session_thresh: float = CROSS_SESSION_THRESH,
        topk:                 int   = TOPK_PER_PLAYER,
    ):
        self._min_conf  = min_conf
        self._cs_thresh = cross_session_thresh
        self._topk      = topk

    # ── Main matching ─────────────────────────────────────────────────────────

    def match(
        self,
        query:        np.ndarray,
        vector_store,                      # VectorStore
        min_samples:  int = MIN_SAMPLES,
        exclude_pids: set[str] | None = None,
    ) -> Optional[MatchResult]:
        """
        Match one query embedding against the full gallery.

        Steps
        ─────
        1. Search vector_store for the top-64 nearest neighbours of `query`.
        2. Filter out players with fewer than `min_samples` stored embeddings.
        3. Per remaining player: average their top-TOPK hit scores.
        4. Best player must score ≥ min_conf.
        5. Confidence = scaled margin between best and second-best player.

        Returns None if no player clears the confidence threshold.
        """
        raw = vector_store.search(query, k=min(64, max(vector_store.total, 1)))
        if not raw:
            return None

        # Group hits by player, filter ineligible players
        by_pid: dict[str, list[float]] = {}
        for pid, score in raw:
            if exclude_pids and pid in exclude_pids:
                continue
            if vector_store.player_count(pid) < min_samples:
                continue
            by_pid.setdefault(pid, []).append(score)

        if not by_pid:
            return None

        # Per-player score: mean of top-K hits
        player_scores: dict[str, float] = {}
        for pid, hits in by_pid.items():
            top = sorted(hits, reverse=True)[:self._topk]
            player_scores[pid] = float(np.mean(top))

        ranked = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
        best_pid, best_score = ranked[0]

        if best_score < self._min_conf:
            return None

        # Margin-based confidence
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin       = best_score - second_score
        confidence   = min(1.0, margin * MARGIN_SCALE)

        return MatchResult(
            player_id  = best_pid,
            score      = best_score,
            confidence = confidence,
        )

    # ── Cross-session identity resolution ────────────────────────────────────

    def cross_session_match(
        self,
        query:        np.ndarray,
        vector_store,
        exclude_pids: set[str] | None = None,
        min_samples:  int = MIN_SAMPLES,
    ) -> Optional[str]:
        """
        High-confidence cross-session identity check.

        Called when a NEW player slot is about to be created.  If a strong
        match is found in the historical gallery (score ≥ CROSS_SESSION_THRESH),
        the caller should reuse that player_id instead of creating a duplicate.

        Uses a stricter threshold than `match()` to avoid false merges.
        Returns the matched player_id, or None if no strong match found.
        """
        raw = vector_store.search(query, k=min(32, max(vector_store.total, 1)))
        if not raw:
            return None

        by_pid: dict[str, list[float]] = {}
        for pid, score in raw:
            if exclude_pids and pid in exclude_pids:
                continue
            if vector_store.player_count(pid) < min_samples:
                continue
            by_pid.setdefault(pid, []).append(score)

        if not by_pid:
            return None

        best_pid   = max(by_pid, key=lambda p: float(np.mean(
            sorted(by_pid[p], reverse=True)[:self._topk])))
        best_score = float(np.mean(sorted(by_pid[best_pid], reverse=True)[:self._topk]))

        return best_pid if best_score >= self._cs_thresh else None

    # ── Prototype utilities ───────────────────────────────────────────────────

    @staticmethod
    def build_prototype(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute the class prototype from a set of support embeddings.

        Each row is first L2-normalised (projecting onto unit sphere),
        then the normalised rows are averaged, and the mean is re-normalised.
        This is the centroid of the unit-hypersphere cluster for the player.

        Handles the edge case of a single embedding (returns it as-is).
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-8)      # (N, dim) on unit sphere

        mean = normed.mean(axis=0)                          # (dim,)
        mean_norm = float(np.linalg.norm(mean))
        return (mean / max(mean_norm, 1e-8)).astype(np.float32)

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two flat vectors."""
        a = a.flatten().astype(np.float32)
        b = b.flatten().astype(np.float32)
        denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        return float(np.dot(a, b) / max(denom, 1e-8))

    @staticmethod
    def prototype_distance(
        query:      np.ndarray,
        vector_store,
        player_id:  str,
    ) -> float:
        """
        Compute cosine similarity between a query and a specific player's
        prototype.  Returns 0.0 if the player has no stored embeddings.
        Useful for targeted re-identification after a known occlusion.
        """
        embs = vector_store.get_embeddings(player_id)
        if embs.shape[0] == 0:
            return 0.0
        proto = FewShotRecognizer.build_prototype(embs)
        return FewShotRecognizer.cosine_sim(query, proto)
