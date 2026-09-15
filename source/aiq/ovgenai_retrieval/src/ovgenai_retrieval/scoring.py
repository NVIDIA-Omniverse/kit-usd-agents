"""Confidence scoring and labeling from raw retrieval outputs."""

from __future__ import annotations

import math
from typing import Literal

# Named thresholds so callers can override or inspect (REQ-CS-2).
HIGH_RERANK_LOGIT = 5.0
MED_RERANK_LOGIT = 2.0
HIGH_FUSED_SCORE = 0.5
MED_FUSED_SCORE = 0.2

ConfidenceLabel = Literal["high", "medium", "low"]


def semantic_confidence(score: float) -> float:
    """Normalize a semantic similarity into ``[0, 1]``.

    For cosine-similarity-like inner-product scores in ``[-1, 1]`` we shift to
    ``[0, 1]``. For negated-L2 scores (``-d``) we apply a monotonic squash.
    """
    if score >= -1.0 and score <= 1.0:
        return max(0.0, (score + 1.0) / 2.0)
    # Negated L2: smaller magnitude → higher similarity
    return 1.0 / (1.0 + math.exp(-score))  # logistic


def bm25_confidence(score: float, corpus_mean_bm25: float = 2.0) -> float:
    """Squash BM25 into ``[0, 1]`` using a rough scaling vs the corpus mean."""
    return 1.0 - 1.0 / (1.0 + max(0.0, score) / max(1e-6, corpus_mean_bm25))


def combined_confidence(semantic: float | None, lexical: float | None) -> float:
    """Average whichever components are available. Returns 0.0 if both missing."""
    parts = [p for p in (semantic, lexical) if p is not None]
    return sum(parts) / len(parts) if parts else 0.0


def confidence_label(
    score_or_logit: float,
    *,
    use_logit: bool = False,
    high: float | None = None,
    medium: float | None = None,
) -> ConfidenceLabel:
    """Bucket a score into {high, medium, low} (REQ-CS-1, REQ-CS-2).

    When ``use_logit=True`` the caller supplies a raw cross-encoder logit
    (typical MS-MARCO scale: below 0 is weak, 5+ is strong). Otherwise the
    caller supplies a fused/RRF-derived confidence roughly in [0, 1].

    Threshold overrides (``high`` / ``medium``) take precedence; otherwise the
    module-level named constants apply.
    """
    if use_logit:
        hi = HIGH_RERANK_LOGIT if high is None else high
        md = MED_RERANK_LOGIT if medium is None else medium
    else:
        hi = HIGH_FUSED_SCORE if high is None else high
        md = MED_FUSED_SCORE if medium is None else medium
    if score_or_logit >= hi:
        return "high"
    if score_or_logit >= md:
        return "medium"
    return "low"


def hit_confidence_label(hit) -> ConfidenceLabel:
    """Bucket a :class:`Hit` using its best available signal.

    Prefers the cross-encoder logit (``provenance["rerank_score"]``) when
    present, falls back to the fused score.
    """
    provenance = getattr(hit, "provenance", None) or {}
    logit = provenance.get("rerank_score")
    if logit is not None:
        return confidence_label(float(logit), use_logit=True)
    return confidence_label(float(hit.score), use_logit=False)
