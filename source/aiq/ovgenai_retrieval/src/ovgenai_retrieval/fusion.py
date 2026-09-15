"""Reciprocal Rank Fusion for combining semantic + lexical result lists.

RRF reference: Cormack, Clarke, Buettcher (2009). ``score = sum(1 / (k + rank_i))``
across the contributing ranked lists. The common choice ``k=60`` smooths the
tail so that list-1 rank 3 and list-2 rank 1 don't completely swamp each other.

Per-list weights (REQ-HS-3) generalize the score to::

    score(doc) = sum_i weights[i] * 1 / (k + rank_i(doc))

This lets callers bias fusion toward one side (e.g., SRD default 0.4 lexical /
0.6 semantic) without touching the recall pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence


def rrf_fuse(
    ranked_lists: Iterable[list[int]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists of integer IDs via (weighted) RRF.

    Args:
        ranked_lists: An iterable of ranked lists; each inner list is a ranking
            of ID integers, best first.
        k: RRF constant (default 60).
        weights: Optional per-list weights. When provided, its length MUST
            match ``len(ranked_lists)`` or a ``ValueError`` is raised. When
            omitted, all lists contribute with unit weight (classical RRF).

    Returns:
        ``[(id, rrf_score), ...]`` sorted by descending score.
    """
    lists = list(ranked_lists)
    if weights is not None:
        weights = list(weights)
        if len(weights) != len(lists):
            raise ValueError(f"weights length {len(weights)} does not match ranked_lists length {len(lists)}")
    else:
        weights = [1.0] * len(lists)

    scores: dict[int, float] = defaultdict(float)
    for lst, w in zip(lists, weights):
        for rank, item in enumerate(lst):
            scores[item] += w * (1.0 / (k + rank + 1))  # ranks are 1-indexed for RRF
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
