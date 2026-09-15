"""rank_bm25 backend — default, pure-Python, zero build dependencies.

The on-disk sidecar format is ``bm25.json`` (see :mod:`ovgenai_retrieval.bm25.bm25_safe`).
Pickle sidecars are **refused at load time** — mirrors the faiss_safe migration
in kit-usd-agents-master (security scanners flag ``*.pkl`` files).
Call :func:`ovgenai_retrieval.bm25.bm25_safe.convert_pkl_to_json` to migrate
a legacy sidecar in place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ovgenai_retrieval.bm25.base import Bm25Index
from ovgenai_retrieval.bm25.bm25_safe import load_bm25_json, save_bm25_json
from rank_bm25 import BM25Okapi


class RankBm25Index(Bm25Index):
    backend_name = "rank_bm25"

    def __init__(self, bm25: BM25Okapi, tokenized_corpus_or_len):
        self._bm25 = bm25
        if isinstance(tokenized_corpus_or_len, int):
            self._corpus_len = tokenized_corpus_or_len
        else:
            self._corpus_len = len(tokenized_corpus_or_len)

    @classmethod
    def build(cls, corpus: list[list[str]], k1: float = 1.2, b: float = 0.75) -> "RankBm25Index":
        # rank_bm25.BM25Okapi accepts its own k1/b but the constructor signature
        # varies slightly across versions; pass through what we can.
        try:
            bm25 = BM25Okapi(corpus, k1=k1, b=b)
        except TypeError:
            bm25 = BM25Okapi(corpus)
            bm25.k1 = k1
            bm25.b = b
        return cls(bm25, corpus)

    @classmethod
    def load(cls, path: str | Path) -> "RankBm25Index":
        """Load a BM25 sidecar. JSON-only — pickle path is refused.

        To migrate legacy ``bm25.pkl`` in place, call
        :func:`ovgenai_retrieval.bm25.bm25_safe.convert_pkl_to_json`.
        """
        p = Path(path)
        if p.suffix == ".pkl" or (p.suffix == "" and p.name == "bm25.pkl"):
            json_sibling = p.with_suffix(".json")
            raise FileNotFoundError(
                f"Legacy pickle BM25 sidecar at {p} — run "
                f"ovgenai_retrieval.bm25.bm25_safe.convert_pkl_to_json('{p}') to "
                f"produce {json_sibling}. Pickle loading is disabled for security "
                f"(security scanners flag *.pkl files)."
            )
        bm25 = load_bm25_json(p)
        inst = cls.__new__(cls)
        inst._bm25 = bm25
        inst._corpus_len = int(bm25.corpus_size)
        return inst

    def save(self, path: str | Path) -> None:
        """Write the BM25 state as a safe JSON sidecar."""
        p = Path(path)
        if p.suffix == ".pkl":
            raise ValueError(
                f"Refusing to write {p}: pickle BM25 sidecars are banned " f"(security-scanner posture). Use a .json path instead."
            )
        save_bm25_json(self._bm25, p)

    def top_k(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        if not query_tokens or self._corpus_len == 0:
            return []
        scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32)
        k = min(k, scores.size)
        # Partial sort for speed on large corpora
        top_idx = np.argpartition(-scores, k - 1)[:k]
        # Then order those k by actual score desc
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx]
