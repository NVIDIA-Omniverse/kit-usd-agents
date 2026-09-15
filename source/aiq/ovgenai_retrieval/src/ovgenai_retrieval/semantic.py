"""FAISS semantic-search wrapper. LangChain-free — direct faiss-cpu API."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from ovgenai_retrieval.embedder import Embedder


class SemanticSearcher:
    """Wraps a FAISS flat/HNSW/IVF index plus an embedder for the query side."""

    def __init__(
        self,
        faiss_path: str | Path,
        embedder: Embedder,
        expected_dimension: int | None = None,
        expected_count: int | None = None,
    ):
        self._index = faiss.read_index(str(faiss_path))
        self._embedder = embedder
        if expected_dimension is not None and self.dimension != expected_dimension:
            raise ValueError(
                f"FAISS dimension mismatch for {faiss_path}: index has "
                f"{self.dimension}, manifest declares {expected_dimension}. Reindex "
                "the bundle with the configured embedding model."
            )
        if expected_count is not None and self.ntotal != expected_count:
            raise ValueError(
                f"FAISS row count mismatch for {faiss_path}: index has {self.ntotal}, "
                f"docstore has {expected_count}. Rebuild the bundle."
            )

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    def top_k(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return ``[(faiss_row_idx, similarity_score), ...]`` sorted desc.

        NVIDIA retrieval embeddings are L2-normalized by convention. We return
        inner products directly and negate L2 distances so larger scores are
        always better.
        """
        if self.ntotal == 0 or k <= 0:
            return []
        vec = np.asarray([self._embedder.embed_query(query)], dtype=np.float32)
        if vec.ndim != 2 or vec.shape[1] != self.dimension:
            actual = vec.shape[1] if vec.ndim == 2 else tuple(vec.shape)
            raise ValueError(
                f"Query embedding dimension mismatch: embedder returned {actual}, "
                f"but the FAISS index requires {self.dimension}. Reindex the bundle "
                "or configure the embedding model recorded in its manifest."
            )
        # LangChain's FAISS store uses L2 distance by default for flat indexes;
        # rag-prep builds with that default. Convert L2 -> similarity via 1 - d/2
        # after assuming unit-norm vectors. We detect by metric type on the index.
        distances, indices = self._index.search(vec, min(k, self.ntotal))
        out: list[tuple[int, float]] = []
        metric = getattr(self._index, "metric_type", faiss.METRIC_L2)
        for d, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            if metric == faiss.METRIC_INNER_PRODUCT:
                score = float(d)
            else:  # METRIC_L2: convert distance to similarity-ish score
                score = float(-d)  # higher = closer (more similar)
            out.append((int(idx), score))
        return out
