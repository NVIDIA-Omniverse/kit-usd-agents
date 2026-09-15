"""Pluggable BM25 backends."""

from ovgenai_retrieval.bm25.base import Bm25Index

__all__ = ["Bm25Index", "build_backend"]


def build_backend(
    backend: str,
    corpus: list[list[str]] | None = None,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> Bm25Index:
    """Factory — construct the named backend. Lazy-imports to avoid hard deps."""
    backend = backend.lower()
    if backend == "rank_bm25":
        from ovgenai_retrieval.bm25.rank_bm25_backend import RankBm25Index

        return RankBm25Index.build(corpus or [], k1=k1, b=b)
    if backend == "tantivy":
        from ovgenai_retrieval.bm25.tantivy_backend import TantivyIndex

        return TantivyIndex.build(corpus or [], k1=k1, b=b)
    if backend == "whoosh":
        from ovgenai_retrieval.bm25.whoosh_backend import WhooshIndex

        return WhooshIndex.build(corpus or [], k1=k1, b=b)
    raise ValueError(f"Unknown BM25 backend: {backend!r}")
