"""Tantivy backend — opt-in. Fast Rust-backed BM25. Requires ``pip install tantivy``."""

from __future__ import annotations

from pathlib import Path

from ovgenai_retrieval.bm25.base import Bm25Index


class TantivyIndex(Bm25Index):
    backend_name = "tantivy"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Tantivy backend is not yet implemented. Install with "
            "`pip install ovgenai-retrieval[tantivy]` and re-check in a future release."
        )

    @classmethod
    def build(cls, corpus, k1: float = 1.2, b: float = 0.75) -> "TantivyIndex":
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "TantivyIndex":
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def top_k(self, query_tokens, k: int = 10):
        raise NotImplementedError
