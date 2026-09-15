"""Whoosh backend — opt-in. Pure-Python alternative if rank_bm25 proves too slow."""

from __future__ import annotations

from pathlib import Path

from ovgenai_retrieval.bm25.base import Bm25Index


class WhooshIndex(Bm25Index):
    backend_name = "whoosh"

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Whoosh backend is not yet implemented. Install with "
            "`pip install ovgenai-retrieval[whoosh]` and re-check in a future release."
        )

    @classmethod
    def build(cls, corpus, k1: float = 1.2, b: float = 0.75) -> "WhooshIndex":
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "WhooshIndex":
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def top_k(self, query_tokens, k: int = 10):
        raise NotImplementedError
