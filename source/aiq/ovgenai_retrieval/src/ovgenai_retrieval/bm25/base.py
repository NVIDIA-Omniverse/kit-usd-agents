"""Abstract BM25 backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


def default_tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokenizer matching the default manifest tokenizer."""
    return [t for t in text.lower().split() if t]


class Bm25Index(ABC):
    """A BM25-style lexical index.

    Backends differ in storage (pickle, Tantivy native, Whoosh index) but expose
    the same ``top_k`` API. Each position in the corpus corresponds to one doc,
    and :meth:`top_k` returns integer positions, not doc IDs.
    """

    backend_name: str = "abstract"

    @classmethod
    @abstractmethod
    def build(cls, corpus: list[list[str]], k1: float = 1.2, b: float = 0.75) -> "Bm25Index":
        """Build from an in-memory tokenized corpus."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Bm25Index":
        """Load a sidecar from disk."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist a sidecar to disk."""

    @abstractmethod
    def top_k(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        """Return ``[(corpus_idx, score), ...]`` sorted by descending score."""

    def top_k_text(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Convenience wrapper that applies the default tokenizer."""
        return self.top_k(default_tokenize(query), k=k)
