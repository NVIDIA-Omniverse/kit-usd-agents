from __future__ import annotations

from pathlib import Path

import pytest
from ovgenai_retrieval.semantic import SemanticSearcher


def test_manifest_dimension_mismatch_fails_at_load(tiny_text_bundle: Path, stub_embedder):
    with pytest.raises(ValueError, match="manifest declares 2048"):
        SemanticSearcher(
            tiny_text_bundle / "index.faiss",
            stub_embedder,
            expected_dimension=2048,
        )


class _WrongDimensionEmbedder:
    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_query_dimension_mismatch_fails_before_faiss_search(tiny_text_bundle: Path):
    searcher = SemanticSearcher(tiny_text_bundle / "index.faiss", _WrongDimensionEmbedder())
    with pytest.raises(ValueError, match="returned 3.*requires 4"):
        searcher.top_k("query")


def test_faiss_row_count_mismatch_fails_at_load(tiny_text_bundle: Path, stub_embedder):
    with pytest.raises(ValueError, match="index has 5, docstore has 4"):
        SemanticSearcher(
            tiny_text_bundle / "index.faiss",
            stub_embedder,
            expected_count=4,
        )
