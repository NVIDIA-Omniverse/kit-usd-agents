"""Tests for :mod:`ovgenai_retrieval.rerank`.

Covers REQ-ST-6 NvidiaReranker wrapping the hosted NVIDIARerank endpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ovgenai_retrieval.rerank import NvidiaReranker


def _mk_doc(page_content: str, original_index: int, relevance_score: float):
    """Build a Document-like object with the metadata NvidiaReranker reads."""
    from langchain_core.documents import Document

    return Document(
        page_content=page_content,
        metadata={"_i": original_index, "relevance_score": relevance_score},
    )


def test_nvidia_reranker_empty_candidates_returns_empty_without_impl():
    """No candidates => no endpoint call, no impl construction."""
    rr = NvidiaReranker()
    with patch("langchain_nvidia_ai_endpoints.NVIDIARerank") as Mocked:
        scores = rr.score("query", [])
        Mocked.assert_not_called()
    assert scores == []


def test_nvidia_reranker_preserves_candidate_order(monkeypatch):
    """NVIDIARerank may return docs in reranked order; our score() must map
    relevance_score back to the original candidate index."""
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")

    fake_impl = MagicMock()
    # Endpoint returns documents in reranked order (best first).
    # candidate 2 -> 0.9, candidate 0 -> 0.5, candidate 1 -> 0.1
    fake_impl.compress_documents.return_value = [
        _mk_doc("c", original_index=2, relevance_score=0.9),
        _mk_doc("a", original_index=0, relevance_score=0.5),
        _mk_doc("b", original_index=1, relevance_score=0.1),
    ]

    with patch("langchain_nvidia_ai_endpoints.NVIDIARerank", return_value=fake_impl) as Mocked:
        rr = NvidiaReranker()
        scores = rr.score("query", ["a", "b", "c"])

    # Scores must be returned in CANDIDATE order, not reranked order.
    assert scores == [0.5, 0.1, 0.9]
    Mocked.assert_called_once()
    # The call should include model + api_key (from env).
    kwargs = Mocked.call_args.kwargs
    assert kwargs["model"] == "nvidia/llama-nemotron-rerank-vl-1b-v2"
    assert kwargs["api_key"] == "sk-test"
    # compress_documents should be called with a 3-element docs list.
    cd_kwargs = fake_impl.compress_documents.call_args.kwargs
    assert len(cd_kwargs["documents"]) == 3
    assert cd_kwargs["query"] == "query"


def test_nvidia_reranker_missing_index_defaults_to_zero(monkeypatch):
    """If the endpoint drops/strips candidates, the unscored slots stay at 0.0."""
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")

    fake_impl = MagicMock()
    fake_impl.compress_documents.return_value = [
        _mk_doc("only-this-one", original_index=1, relevance_score=0.77),
    ]
    with patch("langchain_nvidia_ai_endpoints.NVIDIARerank", return_value=fake_impl):
        rr = NvidiaReranker()
        scores = rr.score("q", ["a", "b", "c"])

    assert scores == [0.0, 0.77, 0.0]


def test_nvidia_reranker_normalizes_nim_root_url(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    fake_impl = MagicMock()
    fake_impl.compress_documents.return_value = []
    with patch("langchain_nvidia_ai_endpoints.NVIDIARerank", return_value=fake_impl) as Mocked:
        rr = NvidiaReranker(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            api_key="explicit",
            base_url="https://nim.internal:8000",
        )
        rr.score("q", ["x"])

    kwargs = Mocked.call_args.kwargs
    assert kwargs["model"] == "nvidia/llama-nemotron-rerank-vl-1b-v2"
    assert kwargs["api_key"] == "explicit"
    assert kwargs["base_url"] == "https://nim.internal:8000/v1"


def test_nvidia_reranker_accepts_full_nim_ranking_url(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    fake_impl = MagicMock()
    fake_impl.compress_documents.return_value = []
    with patch("langchain_nvidia_ai_endpoints.NVIDIARerank", return_value=fake_impl) as Mocked:
        rr = NvidiaReranker(
            api_key="explicit",
            base_url="https://nim.internal:8000/v1/ranking/",
        )
        rr.score("q", ["x"])

    assert Mocked.call_args.kwargs["base_url"] == "https://nim.internal:8000/v1"


def test_nvidia_reranker_ensure_is_cached(monkeypatch):
    """_ensure must build the impl exactly once across repeated score() calls."""
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")

    fake_impl = MagicMock()
    fake_impl.compress_documents.return_value = [
        _mk_doc("only", original_index=0, relevance_score=1.0),
    ]
    with patch("langchain_nvidia_ai_endpoints.NVIDIARerank", return_value=fake_impl) as Mocked:
        rr = NvidiaReranker()
        rr.score("q1", ["a"])
        rr.score("q2", ["b"])

    assert Mocked.call_count == 1
