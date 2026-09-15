"""Tests for :mod:`ovgenai_retrieval.embedder`.

Covers REQ-IX-4 local-first embedder: ``base_url`` pass-through for on-prem
NIM + ``backend="sentence_transformers"`` local wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ovgenai_retrieval.embedder import EmbedderFactory, SentenceTransformersEmbedder

# ---------------------------------------------------------------------------
# backend="nvidia" (default) — hosted + NIM base_url pass-through
# ---------------------------------------------------------------------------


def test_create_nvidia_default_forwards_model_and_key(monkeypatch):
    """Baseline: no base_url, api_key supplied — behaves like the old create()."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings") as Mocked:
        Mocked.return_value = MagicMock()
        emb = EmbedderFactory.create(model="nvidia/nemotron-3-embed-1b", api_key="sk-test")
        assert emb is Mocked.return_value
        Mocked.assert_called_once_with(model="nvidia/nemotron-3-embed-1b", api_key="sk-test")


def test_create_nvidia_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "env-sk-123")
    with patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings") as Mocked:
        EmbedderFactory.create()
        Mocked.assert_called_once_with(model="nvidia/nemotron-3-embed-1b", api_key="env-sk-123")


def test_create_nvidia_base_url_forwarded(monkeypatch):
    """REQ-IX-4: base_url is threaded through to NVIDIAEmbeddings."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings") as Mocked:
        EmbedderFactory.create(
            model="nvidia/nemotron-3-embed-1b",
            api_key="sk-test",
            base_url="https://nim.internal:8000/v1",
        )
        Mocked.assert_called_once_with(
            model="nvidia/nemotron-3-embed-1b",
            api_key="sk-test",
            base_url="https://nim.internal:8000/v1",
        )


def test_create_nvidia_base_url_allows_missing_api_key(monkeypatch):
    """Auth-less on-prem NIM: no api_key required when base_url is set."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings") as Mocked:
        EmbedderFactory.create(
            model="nvidia/nemotron-3-embed-1b",
            base_url="https://nim.internal:8000/v1",
        )
        # Only model + base_url; no api_key kwarg.
        Mocked.assert_called_once_with(
            model="nvidia/nemotron-3-embed-1b",
            base_url="https://nim.internal:8000/v1",
        )


def test_create_nvidia_missing_key_no_base_url_raises(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        EmbedderFactory.create(model="nvidia/nemotron-3-embed-1b")


# ---------------------------------------------------------------------------
# backend="sentence_transformers" — local wrapper
# ---------------------------------------------------------------------------


def test_create_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown embedder backend"):
        EmbedderFactory.create(backend="bogus")


def test_sentence_transformers_backend_ignores_api_key_and_base_url(monkeypatch):
    """When selecting the ST backend we should NOT hit NVIDIAEmbeddings at all."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    pytest.importorskip("sentence_transformers")
    with patch("langchain_nvidia_ai_endpoints.NVIDIAEmbeddings") as Mocked:
        emb = EmbedderFactory.create(
            model="sentence-transformers/all-MiniLM-L6-v2",
            backend="sentence_transformers",
            api_key="ignored",
            base_url="also-ignored",
        )
        Mocked.assert_not_called()
    assert isinstance(emb, SentenceTransformersEmbedder)


def test_sentence_transformers_embed_roundtrip():
    """Functional smoke test against all-MiniLM-L6-v2 (pulled by [rerank] extra)."""
    pytest.importorskip("sentence_transformers")
    emb = EmbedderFactory.create(
        model="sentence-transformers/all-MiniLM-L6-v2",
        backend="sentence_transformers",
    )
    q_vec = emb.embed_query("hello world")
    assert isinstance(q_vec, list)
    assert q_vec and all(isinstance(x, float) for x in q_vec)

    d_mat = emb.embed_documents(["alpha", "beta gamma"])
    assert len(d_mat) == 2
    assert all(isinstance(v, list) and len(v) == len(q_vec) for v in d_mat)
    assert emb.embed_documents([]) == []


def test_sentence_transformers_missing_dep_raises_clean_error():
    """If sentence_transformers is unavailable, SentenceTransformersEmbedder() must
    surface a RuntimeError (not a bare ImportError) so operators know which
    extra to install."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("sentence_transformers"):
            raise ImportError("simulated missing dep")
        return real_import(name, *a, **kw)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="sentence_transformers"):
            EmbedderFactory.create(
                model="whatever",
                backend="sentence_transformers",
            )
