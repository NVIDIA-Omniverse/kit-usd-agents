# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-package hybrid shim helper."""

from unittest.mock import patch

import pytest
from ovgenai_retrieval.shim import (
    _LEGACY_LOCAL_RERANK_URL_ENV_VARS,
    _LEGACY_RERANK_ENV_VARS,
    _hybrid_enabled,
    _legacy_local_rerank_url,
    _rerank_backend_default,
    _rerank_enabled,
    _resolve_qe_domains,
    hits_to_documents,
    hits_to_documents_with_scores,
    maybe_load_hybrid,
)


@pytest.fixture(autouse=True)
def _clear_rerank_env(monkeypatch):
    """Most tests reason about explicit env-var presence; start each test with
    every relevant var cleared so legacy fallbacks don't leak in from the
    operator's shell."""
    for var in (
        "OVAI_RERANK",
        "OVAI_RERANK_BACKEND",
        "OVAI_RERANK_MODEL",
        *_LEGACY_RERANK_ENV_VARS,
        *_LEGACY_LOCAL_RERANK_URL_ENV_VARS,
    ):
        monkeypatch.delenv(var, raising=False)


def test_hybrid_enabled_default(monkeypatch):
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    assert _hybrid_enabled() is True


def test_hybrid_disabled_explicit_semantic(monkeypatch):
    monkeypatch.setenv("OVAI_RETRIEVAL_MODE", "semantic")
    assert _hybrid_enabled() is False


def test_hybrid_disabled_case_insensitive(monkeypatch):
    monkeypatch.setenv("OVAI_RETRIEVAL_MODE", "SEMANTIC")
    assert _hybrid_enabled() is False


def test_qe_domains_default_csv(monkeypatch):
    monkeypatch.delenv("OVAI_QE_DOMAINS", raising=False)
    assert _resolve_qe_domains("kit,ui,rendering") == {"kit", "ui", "rendering"}


def test_qe_domains_env_overrides_default(monkeypatch):
    monkeypatch.setenv("OVAI_QE_DOMAINS", "physics,sensors")
    assert _resolve_qe_domains("kit,ui,rendering") == {"physics", "sensors"}


def test_qe_domains_empty_returns_none(monkeypatch):
    monkeypatch.setenv("OVAI_QE_DOMAINS", "")
    assert _resolve_qe_domains("kit,ui") is None


def test_qe_domains_whitespace_trimmed(monkeypatch):
    monkeypatch.delenv("OVAI_QE_DOMAINS", raising=False)
    assert _resolve_qe_domains(" kit , ui ") == {"kit", "ui"}


def test_qe_domains_drops_empty_tokens(monkeypatch):
    monkeypatch.delenv("OVAI_QE_DOMAINS", raising=False)
    assert _resolve_qe_domains("kit,,ui") == {"kit", "ui"}


def test_maybe_load_hybrid_returns_none_when_semantic_mode(monkeypatch):
    monkeypatch.setenv("OVAI_RETRIEVAL_MODE", "semantic")
    assert maybe_load_hybrid("/does/not/matter", embedder=None) is None


def test_maybe_load_hybrid_returns_none_when_bundle_fails(monkeypatch, tmp_path):
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    # Point at a non-existent path — load_bundle will raise inside the try/except
    # and the helper must swallow it (caller falls back to legacy).
    assert maybe_load_hybrid(str(tmp_path / "nope"), embedder=None) is None


def _patch_ovgenai_retrieval_for_shim(captured_kwargs):
    """Patch the top-level package symbols ``maybe_load_hybrid`` imports inside its body."""
    import ovgenai_retrieval as _r

    class FakeHybrid:
        def __init__(self, bundle, **kw):
            captured_kwargs.update(kw)

    return patch.multiple(_r, load_bundle=lambda *a, **k: "FAKE_BUNDLE", HybridRetriever=FakeHybrid)


def test_maybe_load_hybrid_passes_rerank_backend_env(monkeypatch):
    """``OVAI_RERANK_BACKEND=nvidia`` must reach HybridRetriever."""
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("OVAI_RERANK", "true")
    monkeypatch.setenv("OVAI_RERANK_BACKEND", "nvidia")

    captured = {}
    with _patch_ovgenai_retrieval_for_shim(captured):
        result = maybe_load_hybrid("/whatever", embedder=None, qe_domains_default="kit,ui")

    assert result is not None
    assert captured.get("rerank") is True
    assert captured.get("rerank_backend") == "nvidia"


def test_maybe_load_hybrid_defaults_rerank_backend_to_cross_encoder(monkeypatch):
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("OVAI_RERANK", "true")
    monkeypatch.delenv("OVAI_RERANK_BACKEND", raising=False)

    captured = {}
    with _patch_ovgenai_retrieval_for_shim(captured):
        result = maybe_load_hybrid("/whatever", embedder=None, qe_domains_default="kit,ui")

    assert result is not None
    assert captured.get("rerank_backend") == "cross_encoder"


def test_maybe_load_hybrid_empty_backend_env_uses_legacy_default(monkeypatch):
    """Compose can pass an empty OVAI_RERANK_BACKEND when no host override is
    set. Empty must behave like unset so legacy KIT_RERANKER_BACKEND=nvidia_api
    still selects the hosted NVIDIA reranker."""
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("OVAI_RERANK", "true")
    monkeypatch.setenv("OVAI_RERANK_BACKEND", "")
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "nvidia_api")
    monkeypatch.setenv("OVAI_RERANK_MODEL", "nvidia/current-rerank-model")

    captured = {}
    with _patch_ovgenai_retrieval_for_shim(captured):
        result = maybe_load_hybrid("/whatever", embedder=None, qe_domains_default="kit,ui")

    assert result is not None
    assert captured.get("rerank") is True
    assert captured.get("rerank_backend") == "nvidia"
    assert captured.get("rerank_model") == "nvidia/current-rerank-model"


def test_maybe_load_hybrid_empty_model_env_uses_cross_encoder_default(monkeypatch):
    """An empty OVAI_RERANK_MODEL pass-through should not erase the default
    cross-encoder model."""
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("OVAI_RERANK", "true")
    monkeypatch.setenv("OVAI_RERANK_MODEL", "")

    captured = {}
    with _patch_ovgenai_retrieval_for_shim(captured):
        result = maybe_load_hybrid("/whatever", embedder=None, qe_domains_default="kit,ui")

    assert result is not None
    assert captured.get("rerank_backend") == "cross_encoder"
    assert captured.get("rerank_model") == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_rerank_disabled_when_no_env_vars():
    assert _rerank_enabled() is False


def test_rerank_enabled_by_explicit_ovai_rerank_true(monkeypatch):
    monkeypatch.setenv("OVAI_RERANK", "true")
    assert _rerank_enabled() is True


def test_rerank_disabled_by_explicit_ovai_rerank_false_overrides_legacy(monkeypatch):
    """Explicit OVAI_RERANK=false wins even when a legacy var is set."""
    monkeypatch.setenv("OVAI_RERANK", "false")
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "local")
    assert _rerank_enabled() is False


def test_rerank_enabled_by_legacy_kit_reranker_backend(monkeypatch):
    """Existing docker-compose.local.yaml ships KIT_RERANKER_BACKEND=local.
    Pre-!658 behavior was rerank-on; preserve that."""
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "local")
    assert _rerank_enabled() is True


def test_rerank_enabled_by_legacy_nvidia_api(monkeypatch):
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "nvidia_api")
    assert _rerank_enabled() is True


def test_rerank_disabled_by_legacy_off_value(monkeypatch):
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "off")
    assert _rerank_enabled() is False


def test_rerank_backend_default_from_legacy_local(monkeypatch):
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "local")
    assert _rerank_backend_default() == "cross_encoder"


def test_rerank_backend_default_from_legacy_nvidia_api(monkeypatch):
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "nvidia_api")
    assert _rerank_backend_default() == "nvidia"


def test_rerank_backend_default_falls_back_when_no_legacy():
    assert _rerank_backend_default() == "cross_encoder"


def test_rerank_backend_default_local_with_url_routes_to_nvidia(monkeypatch):
    """docker-compose.local.yaml ships both ``KIT_RERANKER_BACKEND=local`` and
    ``KIT_LOCAL_RERANKER_URL=<NIM URL>``. The shim must route this combo to
    the ``nvidia`` backend (the only one that can talk to a local rerank NIM
    over HTTP) — not to ``cross_encoder`` which needs sentence-transformers
    (not in the shipped Docker images)."""
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "local")
    monkeypatch.setenv("KIT_LOCAL_RERANKER_URL", "https://reranker-nim:8000/v1/ranking")
    assert _rerank_backend_default() == "nvidia"


def test_legacy_local_rerank_url_finds_first_non_empty(monkeypatch):
    monkeypatch.setenv("KIT_LOCAL_RERANKER_URL", "")
    monkeypatch.setenv("ISAACSIM_LOCAL_RERANKER_URL", "https://nim:8000/v1/ranking")
    assert _legacy_local_rerank_url() == "https://nim:8000/v1/ranking"


def test_legacy_local_rerank_url_returns_none_when_all_empty():
    assert _legacy_local_rerank_url() is None


def test_maybe_load_hybrid_legacy_var_enables_rerank(monkeypatch):
    """End-to-end: KIT_RERANKER_BACKEND=local without a URL must reach
    HybridRetriever as rerank=True with backend=cross_encoder. This is the
    regression test for the audit's P2 finding about silently-disabled
    reranking."""
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "local")

    captured = {}
    with _patch_ovgenai_retrieval_for_shim(captured):
        result = maybe_load_hybrid("/whatever", embedder=None, qe_domains_default="kit,ui")

    assert result is not None
    assert captured.get("rerank") is True
    assert captured.get("rerank_backend") == "cross_encoder"


def test_maybe_load_hybrid_local_url_rewires_to_nvidia_with_base_url(monkeypatch):
    """When the docker-compose.local.yaml shape is in effect
    (``KIT_RERANKER_BACKEND=local`` + ``KIT_LOCAL_RERANKER_URL=<NIM>``):
    HybridRetriever is built with rerank_backend=nvidia (so the
    cross_encoder import path is never touched) AND its ``_reranker`` is
    swapped post-construction for a ``NvidiaReranker(base_url=<NIM>)``."""
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "local")
    monkeypatch.setenv("KIT_LOCAL_RERANKER_URL", "https://reranker-nim:8000/v1/ranking")

    captured: dict = {}
    rewired = {"reranker": None}

    class _FakeNvidiaReranker:
        def __init__(self, model, base_url=None):
            rewired["reranker"] = {"model": model, "base_url": base_url}

    class _FakeHybrid:
        def __init__(self, bundle, **kw):
            captured.update(kw)
            self._reranker = "ORIGINAL"

    import ovgenai_retrieval as _r
    import ovgenai_retrieval.rerank as _rk

    with (
        patch.multiple(_r, load_bundle=lambda *a, **k: "FAKE", HybridRetriever=_FakeHybrid),
        patch.object(_rk, "NvidiaReranker", _FakeNvidiaReranker),
    ):
        retriever = maybe_load_hybrid("/whatever", embedder=None, qe_domains_default="kit,ui")

    assert captured.get("rerank") is True
    assert captured.get("rerank_backend") == "nvidia"
    # Post-construction swap actually happened.
    assert retriever._reranker != "ORIGINAL"
    assert rewired["reranker"]["base_url"] == "https://reranker-nim:8000/v1/ranking"
    assert rewired["reranker"]["model"] == "nvidia/llama-nemotron-rerank-vl-1b-v2"


def test_maybe_load_hybrid_local_url_respects_explicit_ovai_rerank_model(monkeypatch):
    """If the operator pinned an explicit nvidia/... rerank model via env, that
    model should be the one used when we rewire to the local NIM."""
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    monkeypatch.setenv("KIT_RERANKER_BACKEND", "local")
    monkeypatch.setenv("KIT_LOCAL_RERANKER_URL", "https://nim:8000/v1/ranking")
    monkeypatch.setenv("OVAI_RERANK_MODEL", "nvidia/some-other-rerank-model")

    rewired = {}

    class _FakeNvidiaReranker:
        def __init__(self, model, base_url=None):
            rewired["model"] = model
            rewired["base_url"] = base_url

    class _FakeHybrid:
        def __init__(self, bundle, **kw):
            self._reranker = "ORIGINAL"

    import ovgenai_retrieval as _r
    import ovgenai_retrieval.rerank as _rk

    with (
        patch.multiple(_r, load_bundle=lambda *a, **k: "FAKE", HybridRetriever=_FakeHybrid),
        patch.object(_rk, "NvidiaReranker", _FakeNvidiaReranker),
    ):
        maybe_load_hybrid("/whatever", embedder=None, qe_domains_default="kit,ui")

    assert rewired["model"] == "nvidia/some-other-rerank-model"
    assert rewired["base_url"] == "https://nim:8000/v1/ranking"


class _FakeHit:
    """Minimal stand-in for ``ovgenai_retrieval.Hit`` for shim tests. Matches
    the real ``Hit`` dataclass's field names (the bug fixed in P1c was that
    the consolidation referenced ``hit.text`` while the dataclass exposes
    ``content``)."""

    def __init__(
        self,
        content,
        score,
        metadata=None,
        provenance=None,
        index_text="",
        file_path=None,
        line_start=None,
        line_end=None,
        section_hierarchy=None,
        url=None,
    ):
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.provenance = provenance
        self.index_text = index_text
        self.file_path = file_path
        self.line_start = line_start
        self.line_end = line_end
        self.section_hierarchy = section_hierarchy or []
        self.url = url


def test_hits_to_documents_uses_hit_content_attribute():
    """Regression: the consolidated shim originally referenced ``hit.text``,
    but ``ovgenai_retrieval.Hit`` exposes the chunk text as ``content``. With
    the buggy version the hybrid search path raised ``AttributeError: 'Hit'
    object has no attribute 'text'`` on every call."""
    hits = [_FakeHit("hello", 0.9)]
    [doc] = hits_to_documents(hits)
    assert doc.page_content == "hello"


def test_hits_to_documents_promotes_first_class_fields_to_metadata():
    """Legacy per-package shims surfaced ``index_text`` / ``file_path`` etc.
    on the LangChain Document's metadata. Downstream RAG-context formatters
    read those keys (e.g. ``rag_result.metadata['index_text']``), so the
    consolidated shim must keep that contract."""
    hits = [
        _FakeHit(
            "chunk",
            0.7,
            metadata={"kit_version": "110.1"},
            index_text="Stage",
            file_path="kit/stage.py",
            line_start=12,
            line_end=34,
            section_hierarchy=["Stage", "Namespace"],
            url="https://docs/...",
        )
    ]
    [doc] = hits_to_documents(hits)
    assert doc.metadata["index_text"] == "Stage"
    assert doc.metadata["file_path"] == "kit/stage.py"
    assert doc.metadata["line_start"] == 12
    assert doc.metadata["line_end"] == 34
    assert doc.metadata["section_hierarchy"] == ["Stage", "Namespace"]
    assert doc.metadata["url"] == "https://docs/..."
    # Hit.metadata keys round-trip too.
    assert doc.metadata["kit_version"] == "110.1"


def test_hits_to_documents_does_not_overwrite_existing_metadata_keys():
    """If the Hit's ``metadata`` blob already supplies one of the first-class
    fields, the first-class value must NOT overwrite it (legacy contract)."""
    hits = [_FakeHit("c", 0.5, metadata={"index_text": "from-md"}, index_text="from-attr")]
    [doc] = hits_to_documents(hits)
    assert doc.metadata["index_text"] == "from-md"


def test_hits_to_documents_with_scores_distance_formula():
    """Distance ``1 - score`` is what the legacy ``similarity_search_with_score``
    callers expect (they recompute similarity as ``1 / (1 + distance)``)."""
    hits = [_FakeHit("a", 0.9), _FakeHit("b", 0.5), _FakeHit("c", 0.0)]
    pairs = hits_to_documents_with_scores(hits)
    assert [round(s, 4) for _, s in pairs] == [round(1.0 - 0.9, 4), 0.5, 1.0]
    # Documents preserve order and page_content.
    assert [d.page_content for d, _ in pairs] == ["a", "b", "c"]


def test_hits_to_documents_with_scores_clamps_raw_lexical_scores():
    """Raw BM25 scores can exceed 1.0 in ``OVAI_FUSION=lexical_only``. The
    legacy services convert distance back to relevance with
    ``1.0 / (1.0 + distance)``, so the shim must never return a negative
    distance."""
    [(doc, distance)] = hits_to_documents_with_scores([_FakeHit("bm25", 2.0)])
    assert doc.page_content == "bm25"
    assert distance == 0.0
    assert 1.0 / (1.0 + distance) == 1.0


def test_hits_to_documents_with_scores_metadata_preserved():
    hits = [_FakeHit("x", 0.7, metadata={"k": "v"}, provenance="bm25")]
    [(doc, _)] = hits_to_documents_with_scores(hits)
    assert doc.metadata["k"] == "v"
    assert doc.metadata["_ovai_score"] == 0.7
    assert doc.metadata["_ovai_provenance"] == "bm25"


def test_maybe_load_hybrid_threads_qe_domains_through(monkeypatch):
    monkeypatch.delenv("OVAI_RETRIEVAL_MODE", raising=False)
    monkeypatch.delenv("OVAI_QE_DOMAINS", raising=False)

    captured = {}
    with _patch_ovgenai_retrieval_for_shim(captured):
        maybe_load_hybrid("/whatever", embedder=None, qe_domains_default="physics,sensors,ai")

    assert captured.get("qe_domains") == {"physics", "sensors", "ai"}
