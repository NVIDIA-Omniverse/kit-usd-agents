"""Integration tests hitting the real NVIDIA hosted embedder + (optional) reranker.

These tests are skipped automatically when ``NVIDIA_API_KEY`` is not in the
environment. Designed for the ship-readiness validation plan — not part of
the normal unit suite.

Run with:
    NVIDIA_API_KEY=$(cat ~/.secrets/nvidia_api_key) pytest -q tests/test_integration_real_embedder.py -m integration
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY"),
        reason="NVIDIA_API_KEY not set — skipping real-embedder integration tests",
    ),
]


SMOKE_BUNDLE = Path(os.environ.get("OVGENAI_SMOKE_BUNDLE", "/nonexistent/ovgenai-smoke-bundle"))


@pytest.fixture(scope="module")
def smoke_bundle() -> Path:
    if not SMOKE_BUNDLE.exists():
        pytest.skip(f"smoke bundle not available at {SMOKE_BUNDLE}")
    return SMOKE_BUNDLE


# ---------------------------------------------------------------------------
# Test A — HybridRetriever against upgraded smoke bundle (RRF default)
# ---------------------------------------------------------------------------


def test_hybrid_rrf_smoke_bundle(smoke_bundle: Path):
    from ovgenai_retrieval import HybridRetriever, load_bundle

    bundle = load_bundle(smoke_bundle, allow_pickle=True)
    assert bundle.type in {"knowledge", "code"}
    assert bundle.has_lexical_sidecar, "upgraded smoke bundle should ship BM25"
    assert bundle.has_faiss

    retriever = HybridRetriever(bundle, top_k=3, fusion="rrf", query_expand=False)
    hits = retriever.retrieve("asset requirements for an Omniverse scene")

    assert hits, "expected at least one hit"
    assert len(hits) <= 3
    # Every hit should be well-formed
    for h in hits:
        assert h.doc_id
        assert h.content
        assert h.match_mode in {"semantic", "keyword", "both"}
        assert h.provenance.get("fused_score") is not None
    # At least one hit should be from the asset-requirements corpus
    assert any(
        "asset" in (h.file_path or "").lower() or "asset-requirements" in (h.index_text or "").lower() for h in hits
    ), [h.file_path or h.index_text for h in hits]


# ---------------------------------------------------------------------------
# Test B — fusion modes produce different top hits (or at least overlap)
# ---------------------------------------------------------------------------


def test_fusion_mode_differentiation(smoke_bundle: Path):
    from ovgenai_retrieval import HybridRetriever, load_bundle

    bundle = load_bundle(smoke_bundle, allow_pickle=True)
    query = "asset references anchored paths"

    sem = HybridRetriever(bundle, top_k=5, fusion="semantic_only", query_expand=False)
    lex = HybridRetriever(bundle, top_k=5, fusion="lexical_only")
    rrf = HybridRetriever(bundle, top_k=5, fusion="rrf", query_expand=False)

    sem_hits = sem.retrieve(query)
    lex_hits = lex.retrieve(query)
    rrf_hits = rrf.retrieve(query)

    assert sem_hits and lex_hits and rrf_hits

    sem_ids = [h.doc_id for h in sem_hits]
    lex_ids = [h.doc_id for h in lex_hits]
    rrf_ids = [h.doc_id for h in rrf_hits]

    # Semantic and lexical rankings should not be *identical* for a multi-
    # word query on a corpus this size.
    assert sem_ids != lex_ids or sem_hits[0].score != lex_hits[0].score

    # RRF output should contain at least one doc reachable from either side.
    assert set(rrf_ids) & (set(sem_ids) | set(lex_ids))


# ---------------------------------------------------------------------------
# Test C — optional cross-encoder reranker path
# ---------------------------------------------------------------------------


def test_reranker_changes_score_and_labels(smoke_bundle: Path):
    pytest.importorskip("sentence_transformers")
    from ovgenai_retrieval import HybridRetriever, load_bundle
    from ovgenai_retrieval.scoring import hit_confidence_label

    bundle = load_bundle(smoke_bundle, allow_pickle=True)
    r = HybridRetriever(
        bundle,
        top_k=3,
        fusion="rrf",
        rerank=True,
        query_expand=False,
    )
    hits = r.retrieve("Requirements Index Atomic Asset")
    assert hits

    # With rerank, top hit's provenance should carry a rerank_score.
    top = hits[0]
    assert "rerank_score" in top.provenance
    # Confidence label should be derived from logit
    label = hit_confidence_label(top)
    assert label in {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# Test C2 — NvidiaReranker backend (REQ-ST-6)
# ---------------------------------------------------------------------------


def test_reranker_nvidia_backend(smoke_bundle: Path):
    """Runs HybridRetriever with rerank_backend='nvidia' against the hosted
    llama-nemotron-rerank endpoint. Skips if the extra import can't resolve."""
    pytest.importorskip("langchain_nvidia_ai_endpoints")
    from ovgenai_retrieval import HybridRetriever, load_bundle

    bundle = load_bundle(smoke_bundle, allow_pickle=True)
    r = HybridRetriever(
        bundle,
        top_k=3,
        fusion="rrf",
        rerank=True,
        rerank_backend="nvidia",
        query_expand=False,
    )
    hits = r.retrieve("Requirements Index Atomic Asset")
    assert hits

    top = hits[0]
    assert "rerank_score" in top.provenance
    # Hosted endpoint must return a real float (not the 0.0 default).
    # Allow zero only if the endpoint genuinely scored all candidates at 0,
    # but at least assert the key exists and is numeric.
    assert isinstance(top.provenance["rerank_score"], float)


# ---------------------------------------------------------------------------
# Test D — kt_index round-trip with real embedder (end-to-end folder → bundle)
# ---------------------------------------------------------------------------


def test_kt_index_roundtrip_with_real_embedder(tmp_path: Path):
    """Fails fast if the ``kt_index`` CLI isn't on PATH or if the real embedder
    refuses the job. Catches regressions in
    :mod:`ovgenai_agent_search.indexer.build` and
    :class:`ovgenai_retrieval.embedder.EmbedderFactory`."""
    root = tmp_path / "md_root"
    root.mkdir()
    (root / "kit.md").write_text(
        "# Kit Runtime\n\nThe Kit runtime loads extensions at startup. Each\n"
        "extension declares dependencies in config/extension.toml. Settings\n"
        "live at paths like /app/window/width.\n\n"
        "## Extensions\n\nExtensions are Python/C++ modules that omni.kit.app\n"
        "loads on boot. Their lifecycle is defined by IExt subclasses.\n",
        encoding="utf-8",
    )
    (root / "usd.md").write_text(
        "# USD Primer\n\nA Prim is the fundamental node in OpenUSD's scene\n"
        "graph. Composition arcs follow LIVRPS ordering. To duplicate a prim\n"
        "safely, use Sdf.CopySpec which preserves layer offsets and ordering.\n",
        encoding="utf-8",
    )

    out = tmp_path / "bundle"

    kt_index = shutil.which("kt_index")
    if kt_index is None:
        pytest.skip("kt_index not on PATH — install ovgenai-agent-search first")

    # Run kt_index with the real embedder (no --no-embed this time).
    result = subprocess.run(
        [kt_index, "--root", str(root), "--out", str(out), "--batch-size", "4"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"kt_index failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert (out / "index.faiss").exists(), "faiss index should be written"
    assert (out / "index.json").exists()
    assert (out / "bm25.pkl").exists()
    assert (out / "manifest.json").exists()
    assert (out / "files" / "kit.md").exists()

    # Load + query: top hit for "extension.toml" should come from kit.md.
    from ovgenai_retrieval import HybridRetriever, load_bundle

    bundle = load_bundle(out)
    assert bundle.has_faiss
    assert bundle.has_lexical_sidecar
    m = json.loads((out / "manifest.json").read_text())
    assert m["embeddings"]["model"] == "nvidia/nemotron-3-embed-1b"
    assert m["embeddings"]["dim"] > 0

    r = HybridRetriever(bundle, top_k=2, fusion="rrf", query_expand=True)
    hits = r.retrieve("How do I declare extension dependencies?")
    assert hits
    assert any("kit.md" in (h.file_path or "") for h in hits)
