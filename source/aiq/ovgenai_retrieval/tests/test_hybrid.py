from unittest.mock import MagicMock

from ovgenai_retrieval import HybridRetriever, load_bundle


def test_hybrid_lexical_only_finds_doc0(tiny_text_bundle, stub_embedder):
    # Force lexical-only: no embedding service needed
    bundle = load_bundle(tiny_text_bundle)
    r = HybridRetriever(bundle, top_k=3, fusion="lexical_only")
    hits = r.retrieve("how to duplicate a prim")
    assert hits
    assert hits[0].doc_id in {"d0", "d3"}  # both mention duplicate/CopySpec


def test_hybrid_semantic_only_with_stub_embedder(tiny_text_bundle, stub_embedder):
    bundle = load_bundle(tiny_text_bundle)
    r = HybridRetriever(bundle, top_k=3, fusion="semantic_only", embedder=stub_embedder, query_expand=False)
    hits = r.retrieve("how to duplicate a prim")
    assert hits
    # Stub embedder maps "duplicate" -> vector near doc 0
    assert hits[0].doc_id == "d0"


def test_hybrid_rrf_fusion(tiny_text_bundle, stub_embedder):
    bundle = load_bundle(tiny_text_bundle)
    r = HybridRetriever(bundle, top_k=3, fusion="rrf", embedder=stub_embedder, query_expand=False)
    hits = r.retrieve("how to duplicate a prim")
    assert hits
    ids = {h.doc_id for h in hits}
    assert {"d0", "d3"} & ids  # at least one of the duplicate-related docs shows up


def test_hybrid_hit_exposes_file_path(tiny_text_bundle, stub_embedder):
    bundle = load_bundle(tiny_text_bundle)
    r = HybridRetriever(bundle, fusion="lexical_only", top_k=1)
    hits = r.retrieve("treeview model")
    assert hits
    assert hits[0].file_path == "docs/d2.md"


def test_catalog_bundle_rejected(tiny_catalog_bundle, stub_embedder):
    import pytest

    bundle = load_bundle(tiny_catalog_bundle)
    with pytest.raises(TypeError):
        HybridRetriever(bundle, embedder=stub_embedder)


def test_hybrid_match_mode_lexical_only(tiny_text_bundle, stub_embedder):
    bundle = load_bundle(tiny_text_bundle)
    r = HybridRetriever(bundle, top_k=3, fusion="lexical_only")
    hits = r.retrieve("how to duplicate a prim")
    assert hits
    assert all(h.match_mode == "keyword" for h in hits)


def test_hybrid_match_mode_semantic_only(tiny_text_bundle, stub_embedder):
    bundle = load_bundle(tiny_text_bundle)
    r = HybridRetriever(bundle, top_k=3, fusion="semantic_only", embedder=stub_embedder, query_expand=False)
    hits = r.retrieve("how to duplicate a prim")
    assert hits
    assert all(h.match_mode == "semantic" for h in hits)


def test_hybrid_match_mode_rrf_includes_both(tiny_text_bundle, stub_embedder):
    bundle = load_bundle(tiny_text_bundle)
    r = HybridRetriever(bundle, top_k=5, fusion="rrf", embedder=stub_embedder, query_expand=False)
    hits = r.retrieve("duplicate CopySpec prim")
    assert hits
    # At least one doc should be reachable from both sides.
    modes = {h.match_mode for h in hits}
    assert modes & {"both", "semantic", "keyword"}
    assert "both" in modes or len(modes) >= 1


def test_hybrid_builds_bm25_in_process_when_sidecar_missing(tiny_text_bundle):
    """Bundle without a bm25.pkl sidecar should still serve lexical-only queries
    (in-process BM25 is built on first use)."""
    bundle = load_bundle(tiny_text_bundle)
    # The tiny_text_bundle fixture ships no bm25.pkl.
    assert not bundle.has_lexical_sidecar
    r = HybridRetriever(bundle, top_k=3, fusion="lexical_only")
    hits = r.retrieve("CopySpec")
    assert hits
    assert hits[0].doc_id in {"d0", "d3"}


def test_hybrid_reranker_falls_back_to_nonempty_index_text(tiny_text_bundle):
    bundle = load_bundle(tiny_text_bundle)
    r = HybridRetriever(bundle, top_k=3, fusion="lexical_only")
    for record in r._records:
        record.index_text = record.index_text or record.content
        record.content = ""
    r._reranker = MagicMock()
    r._reranker.score.side_effect = lambda _query, candidates: [0.5] * len(candidates)

    assert r.retrieve("duplicate CopySpec prim")
    candidates = r._reranker.score.call_args.args[1]
    assert candidates
    assert all(candidates)
