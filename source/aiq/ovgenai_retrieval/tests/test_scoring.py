import pytest
from ovgenai_retrieval.hits import Hit
from ovgenai_retrieval.scoring import (
    HIGH_FUSED_SCORE,
    HIGH_RERANK_LOGIT,
    MED_FUSED_SCORE,
    MED_RERANK_LOGIT,
    bm25_confidence,
    combined_confidence,
    confidence_label,
    hit_confidence_label,
    semantic_confidence,
)


def test_semantic_confidence_cosine_range():
    assert semantic_confidence(1.0) == 1.0
    assert semantic_confidence(-1.0) == 0.0
    assert 0.4 < semantic_confidence(0.0) < 0.6


def test_bm25_confidence_monotonic():
    assert bm25_confidence(0) < bm25_confidence(5) < bm25_confidence(50)


def test_combined_confidence_averaging():
    assert combined_confidence(0.8, 0.4) == pytest.approx(0.6)
    assert combined_confidence(0.8, None) == pytest.approx(0.8)
    assert combined_confidence(None, None) == 0.0


def test_confidence_label_fused_buckets():
    assert confidence_label(HIGH_FUSED_SCORE + 0.01) == "high"
    assert confidence_label(HIGH_FUSED_SCORE) == "high"
    assert confidence_label(MED_FUSED_SCORE) == "medium"
    assert confidence_label(MED_FUSED_SCORE - 0.01) == "low"
    assert confidence_label(0.0) == "low"


def test_confidence_label_logit_buckets():
    assert confidence_label(HIGH_RERANK_LOGIT + 0.01, use_logit=True) == "high"
    assert confidence_label(HIGH_RERANK_LOGIT, use_logit=True) == "high"
    assert confidence_label(MED_RERANK_LOGIT, use_logit=True) == "medium"
    assert confidence_label(MED_RERANK_LOGIT - 0.01, use_logit=True) == "low"
    assert confidence_label(-1.0, use_logit=True) == "low"


def test_confidence_label_threshold_override():
    assert confidence_label(0.3, high=0.25, medium=0.1) == "high"
    assert confidence_label(0.15, high=0.25, medium=0.1) == "medium"


def test_hit_confidence_label_prefers_rerank_logit():
    h = Hit(doc_id="a", content="x", score=0.01, provenance={"rerank_score": 6.0})
    assert hit_confidence_label(h) == "high"


def test_hit_confidence_label_falls_back_to_fused():
    h = Hit(doc_id="a", content="x", score=0.6, provenance={})
    assert hit_confidence_label(h) == "high"
    h2 = Hit(doc_id="b", content="x", score=0.05, provenance={})
    assert hit_confidence_label(h2) == "low"
