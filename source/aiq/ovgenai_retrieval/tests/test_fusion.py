import pytest
from ovgenai_retrieval.fusion import rrf_fuse


def test_rrf_basic():
    sem = [7, 2, 5]  # semantic ranks 7→1st, 2→2nd, 5→3rd
    lex = [5, 7, 9]  # lexical    5→1st, 7→2nd, 9→3rd
    fused = rrf_fuse([sem, lex], k=60)
    ids = [i for i, _ in fused]
    # 7 and 5 are top in both lists; 5 (positions 3,1) ties or edges out 7 (1,2)
    assert ids[0] in {5, 7}
    # 9 appears only in lexical, rank 3 -> last
    assert 9 in ids


def test_rrf_empty_lists():
    assert rrf_fuse([[], []], k=60) == []


def test_rrf_scores_descend():
    fused = rrf_fuse([[1, 2, 3], [3, 2, 1]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Weighted RRF (REQ-HS-3)
# ---------------------------------------------------------------------------


def test_rrf_equal_weights_matches_unweighted():
    """weights=[1.0, 1.0] must return identical (id, score) pairs to no weights."""
    sem = [10, 20, 30]
    lex = [30, 10, 40]
    baseline = rrf_fuse([sem, lex], k=60)
    weighted = rrf_fuse([sem, lex], k=60, weights=[1.0, 1.0])
    assert baseline == weighted


def test_rrf_asymmetric_weights_biases_toward_heavier_list():
    """With weights=[0.9, 0.1], the #1 of list-0 must outrank the #1 of list-1
    when the two top docs don't overlap."""
    list_a = [100, 101, 102]  # doc 100 is list-0's top
    list_b = [200, 201, 202]  # doc 200 is list-1's top

    fused = rrf_fuse([list_a, list_b], k=60, weights=[0.9, 0.1])
    ids = [i for i, _ in fused]
    # The heavier list's top must win.
    assert ids[0] == 100
    # The lighter list's top (200) should place below 100 and below 101
    # (since 101 carries more weight than 200's single 0.1 contribution).
    assert ids.index(100) < ids.index(200)
    assert ids.index(101) < ids.index(200)


def test_rrf_weights_length_mismatch_raises():
    with pytest.raises(ValueError, match="weights length"):
        rrf_fuse([[1, 2], [3, 4]], k=60, weights=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="weights length"):
        rrf_fuse([[1, 2], [3, 4]], k=60, weights=[1.0])
