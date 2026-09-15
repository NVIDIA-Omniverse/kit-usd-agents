from ovgenai_retrieval.bm25 import build_backend
from ovgenai_retrieval.bm25.base import default_tokenize


def test_rank_bm25_build_and_search():
    corpus = [
        default_tokenize("duplicate a USD prim using Sdf CopySpec"),
        default_tokenize("create a Kit extension with repo sh template"),
        default_tokenize("build a TreeView with AbstractItemModel"),
        default_tokenize("Hydra delegate MaterialBindingAPI Apply"),
    ]
    idx = build_backend("rank_bm25", corpus)
    # Query that should rank doc 0 first
    top = idx.top_k_text("how to duplicate a prim", k=2)
    assert top
    assert top[0][0] == 0


def test_rank_bm25_save_load_roundtrip(tmp_path):
    corpus = [default_tokenize("alpha beta"), default_tokenize("gamma delta")]
    idx = build_backend("rank_bm25", corpus)
    p = tmp_path / "bm25.json"
    idx.save(p)
    loaded = type(idx).load(p)
    t_orig = idx.top_k(["alpha"], k=2)
    t_load = loaded.top_k(["alpha"], k=2)
    assert t_orig == t_load


def test_rank_bm25_refuses_pkl_save(tmp_path):
    import pytest

    corpus = [default_tokenize("alpha beta")]
    idx = build_backend("rank_bm25", corpus)
    with pytest.raises(ValueError, match="pickle BM25 sidecars are banned"):
        idx.save(tmp_path / "bm25.pkl")


def test_rank_bm25_refuses_pkl_load(tmp_path):
    import pytest

    (tmp_path / "bm25.pkl").write_bytes(b"\x80\x04")  # dummy pickle header
    with pytest.raises(FileNotFoundError, match="Legacy pickle"):
        from ovgenai_retrieval.bm25.rank_bm25_backend import RankBm25Index

        RankBm25Index.load(tmp_path / "bm25.pkl")


def test_rank_bm25_empty_query():
    idx = build_backend("rank_bm25", [default_tokenize("alpha")])
    assert idx.top_k([], k=5) == []


def test_tantivy_and_whoosh_stubs_raise():
    import pytest

    with pytest.raises(NotImplementedError):
        build_backend("tantivy", [["hello"]])
    with pytest.raises(NotImplementedError):
        build_backend("whoosh", [["hello"]])
