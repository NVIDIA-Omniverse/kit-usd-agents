"""Tests for the JSON-based BM25 sidecar format (bm25_safe)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from ovgenai_retrieval.bm25.bm25_safe import (
    FORMAT_VERSION,
    _bm25_to_safe_dict,
    convert_pkl_to_json,
    load_bm25_json,
    save_bm25_json,
)


def _ref_bm25():
    from rank_bm25 import BM25Okapi

    corpus = [
        ["hello", "world"],
        ["foo", "bar", "baz"],
        ["hello", "foo"],
        ["kit", "extension", "lifecycle"],
    ]
    return BM25Okapi(corpus, k1=1.2, b=0.75)


def test_roundtrip_scores_match(tmp_path: Path):
    ref = _ref_bm25()
    p = tmp_path / "bm25.json"
    save_bm25_json(ref, p)
    loaded = load_bm25_json(p)
    for q in [["hello"], ["foo", "bar"], ["kit", "extension"], ["nothing"]]:
        assert np.allclose(ref.get_scores(q), loaded.get_scores(q))


def test_format_version_check(tmp_path: Path):
    ref = _ref_bm25()
    p = tmp_path / "bm25.json"
    save_bm25_json(ref, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["format_version"] = 99
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="format_version"):
        load_bm25_json(p)


def test_backend_check(tmp_path: Path):
    ref = _ref_bm25()
    p = tmp_path / "bm25.json"
    save_bm25_json(ref, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["backend"] = "tantivy"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="backend"):
        load_bm25_json(p)


def test_safe_dict_is_plain_json(tmp_path: Path):
    d = _bm25_to_safe_dict(_ref_bm25())
    # Must serialize without custom encoder.
    s = json.dumps(d)
    reparsed = json.loads(s)
    assert reparsed["format_version"] == FORMAT_VERSION
    assert reparsed["backend"] == "rank_bm25"
    assert reparsed["corpus_size"] == 4
    assert isinstance(reparsed["doc_freqs"], list)
    assert isinstance(reparsed["doc_freqs"][0], dict)


def test_convert_pkl_to_json_round_trip(tmp_path: Path):
    ref = _ref_bm25()
    pkl = tmp_path / "bm25.pkl"
    # Reproduce the old payload shape produced by the pre-G-7 upgrade.
    with open(pkl, "wb") as f:
        pickle.dump({"bm25": ref, "corpus_len": 4}, f)
    json_path = tmp_path / "bm25.json"
    assert convert_pkl_to_json(pkl, json_path, delete_pkl=True) is True
    assert json_path.exists()
    assert not pkl.exists()
    loaded = load_bm25_json(json_path)
    assert np.allclose(ref.get_scores(["hello"]), loaded.get_scores(["hello"]))


def test_convert_pkl_to_json_idempotent(tmp_path: Path):
    ref = _ref_bm25()
    pkl = tmp_path / "bm25.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"bm25": ref, "corpus_len": 4}, f)
    json_path = tmp_path / "bm25.json"
    save_bm25_json(ref, json_path)
    # Second conversion is a no-op once JSON exists.
    assert convert_pkl_to_json(pkl, json_path) is False
