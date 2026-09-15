import json

import faiss
import numpy as np
import pytest
from kit_fns.utils.faiss_safe import load_faiss_safe


class _Embedder:
    model = "test/embedder"

    def embed_query(self, _text):
        return [1.0, 0.0, 0.0]


def _write_bundle(path):
    path.mkdir()
    index = faiss.IndexFlatL2(3)
    index.add(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    faiss.write_index(index, str(path / "index.faiss"))
    (path / "index.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "docstore": {"doc": {"page_content": "USD prim", "metadata": {}}},
                "index_to_docstore_id": {"0": "doc"},
            }
        ),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "embeddings": {"model": "test/embedder", "dim": 3},
                "chunks": {"count": 1},
            }
        ),
        encoding="utf-8",
    )


def test_load_faiss_safe_validates_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    loaded = load_faiss_safe(bundle, _Embedder())

    assert loaded.index.ntotal == 1


def test_load_faiss_safe_rejects_model_mismatch(tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        load_faiss_safe(bundle, _Embedder(), expected_model="different/embedder")


def test_load_faiss_safe_rejects_row_count_mismatch(tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    payload = json.loads((bundle / "index.json").read_text(encoding="utf-8"))
    payload["index_to_docstore_id"] = {}
    (bundle / "index.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="row mapping"):
        load_faiss_safe(bundle, _Embedder())
