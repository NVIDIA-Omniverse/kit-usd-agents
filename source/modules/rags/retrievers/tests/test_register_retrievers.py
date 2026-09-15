import json
import sys
import types

import faiss
import numpy as np
import pytest

lc_agent_stub = types.ModuleType("lc_agent")
lc_agent_stub.get_retriever_registry = lambda: None
sys.modules.setdefault("lc_agent", lc_agent_stub)

from lc_agent_retrievers.register_retrievers import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    _load_faiss_index,
)


class _Embedder:
    def embed_query(self, _text):
        return [1.0, 0.0, 0.0]


def _write_bundle(path, model="test/embedder"):
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
        json.dumps({"embeddings": {"model": model, "dim": 3}}),
        encoding="utf-8",
    )


def test_load_faiss_index_validates_manifest(tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    loaded = _load_faiss_index(bundle, _Embedder(), "test/embedder")

    assert loaded.index.ntotal == 1


def test_load_faiss_index_rejects_model_mismatch(tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        _load_faiss_index(bundle, _Embedder(), "different/embedder")


def test_load_faiss_index_rejects_stale_default_dimension(tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, model=DEFAULT_EMBEDDING_MODEL)

    with pytest.raises(ValueError, match="requires 2048"):
        _load_faiss_index(bundle, _Embedder(), DEFAULT_EMBEDDING_MODEL)
