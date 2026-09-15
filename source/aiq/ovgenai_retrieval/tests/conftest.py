"""Shared fixtures — construct tiny bundle dirs on the fly so tests don't need a real FAISS index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Tiny text bundle — manually authored embeddings (5 unit-norm 4-dim vectors)
# paired with a matching JSON docstore. Enough to exercise the full pipeline.
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


@pytest.fixture
def tiny_text_bundle(tmp_path: Path) -> Path:
    """A bundle_type='knowledge' bundle with 5 docs, no manifest (legacy JSON)."""
    import faiss

    bundle = tmp_path / "tiny_text_bundle"
    bundle.mkdir()

    # 4-dim "semantic space" — each doc parked near a distinct axis
    vecs = np.asarray(
        [
            _unit(np.asarray([1, 0, 0, 0], dtype=np.float32)),
            _unit(np.asarray([0, 1, 0, 0], dtype=np.float32)),
            _unit(np.asarray([0, 0, 1, 0], dtype=np.float32)),
            _unit(np.asarray([0.9, 0.1, 0, 0], dtype=np.float32)),  # near doc 0
            _unit(np.asarray([0, 0, 0, 1], dtype=np.float32)),
        ],
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(4)
    index.add(vecs)
    faiss.write_index(index, str(bundle / "index.faiss"))

    docs = [
        ("d0", "duplicate a USD prim", "Use Sdf.CopySpec to duplicate a USD prim."),
        ("d1", "create a Kit extension", "Scaffold a Kit extension with repo.sh template new."),
        ("d2", "omni.ui TreeView", "Build TreeView with AbstractItemModel + AbstractItemDelegate."),
        ("d3", "copy USD prim safely", "Sdf.CopySpec preserves composition arcs during duplication."),
        ("d4", "Hydra rendering delegate", "Hydra 2 requires explicit MaterialBindingAPI.Apply()."),
    ]
    documents = {}
    index_to_docstore = {}
    for i, (doc_id, idx_text, content) in enumerate(docs):
        documents[doc_id] = {
            "page_content": content,
            "metadata": {"index_text": idx_text, "file_path": f"docs/{doc_id}.md"},
        }
        index_to_docstore[str(i)] = doc_id
    (bundle / "index.json").write_text(
        json.dumps(
            {"documents": documents, "index_to_docstore_id": index_to_docstore},
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle


@pytest.fixture
def tiny_catalog_bundle(tmp_path: Path) -> Path:
    """A bundle_type='catalog' bundle with a name-keyed JSON."""
    bundle = tmp_path / "tiny_catalog_bundle"
    bundle.mkdir()

    manifest = {
        "schema_version": "1",
        "bundle_id": "tiny_cat",
        "bundle_type": "catalog",
        "catalog": {"file": "entries.json", "key_field": "name"},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    catalog = [
        {"name": "omni.ui", "description": "UI widgets", "tags": ["ui"]},
        {"name": "omni.ui.scene", "description": "Scene UI", "tags": ["ui", "scene"]},
        {"name": "omni.usd", "description": "USD bindings", "tags": ["usd"]},
        {"name": "omni.usd.core", "description": "Core USD", "tags": ["usd"]},
        {"name": "carb.audio", "description": "Audio", "tags": ["deprecated"]},
    ]
    (bundle / "entries.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return bundle


class _StubEmbedder:
    """Tiny stub that maps known queries to known unit vectors in the 4-dim tiny space."""

    # Query text -> target doc index it should top-1 match
    _mapping = {
        "how to duplicate": np.asarray([0.95, 0.05, 0, 0], dtype=np.float32),
        "create an extension": np.asarray([0, 1, 0, 0], dtype=np.float32),
        "treeview model": np.asarray([0, 0, 1, 0], dtype=np.float32),
        "hydra delegate": np.asarray([0, 0, 0, 1], dtype=np.float32),
    }

    def embed_query(self, text: str) -> list[float]:
        key = next((k for k in self._mapping if k in text.lower()), None)
        if key is None:
            v = np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        else:
            v = self._mapping[key]
        v = v / np.linalg.norm(v)
        return v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [self.embed_query(t) for t in texts]


@pytest.fixture
def stub_embedder():
    return _StubEmbedder()
