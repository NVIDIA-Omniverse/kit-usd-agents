from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from ovgenai_retrieval.reindex import EmbeddingCache, discover_bundles, reindex_bundle


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            raw = np.asarray([len(text) + 1, text.count("USD") + 1, 1.0], dtype=np.float32)
            vectors.append((raw / np.linalg.norm(raw)).tolist())
        return vectors


def test_reindex_bundle_rebuilds_vectors_and_manifest(tiny_text_bundle: Path, tmp_path: Path):
    payload = json.loads((tiny_text_bundle / "index.json").read_text(encoding="utf-8"))
    documents = payload["documents"]
    ids = list(documents)
    documents[ids[1]]["metadata"]["index_text"] = documents[ids[0]]["metadata"]["index_text"]
    (tiny_text_bundle / "index.json").write_text(json.dumps(payload), encoding="utf-8")
    (tiny_text_bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "bundle_id": "tiny",
                "copied_from": "fixture",
                "chunks": {"count": 4},
            }
        ),
        encoding="utf-8",
    )

    embedder = _FakeEmbedder()
    with EmbeddingCache(tmp_path / "vectors.sqlite3") as cache:
        result = reindex_bundle(
            tiny_text_bundle,
            embedder,
            model="nvidia/test-embedder",
            dimension=3,
            batch_size=2,
            workers=1,
            cache=cache,
        )

    index = faiss.read_index(str(tiny_text_bundle / "index.faiss"))
    manifest = json.loads((tiny_text_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert result["rows"] == 5
    assert index.d == 3
    assert index.ntotal == 5
    assert sum(len(call) for call in embedder.calls) == 4
    assert manifest["copied_from"] == "fixture"
    assert manifest["embeddings"]["model"] == "nvidia/test-embedder"
    assert manifest["embeddings"]["dim"] == 3
    assert manifest["chunks"]["count"] == 5
    assert manifest["chunks"]["index_key"] == "index_text"
    assert manifest["lexical"]["file"] == "bm25.json"


def test_discover_bundles_deduplicates_nested_roots(tiny_text_bundle: Path):
    assert discover_bundles([tiny_text_bundle, tiny_text_bundle.parent]) == [tiny_text_bundle.resolve()]
