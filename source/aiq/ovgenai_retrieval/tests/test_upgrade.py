"""In-place upgrade of legacy bundles: BM25 sidecar + manifest.json."""

from __future__ import annotations

import json
from pathlib import Path

from ovgenai_retrieval import HybridRetriever, load_bundle
from ovgenai_retrieval.upgrade import (
    build_bm25_sidecar_from_bundle,
    discover_bundles,
    upgrade_bundle,
    write_or_update_manifest,
)


def test_build_bm25_sidecar_from_legacy_bundle(tiny_text_bundle: Path):
    # tiny_text_bundle has no manifest and no bm25 sidecar.
    assert not (tiny_text_bundle / "bm25.json").exists()
    assert not (tiny_text_bundle / "bm25.pkl").exists()
    assert not (tiny_text_bundle / "manifest.json").exists()

    result = build_bm25_sidecar_from_bundle(tiny_text_bundle)
    assert result["status"] == "created"
    assert result["corpus_len"] == 5  # fixture has 5 docs
    assert (tiny_text_bundle / "bm25.json").exists()
    assert not (tiny_text_bundle / "bm25.pkl").exists()  # pickle banned

    # Second call without --force is a no-op
    again = build_bm25_sidecar_from_bundle(tiny_text_bundle)
    assert again["status"] == "exists"

    # With --force, it overwrites
    forced = build_bm25_sidecar_from_bundle(tiny_text_bundle, force=True)
    assert forced["status"] == "created"


def test_build_bm25_sidecar_migrates_legacy_pkl(tiny_text_bundle: Path):
    """A pre-existing bm25.pkl (legacy bundle) is migrated to bm25.json
    in place and the pickle is deleted."""
    # Seed a legacy pkl by dropping one in via the safe module helper.
    import pickle as _pickle

    from rank_bm25 import BM25Okapi

    seed = BM25Okapi([["hello", "world"], ["foo"]], k1=1.2, b=0.75)
    payload = {"bm25": seed, "corpus_len": 2}
    pkl_path = tiny_text_bundle / "bm25.pkl"
    with open(pkl_path, "wb") as f:
        _pickle.dump(payload, f)

    result = build_bm25_sidecar_from_bundle(tiny_text_bundle)
    assert result["status"] == "migrated"
    assert (tiny_text_bundle / "bm25.json").exists()
    assert not (tiny_text_bundle / "bm25.pkl").exists()


def test_build_bm25_sidecar_rejects_pkl_filename(tiny_text_bundle: Path):
    result = build_bm25_sidecar_from_bundle(tiny_text_bundle, filename="bm25.pkl")
    assert result["status"] == "skipped"
    assert "banned" in (result.get("reason") or "").lower()


def test_write_manifest_includes_bm25_sidecar(tiny_text_bundle: Path):
    build_bm25_sidecar_from_bundle(tiny_text_bundle)
    result = write_or_update_manifest(tiny_text_bundle)
    assert result["status"] == "created"

    m = json.loads((tiny_text_bundle / "manifest.json").read_text())
    assert m["schema_version"] == "1"
    assert m["bundle_type"] == "knowledge"
    assert m["lexical"]["backend"] == "rank_bm25"
    assert m["lexical"]["file"] == "bm25.json"
    assert m["embeddings"]["model"] == "unknown"
    assert m["embeddings"]["dim"] == 4
    assert m["chunks"]["count"] == 5


def test_manifest_without_bm25_omits_lexical(tiny_text_bundle: Path):
    # Don't build bm25 first.
    result = write_or_update_manifest(tiny_text_bundle)
    assert result["status"] == "created"
    m = json.loads((tiny_text_bundle / "manifest.json").read_text())
    assert "lexical" not in m or m["lexical"] is None


def test_upgrade_bundle_round_trip(tiny_text_bundle: Path, stub_embedder):
    # Upgrade — emits both sidecar and manifest.
    r = upgrade_bundle(tiny_text_bundle)
    assert r["bm25"]["status"] == "created"
    assert r["manifest"]["status"] == "created"

    # After upgrade, loader picks up the sidecar (has_lexical_sidecar is True).
    bundle = load_bundle(tiny_text_bundle)
    assert bundle.has_lexical_sidecar is True
    assert bundle.manifest.lexical is not None
    assert bundle.manifest.lexical.backend == "rank_bm25"

    # And HybridRetriever can use the preloaded sidecar.
    r2 = HybridRetriever(bundle, fusion="lexical_only", top_k=2, embedder=stub_embedder)
    hits = r2.retrieve("duplicate USD prim")
    assert hits


def test_upgrade_bundle_idempotent(tiny_text_bundle: Path):
    upgrade_bundle(tiny_text_bundle)
    r2 = upgrade_bundle(tiny_text_bundle)
    assert r2["bm25"]["status"] == "exists"
    assert r2["manifest"]["status"] == "exists"


def test_discover_bundles(tmp_path: Path, tiny_text_bundle: Path):
    # Move tiny_text_bundle under tmp_path to test recursive discovery
    import shutil

    root = tmp_path / "root"
    root.mkdir()
    dest_a = root / "sub" / "a"
    dest_b = root / "sub" / "b"
    shutil.copytree(tiny_text_bundle, dest_a)
    shutil.copytree(tiny_text_bundle, dest_b)

    found = discover_bundles(root)
    assert len(found) == 2
    assert {p.name for p in found} == {"a", "b"}


def test_upgrade_skips_catalog_bundle(tiny_catalog_bundle: Path):
    # Catalogs don't need BM25.
    r = build_bm25_sidecar_from_bundle(tiny_catalog_bundle)
    assert r["status"] == "skipped"
    assert "catalog" in r["reason"].lower()
