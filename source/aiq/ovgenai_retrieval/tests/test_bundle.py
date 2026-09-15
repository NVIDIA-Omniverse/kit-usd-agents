import json

import pytest
from ovgenai_retrieval import load_bundle


def test_legacy_json_bundle_synthesized_manifest(tiny_text_bundle):
    bundle = load_bundle(tiny_text_bundle)
    assert bundle.type == "knowledge"
    assert bundle.has_faiss is True
    # docstore loaded with 5 records
    assert bundle.docstore is not None
    assert len(bundle.docstore.records) == 5
    # id_to_faiss_index maps every record
    assert set(bundle.docstore.id_to_faiss_index.values()) == {0, 1, 2, 3, 4}


def test_iter_records_ordered_by_faiss_row(tiny_text_bundle):
    bundle = load_bundle(tiny_text_bundle)
    recs = bundle.iter_records()
    ids = [r.doc_id for r in recs]
    assert ids == [f"d{i}" for i in range(5)]


def test_catalog_bundle(tiny_catalog_bundle):
    bundle = load_bundle(tiny_catalog_bundle)
    assert bundle.type == "catalog"
    assert bundle.has_faiss is False
    assert bundle.docstore is None
    assert isinstance(bundle.catalog_data, list)
    assert len(bundle.catalog_data) == 5


def test_json_bundle_rejects_non_contiguous_row_map(tiny_text_bundle):
    path = tiny_text_bundle / "index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["index_to_docstore_id"]["7"] = payload["index_to_docstore_id"].pop("4")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contiguous"):
        load_bundle(tiny_text_bundle)


def test_json_bundle_rejects_unmapped_document(tiny_text_bundle):
    path = tiny_text_bundle / "index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["index_to_docstore_id"].pop("4")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="row map and docstore IDs differ"):
        load_bundle(tiny_text_bundle)
