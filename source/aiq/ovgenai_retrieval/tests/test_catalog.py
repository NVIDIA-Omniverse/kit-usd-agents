from ovgenai_retrieval import CatalogIndex, load_bundle


def test_get_by_name(tiny_catalog_bundle):
    idx = CatalogIndex(load_bundle(tiny_catalog_bundle))
    ent = idx.get_by_name("omni.ui")
    assert ent is not None
    assert ent["description"] == "UI widgets"
    assert idx.get_by_name("nonexistent") is None


def test_prefix_search(tiny_catalog_bundle):
    idx = CatalogIndex(load_bundle(tiny_catalog_bundle))
    results = idx.prefix_search("omni.ui")
    names = {e["name"] for e in results}
    assert names == {"omni.ui", "omni.ui.scene"}


def test_filter(tiny_catalog_bundle):
    idx = CatalogIndex(load_bundle(tiny_catalog_bundle))
    deprecated = idx.filter(lambda e: "deprecated" in e.get("tags", []))
    assert len(deprecated) == 1
    assert deprecated[0]["name"] == "carb.audio"


def test_catalog_size(tiny_catalog_bundle):
    idx = CatalogIndex(load_bundle(tiny_catalog_bundle))
    assert idx.size == 5
