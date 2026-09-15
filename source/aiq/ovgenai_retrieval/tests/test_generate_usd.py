"""USD glossary auto-generation from a usd_atlas_vXX_YY.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from ovgenai_retrieval.glossaries.generate_usd import generate_from_atlas, main


def _tiny_atlas() -> dict:
    return {
        "modules": {
            "Gf": {"name": "Gf", "class_names": ["Matrix4d", "Vec3d", "Rotation"]},
            "Sdf": {"name": "Sdf", "class_names": ["Layer", "PrimSpec"]},
            "WeirdNewOne": {"name": "WeirdNewOne", "class_names": ["Foo", "Bar"]},
        },
        "classes": {},
        "methods": {},
    }


def test_generate_from_atlas_has_known_gloss():
    doc = generate_from_atlas(_tiny_atlas(), version_label="v25_02")
    parsed = yaml.safe_load(doc)
    assert "usd_modules" in parsed
    gf = parsed["usd_modules"]["Gf"]
    assert gf[0] == "Graphics Foundation"
    # Classes summary should be included as a secondary expansion.
    assert any("Matrix4d" in s for s in gf)


def test_generate_unknown_module_falls_back():
    doc = generate_from_atlas(_tiny_atlas())
    parsed = yaml.safe_load(doc)
    entry = parsed["usd_modules"]["WeirdNewOne"]
    assert "USD pxr module WeirdNewOne" in entry[0]


def test_generate_skips_dot_module():
    atlas = {"modules": {".": {"name": ""}, "Gf": {"name": "Gf", "class_names": []}}}
    parsed = yaml.safe_load(generate_from_atlas(atlas))
    assert "." not in parsed["usd_modules"]
    assert "Gf" in parsed["usd_modules"]


def test_generate_cli(tmp_path: Path):
    atlas_path = tmp_path / "usd_atlas_v25_02.json"
    atlas_path.write_text(json.dumps(_tiny_atlas()), encoding="utf-8")
    rc = main([str(atlas_path), "--out", str(tmp_path / "out.yaml"), "--version-label", "v25_02"])
    assert rc == 0
    parsed = yaml.safe_load((tmp_path / "out.yaml").read_text(encoding="utf-8"))
    assert parsed["_meta"]["version_label"] == "v25_02"
    assert "Gf" in parsed["usd_modules"]


def test_generated_yaml_is_loadable_by_query_expansion(tmp_path: Path, monkeypatch):
    """Emit a generated YAML into a fake glossaries dir and verify expansion picks it up."""
    from ovgenai_retrieval import query_expansion

    # Build a fresh glossaries dir and redirect the loader at it.
    fake_glossaries = tmp_path / "glossaries"
    fake_glossaries.mkdir()
    doc = generate_from_atlas(_tiny_atlas(), version_label="v25_02")
    (fake_glossaries / "usd_v25_02.yaml").write_text(doc, encoding="utf-8")

    monkeypatch.setattr(query_expansion, "_GLOSSARIES_DIR", fake_glossaries)
    query_expansion.clear_cache()
    try:
        # Gf is in the generated glossary — expand() should append its gloss.
        out = query_expansion.expand("resolve Gf.Matrix4d scaling")
        assert "Graphics Foundation" in out
    finally:
        query_expansion.clear_cache()
