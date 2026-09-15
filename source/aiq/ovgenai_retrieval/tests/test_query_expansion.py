"""Query-expansion glossary coverage — REQ-QE-1..6."""

from __future__ import annotations

import logging

import pytest
from ovgenai_retrieval.query_expansion import clear_cache, expand, expand_query


def setup_function(fn):
    clear_cache()


# ---------------------------------------------------------------------------
# Logic (REQ-QE-2): tokenise, case-insensitive, skip redundant, no-match passthrough
# ---------------------------------------------------------------------------


def test_no_match_passthrough():
    assert expand("plain english no abbreviations here") == ("plain english no abbreviations here")


def test_empty_passthrough():
    assert expand("") == ""


def test_case_insensitive():
    r1 = expand("how to use Gf.Matrix4d")
    r2 = expand("how to use gf.matrix4d")  # lowercase
    assert "Graphics Foundation" in r1
    assert "Graphics Foundation" in r2


def test_redundancy_skip_when_expansion_in_query():
    # If the user already wrote "Subsurface Scattering" in the query,
    # expanding "SSS" should not re-add "Subsurface Scattering".
    r = expand("How does Subsurface Scattering (SSS) work in MDL?")
    occurrences = r.lower().count("subsurface scattering")
    assert occurrences == 1  # only the one the user wrote


def test_redundancy_skip_for_multiple_hits_in_same_query():
    # SSS appears twice; we should only add the expansion once.
    r = expand("SSS and more SSS everywhere")
    assert r.count("Subsurface Scattering") == 1


def test_token_boundary_does_not_match_substring():
    # "ass" should not match "SSS" — boundary matters.
    assert expand("assess the assassin") == "assess the assassin"


# ---------------------------------------------------------------------------
# Domain coverage (REQ-QE-1): each of 9 domains fires
# ---------------------------------------------------------------------------


def test_usd_module_expansion_Gf():
    r = expand("Gf vector math")
    assert "Graphics Foundation" in r


def test_usd_module_expansion_Sdf():
    # Sdf is now ambiguous (usd_modules vs physics' "Signed Distance Field"),
    # so a query has to anchor the domain via another USD-side token.
    # ``UsdGeom`` lives in usd_modules → that's enough signal to pick the
    # USD meaning.
    r = expand("UsdGeom and Sdf.CopySpec")
    assert "Scene Description Format" in r or "Scene Description Foundations" in r
    assert "Signed Distance Field" not in r


def test_usd_LIVRPS():
    r = expand("LIVRPS ordering")
    assert "composition arc strength ordering" in r


def test_kit_expansion_KAT():
    r = expand("scaffold a new KAT app")
    assert "kit-app-template" in r


def test_kit_expansion_OmniGraph():
    r = expand("OG node editor")
    assert "OmniGraph" in r


def test_rendering_PBR():
    r = expand("PBR materials in MDL")
    assert "Physically Based Rendering" in r
    assert "Material Definition Language" in r


def test_rendering_SSS():
    r = expand("reducing SSS artifacts")
    assert "Subsurface Scattering" in r


def test_rendering_DLSS():
    r = expand("DLSS frame generation")
    assert "Deep Learning Super Sampling" in r


def test_physics_PhysX_RB():
    r = expand("PhysX RB articulation")
    assert "NVIDIA PhysX simulation engine" in r
    assert "Rigid Body" in r


def test_sensors_LiDAR():
    r = expand("LiDAR point cloud in IsaacSim")
    assert "Light Detection And Ranging" in r


def test_ui_VStack():
    r = expand("VStack layout with ComboBox inputs")
    assert "Vertical Stack layout" in r
    assert "drop-down selector widget" in r


def test_ai_LLM_NIM():
    r = expand("NIM serving an LLM")
    assert "NVIDIA Inference Microservice" in r
    assert "Large Language Model" in r


def test_geometry_Xform():
    r = expand("Xform prim with Xformable ops")
    assert "USD transform prim" in r


def test_general_OV_IsaacLab():
    r = expand("OV and IsaacLab integration")
    assert "Omniverse" in r
    # "IsaacLab" also fires
    assert "Isaac Lab" in r


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def test_expand_query_alias():
    # Legacy callers import expand_query; alias should match expand.
    assert expand_query("Gf.Matrix4d") == expand("Gf.Matrix4d")


# ---------------------------------------------------------------------------
# Logging (REQ-QE-6)
# ---------------------------------------------------------------------------


def test_logging_off_by_default(caplog, monkeypatch):
    monkeypatch.delenv("OVAI_QE_LOG", raising=False)
    caplog.set_level(logging.INFO)
    expand("SSS in PBR")
    # No INFO message from query_expansion
    assert not any("query_expansion" in r.message for r in caplog.records)


def test_logging_on_when_env_set(caplog, monkeypatch):
    monkeypatch.setenv("OVAI_QE_LOG", "1")
    caplog.set_level(logging.INFO)
    expand("SSS")
    assert any("query_expansion fired" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Custom glossary override (REQ-QE-2 flexibility)
# ---------------------------------------------------------------------------


def test_custom_glossary_override():
    g = {"FOO": ["Food Object Ontology"]}
    assert "Food Object Ontology" in expand("explain FOO", glossary=g)
    # Default glossary not consulted:
    assert "Subsurface Scattering" not in expand("SSS", glossary=g)


# ---------------------------------------------------------------------------
# Glossary collision audit — Phase 1+2+3 of glossary_fix.md.
# Goal: SDF / DOF have multiple meanings across categories. Without a
# domain filter or contextual signal, no expansion fires (no pollution).
# With clear context (other in-domain abbreviations in the query), the
# right meaning is auto-selected from the glossary's own structure.
# ---------------------------------------------------------------------------


def test_ambiguous_no_context_no_expansion():
    """SDF / DOF with no domain-anchoring context: don't expand at all.

    Pre-fix: both meanings got appended (Scene Description Format,
    SdfLayer SdfPath, Signed Distance Field) on every SDF query. That
    polluted retrieval. Now we stay silent until context disambiguates.
    """
    assert expand("what is SDF") == "what is SDF"
    assert expand("about DOF") == "about DOF"
    # And the inserted-noise terms must not appear:
    out = expand("SDF and DOF in a vacuum")
    assert "Signed Distance Field" not in out
    assert "Scene Description Format" not in out
    assert "Depth of Field" not in out
    assert "Degree Of Freedom" not in out


def test_sdf_picks_usd_when_usd_context_in_query():
    """A USD-flavoured token in the query disambiguates SDF → USD side.

    Pre-fix this gave both meanings; post-fix it gives only the USD one
    because ``UsdGeom`` (also in the ``usd_modules`` category) signals
    that the query lives in that domain.
    """
    out = expand("UsdGeom and SDF references")
    assert "Scene Description Format" in out or "Scene Description Foundations" in out
    assert "Signed Distance Field" not in out


def test_dof_picks_physics_when_physics_context_in_query():
    """A physics-flavoured token disambiguates DOF → physics side.

    The glossary's ``physics`` category has SDF as one of its other
    members. Putting SDF in the same query (under physics intent)
    raises the physics score for DOF.
    """
    # Put PBR (rendering category) → DOF resolves to rendering's "Depth of Field".
    out = expand("PBR shading and DOF blur")
    assert "Depth of Field" in out
    assert "Degree Of Freedom" not in out


def test_explicit_domains_kwarg_scopes_expansion():
    """Caller passes domains={'physics'} → only physics-tagged entries fire.

    DOF should expand to "Degree Of Freedom" (physics) regardless of
    other context in the query.
    """
    out = expand("rigid body DOF constraints", domains={"physics"})
    assert "Degree Of Freedom" in out
    assert "Depth of Field" not in out


def test_explicit_domains_kwarg_filters_out_unrelated():
    """A USD query passed domains={'physics'} → no USD expansion fires."""
    out = expand("UsdGeom prim", domains={"physics"})
    # Without the USD category enabled, ``UsdGeom`` has no eligible entry.
    assert out == "UsdGeom prim"


def test_env_var_domains_picked_up(monkeypatch):
    """OVAI_QE_DOMAINS=… env var equivalent to domains= kwarg."""
    monkeypatch.setenv("OVAI_QE_DOMAINS", "physics")
    out = expand("rigid body DOF constraints")
    assert "Degree Of Freedom" in out
    assert "Depth of Field" not in out


def test_unambiguous_abbreviation_unaffected():
    """Single-meaning entries (LIVRPS / SSS / PBR) keep working."""
    assert "Subsurface Scattering" in expand("how does SSS work")
    assert "Physically Based Rendering" in expand("PBR materials")
    out = expand("explain LIVRPS")
    # Legacy seed had "local"; YAMLs may capitalise as "Local" — either
    # casing is fine, only that *some* expansion fired.
    assert "local" in out.lower()
