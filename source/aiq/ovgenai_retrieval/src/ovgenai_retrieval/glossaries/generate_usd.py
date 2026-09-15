"""Generate a version-specific USD glossary YAML from a usd_atlas_vXX_YY.json.

The atlas is produced by the USD Code MCP's data_collection pipeline; each
release of USD (25.02, 25.11, etc.) gets its own file. Entries look like::

    {
      "modules": {"Gf": {"name": "Gf", "class_names": ["Matrix4d", ...]}, ...},
      "classes": {"Gf.Matrix4d": {...}, ...}
    }

This generator emits ``usd_v25_XX.yaml`` in the ``ovgenai_retrieval.glossaries``
package format. Loader code (``query_expansion.expand``) merges any
``usd_*.yaml`` it finds alongside ``default.yaml``, so we can layer per-USD
version without touching the defaults.

Usage (as a script)::

    python -m ovgenai_retrieval.glossaries.generate_usd \\
        /path/to/usd_atlas_v25_02.json \\
        --out src/ovgenai_retrieval/glossaries/usd_v25_02.yaml

Or programmatically via :func:`generate_from_atlas`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# Well-known gloss strings for the pxr modules we already hand-authored in
# ``default.yaml``. When we encounter the same module here we emit the hand
# gloss as the first expansion term so per-version files stay coherent with
# the default shipped with the package.
_KNOWN_GLOSS: dict[str, str] = {
    "Ar": "Asset Resolver",
    "CameraUtil": "Camera Utilities",
    "Garch": "Graphics Architecture",
    "GeomUtil": "Geometry Utilities",
    "Gf": "Graphics Foundation",
    "Glf": "GL Foundation",
    "Hd": "Hydra",
    "HdSt": "Hydra Storm",
    "HdGp": "Hydra Generative Plugin",
    "HdMtlx": "Hydra MaterialX",
    "HdPrman": "Hydra RenderMan",
    "HdArnold": "Hydra Arnold",
    "HdCycles": "Hydra Cycles",
    "HdRpr": "Hydra Render Plugin",
    "HdRsg": "Hydra Render Scene Graph",
    "HdSi": "Hydra Scene Index",
    "Hf": "Hydra Foundations",
    "Hgi": "Hydra Graphics Interface",
    "HgiGL": "Hydra Graphics Interface GL",
    "HgiMetal": "Hydra Graphics Interface Metal",
    "HgiVulkan": "Hydra Graphics Interface Vulkan",
    "HgiInterop": "Hydra Graphics Interop",
    "Hio": "Hydra I/O",
    "Hdui": "Hydra UI",
    "Js": "JSON",
    "Kind": "Kind Registry",
    "Ndr": "Node Definition Registry",
    "Pcp": "Prim Cache Population",
    "Plug": "Plugin Registry",
    "Sdf": "Scene Description Foundations",
    "Sdr": "Shader Definition Registry",
    "Tf": "Tools Foundations",
    "Trace": "Trace",
    "Usd": "Universal Scene Description",
    "UsdAppUtils": "USD App Utilities",
    "UsdGeom": "USD Geometry",
    "UsdHydra": "USD Hydra",
    "UsdLux": "USD Lighting",
    "UsdMedia": "USD Media",
    "UsdPhysics": "USD Physics",
    "UsdProc": "USD Procedural",
    "UsdRender": "USD Render",
    "UsdRi": "USD RenderMan",
    "UsdShade": "USD Shading",
    "UsdSkel": "USD Skeleton / Animation",
    "UsdUI": "USD UI",
    "UsdUtils": "USD Utilities",
    "UsdVol": "USD Volumes",
    "Usdviewq": "usdview Qt shell",
    "Vt": "Value Types",
    "Work": "Work Dispatcher",
}


def _gloss_for(module_name: str) -> str:
    return _KNOWN_GLOSS.get(module_name, f"USD pxr module {module_name}")


def _classes_summary(class_names: Iterable[str], *, limit: int = 8) -> str | None:
    """Return a comma-joined list of up to ``limit`` class names or None."""
    names = list(class_names or [])
    if not names:
        return None
    shown = names[:limit]
    extra = f" and {len(names) - limit} more" if len(names) > limit else ""
    return f"classes: {', '.join(shown)}{extra}"


def generate_from_atlas(
    atlas: dict[str, Any],
    *,
    version_label: str | None = None,
) -> str:
    """Render a YAML glossary document string from a parsed atlas.

    The document has a single ``usd_modules`` section (same key as the
    default glossary) plus an optional ``usd_modules_classes`` section that
    lists the top classes per module as secondary expansions.
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError("Generating a USD glossary requires PyYAML: pip install pyyaml") from e

    modules = atlas.get("modules") or {}
    usd_modules: dict[str, list[str]] = {}
    for mod_name, meta in modules.items():
        if not mod_name or mod_name == ".":
            continue
        gloss = _gloss_for(mod_name)
        classes = (meta or {}).get("class_names") if isinstance(meta, dict) else None
        expansions = [gloss]
        summary = _classes_summary(classes or [])
        if summary:
            expansions.append(summary)
        usd_modules[mod_name] = expansions

    doc: dict[str, Any] = {}
    if version_label:
        doc["_meta"] = {
            "generator": "ovgenai_retrieval.glossaries.generate_usd",
            "version_label": version_label,
            "module_count": len(usd_modules),
        }
    doc["usd_modules"] = dict(sorted(usd_modules.items()))
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="generate_usd_glossary",
        description="Emit a USD glossary YAML derived from a usd_atlas_vXX_YY.json.",
    )
    ap.add_argument("atlas", type=Path, help="Path to usd_atlas_vXX_YY.json")
    ap.add_argument("--out", type=Path, default=None, help="Output path (default: usd_vXX_YY.yaml next to the atlas).")
    ap.add_argument("--version-label", default=None, help="Version label to include in the YAML's _meta block.")
    args = ap.parse_args(argv)

    atlas_path: Path = args.atlas
    if not atlas_path.is_file():
        print(f"Atlas not found: {atlas_path}", file=sys.stderr)
        return 2
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))

    label = args.version_label or atlas_path.stem.replace("usd_atlas_", "")
    yaml_text = generate_from_atlas(atlas, version_label=label)

    out = args.out or atlas_path.with_name(f"usd_{label}.yaml")
    out.write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {out}  ({yaml_text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
