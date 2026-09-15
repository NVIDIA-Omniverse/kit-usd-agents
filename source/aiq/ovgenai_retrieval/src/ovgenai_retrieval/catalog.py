"""CatalogIndex — by-name / prefix / filter lookup for structured JSON bundles.

Examples of catalog corpora: ``extensions_database.json``, ``usd_atlas.json``,
``app_templates.json``, per-extension ``codeatlas/*.json``, ``api_docs/*.json``.

These do NOT want FAISS or BM25 over chunks. Consumers look up by exact name
or prefix (e.g., ``get_kit_extension_details(name="omni.ui")``). Keeping them
outside HybridRetriever avoids forcing an artificial text-search interface onto
a by-name lookup.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ovgenai_retrieval.bundle import Bundle


class CatalogIndex:
    """Lookup over a structured JSON catalog bundle."""

    def __init__(self, bundle: Bundle, *, key_field: str | None = None):
        if bundle.type != "catalog":
            raise TypeError(f"CatalogIndex requires bundle_type='catalog', got {bundle.type!r}")
        data = bundle.catalog_data
        self._bundle = bundle
        self._key_field = key_field or (bundle.manifest.catalog.key_field if bundle.manifest.catalog else "name")

        # The catalog can be either a dict (name -> entry) or a list of entries.
        if isinstance(data, dict):
            self._by_name: dict[str, Any] = dict(data)
            self._all: list[Any] = list(data.values())
        elif isinstance(data, list):
            self._by_name = {}
            self._all = list(data)
            for entry in data:
                k = entry.get(self._key_field)
                if k is not None:
                    self._by_name[str(k)] = entry
        else:
            raise TypeError(f"Unsupported catalog shape: {type(data).__name__}")

    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._all)

    def get_by_name(self, name: str) -> Any | None:
        return self._by_name.get(name)

    def prefix_search(self, prefix: str, limit: int | None = None) -> list[Any]:
        out: list[Any] = []
        for name, entry in self._by_name.items():
            if name.startswith(prefix):
                out.append(entry)
                if limit is not None and len(out) >= limit:
                    break
        return out

    def filter(self, predicate: Callable[[Any], bool]) -> list[Any]:
        return [e for e in self._all if predicate(e)]

    def all(self) -> list[Any]:
        return list(self._all)
