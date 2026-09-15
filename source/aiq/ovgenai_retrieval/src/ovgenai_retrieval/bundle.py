"""Bundle — wraps a bundle directory: manifest + FAISS + docstore + optional BM25.

Three generations supported:

- **Legacy pickle** (today's MCP data): ``index.faiss`` + ``index.pkl``
- **Legacy JSON** (post-faiss-safe migration): ``index.faiss`` + ``index.json``
- **Manifest v1** (upgraded rag-prep and data_collection scripts):
  ``manifest.json`` + ``index.faiss`` + ``index.json`` + ``bm25.<backend>`` + ``files/``

``load_bundle(path)`` dispatches to the correct loader and returns a uniform
``Bundle`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ovgenai_retrieval._compat import DocStore, DocStoreRecord, load_json_docstore, load_pickle_docstore
from ovgenai_retrieval.manifest import Manifest


@dataclass
class Bundle:
    """A loaded bundle. Consumers read ``manifest.bundle_type`` to dispatch."""

    path: Path
    manifest: Manifest
    docstore: DocStore | None = None
    """None for ``catalog`` bundles (they use :attr:`catalog_data` instead)."""
    catalog_data: Any = None
    """Raw JSON blob loaded from the catalog file, for ``bundle_type='catalog'``."""

    @property
    def type(self) -> str:
        return self.manifest.bundle_type

    @property
    def has_faiss(self) -> bool:
        return self.manifest.embeddings is not None and (self.path / self.manifest.embeddings.file).exists()

    @property
    def has_lexical_sidecar(self) -> bool:
        return self.manifest.lexical is not None and (self.path / self.manifest.lexical.file).exists()

    @property
    def faiss_index_path(self) -> Path:
        assert self.manifest.embeddings is not None, "bundle has no embeddings"
        return self.path / self.manifest.embeddings.file

    def iter_records(self) -> list[DocStoreRecord]:
        """Convenience: returns all docstore records ordered by faiss row index."""
        assert self.docstore is not None, "bundle has no docstore"
        # Build index->record map; records are stored doc_id-keyed
        idx_to_record: dict[int, DocStoreRecord] = {}
        for rec in self.docstore.records:
            faiss_idx = self.docstore.id_to_faiss_index.get(rec.doc_id)
            if faiss_idx is not None:
                idx_to_record[faiss_idx] = rec
        expected_rows = list(range(len(self.docstore.records)))
        if sorted(idx_to_record) != expected_rows:
            raise ValueError(f"Bundle {self.path} has a non-contiguous or incomplete FAISS row map")
        if self.manifest.chunks is not None and self.manifest.chunks.count != len(expected_rows):
            raise ValueError(
                f"Bundle {self.path} manifest declares {self.manifest.chunks.count} chunks "
                f"but the docstore contains {len(expected_rows)}"
            )
        return [idx_to_record[i] for i in expected_rows]


def _load_catalog(bundle_dir: Path, manifest: Manifest) -> Any:
    if manifest.catalog is None:
        # Fallback: look for a single *.json that isn't index.json
        candidates = [p for p in bundle_dir.glob("*.json") if p.name not in {"index.json", "manifest.json"}]
        if len(candidates) == 1:
            return _read_json(candidates[0])
        raise ValueError(
            f"Catalog bundle at {bundle_dir} has no 'catalog' field in manifest "
            f"and no single catalog JSON could be auto-detected."
        )
    return _read_json(bundle_dir / manifest.catalog.file)


def _read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(
    path: str | Path,
    *,
    allow_pickle: bool = False,
) -> Bundle:
    """Load a bundle from a directory.

    Args:
        path: Path to the bundle directory.
        allow_pickle: Must be True to load legacy pickle docstores. Default False.

    Returns:
        A :class:`Bundle` instance.
    """
    bundle_dir = Path(path)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    manifest_file = bundle_dir / "manifest.json"
    if manifest_file.exists():
        manifest = Manifest.load(manifest_file)
    else:
        manifest = Manifest.synthesize_from_legacy(bundle_dir)

    # Catalog-shaped bundles skip docstore loading
    if manifest.bundle_type == "catalog":
        catalog_data = _load_catalog(bundle_dir, manifest)
        return Bundle(path=bundle_dir, manifest=manifest, catalog_data=catalog_data)

    docstore: DocStore | None = None
    if manifest.docstore is not None:
        docfile = bundle_dir / manifest.docstore.file
        if manifest.docstore.format == "json":
            docstore = load_json_docstore(docfile)
        elif manifest.docstore.format == "pickle":
            docstore = load_pickle_docstore(docfile, allow_pickle=allow_pickle)

    # Fallback for legacy bundles with no docstore field in synthesized manifest
    if docstore is None:
        if (bundle_dir / "index.json").exists():
            docstore = load_json_docstore(bundle_dir / "index.json")
        elif (bundle_dir / "index.pkl").exists():
            docstore = load_pickle_docstore(bundle_dir / "index.pkl", allow_pickle=allow_pickle)

    return Bundle(path=bundle_dir, manifest=manifest, docstore=docstore)
