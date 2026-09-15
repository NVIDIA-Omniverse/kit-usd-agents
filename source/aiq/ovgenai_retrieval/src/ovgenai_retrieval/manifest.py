"""Manifest — bundle descriptor. Canonical schema v1 matching the plan spec."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1"

BundleType = Literal[
    "knowledge",
    "code",
    "settings",
    "extensions_semantic",
    "catalog",
    "raw_docs",
    "code_atlas",
]


@dataclass
class SourceInfo:
    kind: str  # "web_crawl" | "sdk_scan" | "catalog_build" | "code_example_harvest"
    base_url: str | None = None
    kit_version: str | None = None
    usd_version: str | None = None
    package_count: int | None = None
    # Free-form extras preserved verbatim by the emitter (e.g., ``nickname``,
    # ``source`` module path, ``mode``). Stored so callers can round-trip
    # without losing producer-specific provenance.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingsInfo:
    backend: str  # "faiss"
    file: str  # "index.faiss"
    model: str  # e.g. "nvidia/nemotron-3-embed-1b"
    dim: int
    metric: str = "cosine"


@dataclass
class DocstoreInfo:
    format: str  # "json" | "pickle"
    file: str


@dataclass
class LexicalInfo:
    backend: str  # "rank_bm25" | "tantivy" | "whoosh"
    file: str
    tokenizer: str = "whitespace+lowercase"
    k1: float = 1.2
    b: float = 0.75


@dataclass
class CatalogInfo:
    file: str
    key_field: str = "name"


@dataclass
class HierarchyInfo:
    root: str  # e.g., "files/"
    layout: str  # e.g., "package/subpath/page.md"


@dataclass
class ChunksInfo:
    count: int
    index_key: str = "index_text"
    metadata_fields: list[str] = field(default_factory=list)


@dataclass
class Manifest:
    """Bundle descriptor. Emit this as ``manifest.json`` inside a bundle dir."""

    schema_version: str = SCHEMA_VERSION
    bundle_id: str = ""
    bundle_type: BundleType = "knowledge"
    created_at: str = ""
    created_by: str = ""
    source: SourceInfo | None = None
    embeddings: EmbeddingsInfo | None = None
    docstore: DocstoreInfo | None = None
    lexical: LexicalInfo | None = None
    catalog: CatalogInfo | None = None
    hierarchy: HierarchyInfo | None = None
    chunks: ChunksInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        """Drop None fields for compact JSON."""

        def _clean(d: Any) -> Any:
            if isinstance(d, dict):
                return {k: _clean(v) for k, v in d.items() if v is not None}
            if isinstance(d, list):
                return [_clean(i) for i in d]
            return d

        return _clean(asdict(self))

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def write(self, path: str | os.PathLike) -> None:
        Path(path).write_text(self.dumps(), encoding="utf-8")

    # ------------------------------------------------------------------
    # Loading / legacy synthesis
    # ------------------------------------------------------------------

    @classmethod
    def loads(cls, text: str) -> "Manifest":
        data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Manifest":
        return cls.loads(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        import dataclasses as _dc

        def _sub(klass, sub):
            if not sub:
                return None
            # For SourceInfo specifically, corral unknown keys into ``extras``
            # so producer-added fields (``nickname``, ``mode``, etc.) survive.
            if klass is SourceInfo:
                known = {f.name for f in _dc.fields(klass)} - {"extras"}
                known_kwargs = {k: v for k, v in sub.items() if k in known}
                # Start with any explicitly-stored ``extras`` dict from a
                # prior round-trip, then fold in unknown top-level keys.
                extras: dict[str, Any] = dict(sub.get("extras") or {})
                for k, v in sub.items():
                    if k in known or k == "extras":
                        continue
                    extras[k] = v
                return klass(**known_kwargs, extras=extras)
            return klass(**sub)

        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            bundle_id=data.get("bundle_id", ""),
            bundle_type=data.get("bundle_type", "knowledge"),
            created_at=data.get("created_at", ""),
            created_by=data.get("created_by", ""),
            source=_sub(SourceInfo, data.get("source")),
            embeddings=_sub(EmbeddingsInfo, data.get("embeddings")),
            docstore=_sub(DocstoreInfo, data.get("docstore")),
            lexical=_sub(LexicalInfo, data.get("lexical")),
            catalog=_sub(CatalogInfo, data.get("catalog")),
            hierarchy=_sub(HierarchyInfo, data.get("hierarchy")),
            chunks=_sub(ChunksInfo, data.get("chunks")),
        )

    @classmethod
    def synthesize_from_legacy(cls, bundle_dir: str | os.PathLike) -> "Manifest":
        """Best-effort manifest synthesis for pre-v1 bundles.

        Inspects the directory for ``index.faiss``/``index.pkl``/``index.json`` and
        guesses ``bundle_type`` from the directory name. The returned manifest has
        no ``lexical`` sidecar — callers should build BM25 in-process.
        """

        bundle_dir = Path(bundle_dir)
        name = bundle_dir.name.lower()

        # Heuristic bundle-type detection from directory name
        if "knowledge" in name:
            btype: BundleType = "knowledge"
        elif "code_example" in name or "code_rag" in name:
            btype = "code"
        elif "settings" in name:
            btype = "settings"
        elif "extension" in name:
            btype = "extensions_semantic"
        elif "ui_window" in name or "faiss_index_omni_ui" in name:
            btype = "code"
        else:
            btype = "knowledge"

        # Docstore format
        if (bundle_dir / "index.json").exists():
            docstore = DocstoreInfo(format="json", file="index.json")
        elif (bundle_dir / "index.pkl").exists():
            docstore = DocstoreInfo(format="pickle", file="index.pkl")
        else:
            docstore = None

        embeddings = None
        if (bundle_dir / "index.faiss").exists():
            import faiss

            legacy_index = faiss.read_index(str(bundle_dir / "index.faiss"))
            embeddings = EmbeddingsInfo(
                backend="faiss",
                file="index.faiss",
                # A legacy index does not carry enough provenance to infer its
                # model safely. Guessing here can mix incompatible vector spaces.
                model="unknown",
                dim=int(legacy_index.d),
            )

        return cls(
            bundle_id=bundle_dir.name,
            bundle_type=btype,
            created_by="legacy-synthesis",
            embeddings=embeddings,
            docstore=docstore,
            lexical=None,  # signals caller to build BM25 in-process
        )
