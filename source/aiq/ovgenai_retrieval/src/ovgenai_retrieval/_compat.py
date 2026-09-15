"""Backward compatibility for legacy bundle formats.

Two legacy formats exist:

1. **Pickle docstore** — ``index.faiss`` + ``index.pkl`` produced by LangChain's
   ``FAISS.save_local``. The pickle holds ``(docstore, id_to_uuid_map)``. We
   refuse to deserialize arbitrary pickle payloads; instead we reuse the safe
   JSON conversion that ``kit-usd-agents/source/aiq/*_fns/src/*/utils/faiss_safe.py``
   already ships in the ``faiss-safe-loader`` work.

2. **JSON docstore** — ``index.faiss`` + ``index.json`` with schema compatible
   with the faiss-safe loader. This is the post-migration format.

This module loads both and normalizes them into a uniform ``DocStore`` shape.
"""

from __future__ import annotations

import json
import pickle  # nosec B403 — used only via :class:`PickleRefusal` gate
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocStoreRecord:
    """One record from the docstore. Uniform across pickle and JSON origins."""

    doc_id: str
    content: str
    index_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocStore:
    records: list[DocStoreRecord]
    id_to_faiss_index: dict[str, int]
    """Map from doc_id to the row index inside the FAISS flat store."""


class PickleRefusal(RuntimeError):
    """Raised when a legacy pickle bundle is loaded without explicit opt-in."""


def load_json_docstore(path: str | Path) -> DocStore:
    """Load the faiss-safe JSON docstore.

    Supports two equivalent schemas:

    1. kit-usd-agents's faiss_safe format (format_version=1)::

        {
          "format_version": 1,
          "docstore": {"<id>": {"page_content": "...", "metadata": {...}}, ...},
          "index_to_docstore_id": {"<int>": "<id>", ...}
        }

    2. Legacy/alternate with the same shape but keyed under ``"documents"``
       instead of ``"docstore"`` (used in the library's own test fixtures).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"FAISS docstore at {path} must be a JSON object")
    format_version = data.get("format_version")
    if format_version not in (None, 1):
        raise ValueError(f"Unsupported FAISS docstore format_version={format_version!r} at {path}")
    # Accept both canonical keys. ``docstore`` wins if both are present.
    docs = data.get("docstore") or data.get("documents") or {}
    id_map = data.get("index_to_docstore_id", {})
    if not isinstance(docs, dict) or not isinstance(id_map, dict):
        raise ValueError(f"FAISS docstore and row map at {path} must be JSON objects")

    # Build a reverse map: doc_id -> faiss row index
    rev: dict[str, int] = {}
    for faiss_idx_str, doc_id in id_map.items():
        if not isinstance(doc_id, str):
            raise ValueError(f"FAISS document ID at row {faiss_idx_str!r} must be a string")
        faiss_idx = int(faiss_idx_str)
        if doc_id in rev:
            raise ValueError(f"Duplicate FAISS document ID {doc_id!r} in row map at {path}")
        rev[doc_id] = faiss_idx

    rows = sorted(rev.values())
    if rows != list(range(len(rows))):
        raise ValueError(f"FAISS row map at {path} must be contiguous from zero")
    if set(rev) != set(docs):
        raise ValueError(
            f"FAISS row map and docstore IDs differ at {path}: " f"mapped={len(rev)}, documents={len(docs)}"
        )

    records: list[DocStoreRecord] = []
    for doc_id, doc in docs.items():
        if not isinstance(doc_id, str) or not isinstance(doc, dict):
            raise ValueError(f"Invalid FAISS document record in {path}")
        meta = dict(doc.get("metadata") or {})
        # rag-prep stores index heading in metadata["index_text"]; agents use page_content
        index_text = str(meta.pop("index_text", "")) or ""
        records.append(
            DocStoreRecord(
                doc_id=doc_id,
                content=doc.get("page_content", ""),
                index_text=index_text,
                metadata=meta,
            )
        )
    return DocStore(records=records, id_to_faiss_index=rev)


def load_pickle_docstore(path: str | Path, *, allow_pickle: bool = False) -> DocStore:
    """Load a legacy LangChain pickle docstore.

    Requires explicit ``allow_pickle=True`` because pickles can execute arbitrary
    code. The preferred path is to convert pickle bundles to JSON once using
    kit-usd-agents's ``faiss_safe`` utilities and then use ``load_json_docstore``.
    """
    if not allow_pickle:
        raise PickleRefusal(
            "Legacy pickle docstore detected at %s. Refusing to deserialize "
            "arbitrary pickle. Pass allow_pickle=True only if you trust the "
            "bundle's origin, or convert it to JSON with kit-usd-agents's "
            "faiss_safe tooling first." % path
        )
    with open(path, "rb") as f:
        docstore_obj, id_map = pickle.load(f)  # nosec B301 — gated above

    # LangChain's InMemoryDocstore exposes ._dict: {uuid: Document}
    inner = getattr(docstore_obj, "_dict", None)
    if inner is None and isinstance(docstore_obj, dict):
        inner = docstore_obj

    records: list[DocStoreRecord] = []
    rev: dict[str, int] = {}
    for faiss_idx, doc_id in id_map.items():
        doc = inner.get(doc_id) if inner is not None else None
        content = getattr(doc, "page_content", None) or ""
        metadata_raw = getattr(doc, "metadata", None) or {}
        metadata = dict(metadata_raw)
        index_text = str(metadata.pop("index_text", "")) or ""
        records.append(
            DocStoreRecord(
                doc_id=doc_id,
                content=content,
                index_text=index_text,
                metadata=metadata,
            )
        )
        rev[doc_id] = int(faiss_idx)
    return DocStore(records=records, id_to_faiss_index=rev)
