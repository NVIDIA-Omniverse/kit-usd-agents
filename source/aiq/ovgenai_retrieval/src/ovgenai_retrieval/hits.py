"""Hit — uniform result shape across semantic, lexical, and fused retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hit:
    """One retrieval result.

    Attributes:
        doc_id: Stable identifier for the source chunk.
        content: The chunk text the caller should feed to a downstream LLM.
        score: Fused score (higher is better). Semantic backends report similarity;
            lexical backends report BM25; RRF fusion reports rank-reciprocal sum.
        index_text: The short text (usually section title or question) that the
            embedder keyed on. Useful for debugging and trace records.
        file_path: Relative path within the bundle's ``files/`` subtree.
            None for legacy bundles that didn't emit it.
        line_start: First line of the chunk within ``file_path``. None if unknown.
        line_end: Last line of the chunk within ``file_path``. None if unknown.
        section_hierarchy: Heading chain such as ``["Stage", "Namespace Editing"]``.
        url: Source URL when the chunk came from a web crawl.
        metadata: Anything else the producer stored — kit_version, package, etc.
        provenance: Per-backend rank info — keys such as
            ``{"semantic_rank": 3, "lexical_rank": 7, "rrf_score": 0.024}``.
    """

    doc_id: str
    content: str
    score: float
    index_text: str = ""
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    section_hierarchy: list[str] = field(default_factory=list)
    url: str | None = None
    match_mode: str = "semantic"  # "semantic" | "keyword" | "both"
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
