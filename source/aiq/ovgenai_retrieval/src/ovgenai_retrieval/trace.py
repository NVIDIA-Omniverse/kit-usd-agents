"""Eval trajectory record emitter.

Matches Andrey Voroshilov's MCP Eval Tracker schema — each record describes
one retrieval call with its inputs, hits, timing, and provenance. Records
accumulate into an NDJSON stream that Andrey's runner ingests as-is.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ovgenai_retrieval.hits import Hit


@dataclass
class TraceRecord:
    """One retrieval call. Key fields aligned with mcp-evals import format."""

    timestamp_ms: int
    tool: str  # "search_kit_knowledge" | "get_kit_extension_details" | ...
    query: str
    hits: list[dict[str, Any]]
    latency_ms: float
    condition: str = ""  # "NT" | "WEB" | "MCP+skills" | "FS+skills" | "COMBINED+skills"
    bundle_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class TraceEmitter:
    """NDJSON appender. Safe to construct with path=None (no-op)."""

    def __init__(self, path: str | os.PathLike | None, *, condition: str = ""):
        self._path = Path(path) if path else None
        self._condition = condition
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, rec: TraceRecord) -> None:
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8") as f:
            f.write(rec.to_json())
            f.write("\n")

    @contextmanager
    def time_tool(self, tool: str, query: str, bundle_id: str = ""):
        """Context manager: times a tool call, emits a record on exit.

        Usage::

            with emitter.time_tool("search_kit_knowledge", query, bundle_id) as slot:
                hits = retriever.retrieve(query)
                slot["hits"] = hits
        """
        slot: dict[str, Any] = {"hits": []}
        t0 = time.perf_counter()
        try:
            yield slot
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            hits_raw = slot.get("hits") or []
            hit_dicts: list[dict[str, Any]] = []
            for h in hits_raw:
                if isinstance(h, Hit):
                    hit_dicts.append(
                        {
                            "doc_id": h.doc_id,
                            "score": h.score,
                            "file_path": h.file_path,
                            "section": h.section_hierarchy,
                            "url": h.url,
                            "provenance": h.provenance,
                        }
                    )
                else:
                    hit_dicts.append(dict(h))
            self.emit(
                TraceRecord(
                    timestamp_ms=int(time.time() * 1000),
                    tool=tool,
                    query=query,
                    hits=hit_dicts,
                    latency_ms=latency_ms,
                    condition=self._condition,
                    bundle_id=bundle_id,
                )
            )
