# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-package hybrid-retrieval shim used by ``*_fns`` packages.

Was previously a 185-line file duplicated four times across kit_fns / isaacsim_fns /
usd_code_fns / omni_ui_fns whose only per-package variation was a single string
naming the glossary categories for query expansion. The four shims drifted (each
caller would have needed updating separately to add a new env var like
``OVAI_RERANK_BACKEND``). This module is the single source of truth; each package's
``utils/hybrid_shim.py`` is now a five-line wrapper that supplies its own
``qe_domains_default`` and re-exports ``hits_to_documents``.

Public env vars (read inside :func:`maybe_load_hybrid`):

- ``OVAI_RETRIEVAL_MODE=hybrid|semantic`` (default ``hybrid``)
- ``OVAI_FUSION=rrf|semantic_only|lexical_only`` (default ``rrf``)
- ``OVAI_RERANK=true|false`` (default ``false``, but honors legacy
  ``*_RERANKER_BACKEND`` env vars — see :func:`_rerank_enabled`)
- ``OVAI_RERANK_BACKEND=cross_encoder|nvidia`` (default derived from legacy
  ``*_RERANKER_BACKEND`` — ``local`` plus a ``*_LOCAL_RERANKER_URL`` routes to
  ``nvidia`` with that local NIM URL, ``local`` without a URL falls back to
  ``cross_encoder``, ``nvidia_api`` → ``nvidia``, otherwise ``cross_encoder``)
- ``OVAI_RERANK_MODEL=...`` (default ``cross-encoder/ms-marco-MiniLM-L6-v2``;
  ``NvidiaReranker`` ignores values that don't start with ``nvidia/`` and uses its
  own default, ``nvidia/llama-nemotron-rerank-vl-1b-v2`` — so operators wanting
  NVIDIA-hosted reranking can set ``OVAI_RERANK=true OVAI_RERANK_BACKEND=nvidia``)
- ``OVAI_BM25_BACKEND=rank_bm25|tantivy|whoosh`` (default ``rank_bm25``)
- ``OVAI_QE_DOMAINS=<csv>`` (overrides the package default)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _hybrid_enabled() -> bool:
    return os.environ.get("OVAI_RETRIEVAL_MODE", "hybrid").lower() == "hybrid"


# Legacy backend env vars carried over from the pre-!658 per-function reranker.
# Operators who set any of these in their docker-compose meant "I want
# reranking on" — even though the function-level reranker hook is gone.
# Honoring them keeps existing deployments working without forcing every
# operator to flip to OVAI_RERANK=true.
_LEGACY_RERANK_ENV_VARS = (
    "KIT_RERANKER_BACKEND",
    "ISAACSIM_RERANKER_BACKEND",
    "USD_CODE_RERANKER_BACKEND",
    "OMNI_UI_RERANKER_BACKEND",
)

# Companion local-NIM URLs from the same docker-compose era. When the
# operator has stood up a NIM that speaks the NVIDIA rerank API at one of
# these URLs (the docker-compose.local.yaml shape), we route the
# HybridRetriever's NvidiaReranker at it via ``base_url`` instead of
# falling back to a sentence-transformers cross_encoder (which the MCP
# Docker images don't install).
_LEGACY_LOCAL_RERANK_URL_ENV_VARS = (
    "KIT_LOCAL_RERANKER_URL",
    "ISAACSIM_LOCAL_RERANKER_URL",
    "USD_CODE_LOCAL_RERANKER_URL",
    "OMNI_UI_LOCAL_RERANKER_URL",
)


def _legacy_local_rerank_url() -> Optional[str]:
    """Return the first non-empty legacy local-NIM rerank URL, if any."""
    for var in _LEGACY_LOCAL_RERANK_URL_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return None


def _rerank_enabled() -> bool:
    """Resolve the effective ``rerank`` flag for HybridRetriever.

    Precedence:

    1. ``OVAI_RERANK`` (explicit override) — ``true`` / ``false``.
    2. Any of the legacy ``*_RERANKER_BACKEND`` env vars set to a non-empty
       value other than ``none`` / ``off`` / ``disabled`` → rerank on.
    3. Default: off.

    This preserves the pre-!658 behavior of ``docker-compose.local.yaml``
    and ``docker-compose.internal.yaml`` where ``KIT_RERANKER_BACKEND=local``
    implied reranking was enabled.
    """
    explicit = os.environ.get("OVAI_RERANK")
    if explicit is not None:
        return explicit.strip().lower() == "true"
    for var in _LEGACY_RERANK_ENV_VARS:
        val = os.environ.get(var, "").strip().lower()
        if val and val not in {"none", "off", "disabled"}:
            return True
    return False


def _rerank_backend_default() -> str:
    """Default backend for ``OVAI_RERANK_BACKEND`` honoring legacy env vars.

    Mapping (first match wins):

    * Legacy ``*_RERANKER_BACKEND=nvidia_api`` → ``nvidia`` (hosted endpoint).
    * Legacy ``*_RERANKER_BACKEND=local`` **and** a ``*_LOCAL_RERANKER_URL``
      is set → ``nvidia`` (we re-purpose ``NvidiaReranker`` with the local
      NIM URL threaded through ``base_url`` in :func:`maybe_load_hybrid`).
      The pre-!658 ``docker-compose.local.yaml`` shape always sets both.
    * Legacy ``*_RERANKER_BACKEND=local`` with no URL → ``cross_encoder``.
      Note this requires the optional ``sentence-transformers`` install
      (``ovgenai-retrieval[rerank]``), which the shipped MCP Docker images
      do not include — operators on this path will hit a clear runtime
      error pointing them at either installing the extra or setting a NIM
      URL.
    * Otherwise: ``cross_encoder``.
    """
    has_local_url = _legacy_local_rerank_url() is not None
    for var in _LEGACY_RERANK_ENV_VARS:
        val = os.environ.get(var, "").strip().lower()
        if val == "nvidia_api" or val == "nvidia":
            return "nvidia"
        if val == "local":
            return "nvidia" if has_local_url else "cross_encoder"
    return "cross_encoder"


def _resolve_qe_domains(default_csv: str) -> Optional[set[str]]:
    """Resolve glossary-category scope for query expansion.

    Precedence: ``OVAI_QE_DOMAINS`` env var > caller-supplied default. Empty / unset
    returns ``None`` (every category eligible — legacy behaviour).
    """
    raw = os.environ.get("OVAI_QE_DOMAINS", default_csv).strip()
    if not raw:
        return None
    return {tok.strip() for tok in raw.split(",") if tok.strip()}


def maybe_load_hybrid(
    bundle_path: str,
    embedder: Any | None = None,
    *,
    top_k: int = 10,
    qe_domains_default: str = "",
    allow_pickle: bool = True,
) -> Optional[Any]:
    """Return a ``HybridRetriever`` when enabled and the bundle loads; ``None`` otherwise.

    The caller (a per-package ``*_fns/utils/hybrid_shim.py`` wrapper) supplies
    ``qe_domains_default`` — the comma-separated glossary categories that scope
    query expansion to this package's corpus (e.g. ``"kit,ui,rendering,general"``
    for kit_fns vs ``"physics,sensors,ai,general"`` for isaacsim_fns). Operators
    can override at runtime via ``OVAI_QE_DOMAINS``.

    ``allow_pickle=True`` by default because today's vendored bundles still ship
    a pickle docstore. Once all bundles are on manifest-v1 + JSON docstore the
    caller can tighten this.

    Returns ``None`` when ``OVAI_RETRIEVAL_MODE=semantic``, when
    ``ovgenai_retrieval`` is somehow unimportable from inside itself (unlikely),
    or when ``load_bundle`` raises — the caller is expected to fall back to its
    legacy FAISS path.
    """
    if not _hybrid_enabled():
        return None
    try:
        # Imported here so that the legacy/semantic-mode path doesn't need the
        # full HybridRetriever dependency chain at module import.
        from ovgenai_retrieval import HybridRetriever, load_bundle
    except ImportError as e:  # pragma: no cover — should be impossible
        logger.warning(
            f"OVAI_RETRIEVAL_MODE=hybrid requested but HybridRetriever import failed "
            f"({e!r}); falling back to legacy FAISS."
        )
        return None
    try:
        bundle = load_bundle(bundle_path, allow_pickle=allow_pickle)
        effective_rerank = _rerank_enabled()
        effective_backend = os.environ.get("OVAI_RERANK_BACKEND", "").strip() or _rerank_backend_default()
        effective_model = os.environ.get("OVAI_RERANK_MODEL", "").strip() or "cross-encoder/ms-marco-MiniLM-L6-v2"
        retriever = HybridRetriever(
            bundle,
            top_k=top_k,
            fusion=os.environ.get("OVAI_FUSION", "rrf"),
            rerank=effective_rerank,
            rerank_backend=effective_backend,
            rerank_model=effective_model,
            bm25_backend=os.environ.get("OVAI_BM25_BACKEND", "rank_bm25"),
            embedder=embedder,
            qe_domains=_resolve_qe_domains(qe_domains_default),
        )

        # If the operator's docker-compose set ``*_LOCAL_RERANKER_URL=<NIM>`` ,
        # the HybridRetriever above was built with rerank_backend=nvidia (via
        # ``_rerank_backend_default``) but the constructor doesn't know how to
        # thread a custom base URL. Swap its ``_reranker`` for one that does
        # — same backend, same model, just pointed at the local NIM. This is
        # done from the shim (NOT inside the vendored HybridRetriever) so the
        # vendor-drift check stays clean.
        local_rerank_url = _legacy_local_rerank_url()
        if effective_rerank and effective_backend == "nvidia" and local_rerank_url:
            try:
                from ovgenai_retrieval.rerank import NvidiaReranker

                model = os.environ.get("OVAI_RERANK_MODEL", "").strip()
                if not model.startswith("nvidia/"):
                    model = "nvidia/llama-nemotron-rerank-vl-1b-v2"
                retriever._reranker = NvidiaReranker(model=model, base_url=local_rerank_url)
                logger.info(f"Rerank routed to local NIM at {local_rerank_url}")
            except Exception as e:
                logger.warning(
                    f"Failed to rewire reranker to local NIM at {local_rerank_url}: "
                    f"{e!r}; reverting to default NvidiaReranker."
                )

        logger.info(f"HybridRetriever loaded for bundle {bundle_path}")
        return retriever
    except Exception as e:
        logger.warning(f"HybridRetriever load failed for {bundle_path}: {e!r}; falling back.")
        return None


def hits_to_documents(hits: list) -> list:
    """Convert a list of :class:`ovgenai_retrieval.Hit` into LangChain Documents.

    Preserves all metadata plus ``_ovai_score`` / ``_ovai_provenance`` extras so
    downstream RAG-context formatters can show fusion / rerank scores. Also
    fills in the LangChain-style ``index_text`` / ``file_path`` / etc. fields
    on the document metadata so callers that read
    ``doc.metadata["index_text"]`` (the pre-consolidation contract) keep
    working.
    """
    from langchain_core.documents import Document  # local import — optional dep in tests

    docs: list = []
    for hit in hits:
        md = dict(hit.metadata) if getattr(hit, "metadata", None) else {}
        # Promote the Hit's first-class fields into the Document metadata,
        # matching what the legacy per-package shims used to do. Don't
        # clobber values the Hit's metadata blob already supplies.
        for fld in (
            "index_text",
            "file_path",
            "line_start",
            "line_end",
            "section_hierarchy",
            "url",
        ):
            val = getattr(hit, fld, None)
            if val is not None and fld not in md:
                md[fld] = val
        md.setdefault("_ovai_score", getattr(hit, "score", None))
        md.setdefault("_ovai_provenance", getattr(hit, "provenance", None))
        docs.append(Document(page_content=hit.content, metadata=md))
    return docs


def hits_to_documents_with_scores(hits: list) -> list:
    """Return ``[(Document, distance), ...]`` pairs matching the LangChain
    ``similarity_search_with_score`` API. The distance is ``1 - h.score`` so
    callers that compute ``1.0 / (1.0 + distance)`` (the legacy similarity
    formula used across the search services) get a monotonic mapping back
    from normalized hybrid fusion scores. Raw lexical/BM25 scores can exceed
    ``1.0`` when ``OVAI_FUSION=lexical_only``; clamp at zero so legacy callers
    never divide by zero or a negative denominator.
    """
    docs = hits_to_documents(hits)
    scores = [max(0.0, 1.0 - float(getattr(h, "score", 0.0))) for h in hits]
    return list(zip(docs, scores))


__all__ = [
    "hits_to_documents",
    "hits_to_documents_with_scores",
    "maybe_load_hybrid",
]
