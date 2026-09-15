# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-package hybrid-retrieval shim for OmniUI MCP.

Re-exports :func:`ovgenai_retrieval.maybe_load_hybrid` with this package's
``qe_domains_default`` baked in. See ``ovgenai_retrieval.shim`` for the full
documentation of supported env vars (``OVAI_RETRIEVAL_MODE``,
``OVAI_FUSION``, ``OVAI_RERANK``, ``OVAI_RERANK_BACKEND``,
``OVAI_RERANK_MODEL``, ``OVAI_BM25_BACKEND``, ``OVAI_QE_DOMAINS``).

Before consolidation this file was a ~185-line copy of the shared logic; only
the ``qe_domains_default`` string varied per package. The four copies drifted
silently — each new env var (e.g. ``OVAI_RERANK_BACKEND``) had to be added in
four places. Now it lives in one place.
"""

from __future__ import annotations

from typing import Any, Optional

from ovgenai_retrieval import maybe_load_hybrid as _shared_maybe_load_hybrid
from ovgenai_retrieval.shim import hits_to_documents, hits_to_documents_with_scores  # noqa: F401 — re-exported

# Glossary categories scoped to this MCP's corpus. Operators can override at
# runtime via ``OVAI_QE_DOMAINS=<csv>``.
_QE_DOMAINS_DEFAULT = "ui,kit,general"


def maybe_load_hybrid(
    bundle_path: str,
    embedder: Any | None = None,
    *,
    top_k: int = 10,
    allow_pickle: bool = True,
) -> Optional[Any]:
    """Return a ``HybridRetriever`` for ``bundle_path`` or ``None`` (legacy fallback)."""
    return _shared_maybe_load_hybrid(
        bundle_path,
        embedder,
        top_k=top_k,
        qe_domains_default=_QE_DOMAINS_DEFAULT,
        allow_pickle=allow_pickle,
    )


__all__ = ["hits_to_documents", "hits_to_documents_with_scores", "maybe_load_hybrid"]
