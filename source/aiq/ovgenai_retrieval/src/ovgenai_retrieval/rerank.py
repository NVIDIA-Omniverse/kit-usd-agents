"""Optional rerankers.

Two cross-encoder-style backends are supported:

* :class:`CrossEncoderReranker` — local sentence-transformers model. Opt-in via
  ``pip install ovgenai-retrieval[rerank]``.
* :class:`NvidiaReranker` — hit the NVIDIA hosted rerank endpoint
  (``nvidia/llama-nemotron-rerank-vl-1b-v2`` by default). Uses the existing
  ``langchain-nvidia-ai-endpoints`` hard dependency, so no extra install.
"""

from __future__ import annotations

from typing import Optional

from ovgenai_retrieval.models import DEFAULT_RERANK_MODEL


def _normalize_nim_base_url(base_url: str) -> str:
    """Return the API base expected by ``NVIDIARerank``.

    Retriever NIM exposes ``/v1/ranking`` while the LangChain client appends
    ``/ranking`` itself. Accept the host-root form used by the compose files as
    well as callers that supplied either the API base or full ranking path.
    """
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/ranking"):
        return normalized[: -len("/ranking")]
    if normalized.endswith("/ranking"):
        normalized = normalized[: -len("/ranking")]
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


class CrossEncoderReranker:
    """Thin sentence-transformers wrapper. Lazy-imports on first use."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self._model_name = model
        self._impl: Optional[object] = None  # sentence_transformers.CrossEncoder

    def _ensure(self) -> None:
        if self._impl is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover — exercised only when extra missing
            raise RuntimeError(
                "Cross-encoder reranker requires the optional dependency. "
                "Install with `pip install ovgenai-retrieval[rerank]`."
            ) from e
        self._impl = CrossEncoder(self._model_name)

    def score(self, query: str, candidates: list[str]) -> list[float]:
        """Return one relevance score per candidate."""
        if not candidates:
            return []
        self._ensure()
        pairs = [(query, c) for c in candidates]
        scores = self._impl.predict(pairs)  # type: ignore[attr-defined]
        return [float(s) for s in scores]


class NvidiaReranker:
    """Cross-encoder-style reranker hitting the NVIDIA hosted rerank endpoint.

    Wraps :class:`langchain_nvidia_ai_endpoints.NVIDIARerank` so it can be
    dropped into :class:`~ovgenai_retrieval.hybrid.HybridRetriever` in place of
    :class:`CrossEncoderReranker`. Produces candidate-order scores so callers
    don't need to re-sort the reranked document list themselves.
    """

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._impl: Optional[object] = None  # NVIDIARerank

    def _ensure(self) -> None:
        if self._impl is not None:
            return
        try:
            from langchain_nvidia_ai_endpoints import NVIDIARerank
        except ImportError as e:  # pragma: no cover — hard dep, shouldn't normally fire
            raise RuntimeError(
                "NvidiaReranker requires langchain-nvidia-ai-endpoints " "(already a hard dep of ovgenai-retrieval)."
            ) from e
        import os

        key = self._api_key or os.environ.get("NVIDIA_API_KEY")
        kwargs: dict[str, object] = {"model": self._model}
        if key:
            kwargs["api_key"] = key
        if self._base_url:
            kwargs["base_url"] = _normalize_nim_base_url(self._base_url)
        self._impl = NVIDIARerank(**kwargs)

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        self._ensure()
        # NVIDIARerank.compress_documents(documents, query) -> list[Document]
        # with .metadata['relevance_score']. The endpoint may return the
        # reranked documents in a different order than we sent them, so we
        # tag each input with its original index and rebuild a dense list.
        from langchain_core.documents import Document as _Doc

        docs = [_Doc(page_content=c, metadata={"_i": i}) for i, c in enumerate(candidates)]
        reranked = self._impl.compress_documents(documents=docs, query=query)  # type: ignore[attr-defined]
        scores = [0.0] * len(candidates)
        for d in reranked:
            i = d.metadata.get("_i")
            s = d.metadata.get("relevance_score", 0.0)
            if i is not None:
                scores[i] = float(s)
        return scores


def noop_rerank(query: str, candidates: list[str]) -> list[float]:
    """Returns zeros — caller falls back to original ordering."""
    return [0.0 for _ in candidates]
