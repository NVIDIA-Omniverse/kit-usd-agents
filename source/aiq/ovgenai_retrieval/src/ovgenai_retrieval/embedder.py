"""Embedder factory — pluggable backends for hosted NIM + local sentence-transformers.

Ports the pattern used by kit-usd-agents's ``embedder_service.py`` so both sides
of the retrieval stack use identical embedding calls. Callers should normally
rely on the manifest to know what model was used to build the index and pass
that model name here for query embedding.

Two backends are supported:

1. ``backend="nvidia"`` (default) — wraps
   :class:`langchain_nvidia_ai_endpoints.NVIDIAEmbeddings`. By default this
   calls the hosted ``integrate.api.nvidia.com`` endpoint; pass ``base_url=``
   to point at an on-prem NIM deployment (``api_key`` is then optional for
   auth-less endpoints).

2. ``backend="sentence_transformers"`` — wraps a local
   :class:`sentence_transformers.SentenceTransformer` model. ``api_key`` and
   ``base_url`` are ignored because nothing is called over the wire. Requires
   the ``[rerank]`` (or ``[all]``) extra to be installed.

Example — local qwen3-embedding-4b::

    from ovgenai_retrieval.embedder import EmbedderFactory

    emb = EmbedderFactory.create(
        model="Qwen/Qwen3-Embedding-4B",
        backend="sentence_transformers",
    )
    vec = emb.embed_query("how do I declare an extension dependency?")

Example — on-prem NIM endpoint::

    emb = EmbedderFactory.create(
        model="nvidia/nemotron-3-embed-1b",
        base_url="http://nim.internal:8000/v1",
        api_key=None,  # auth-less on-prem deployment
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from ovgenai_retrieval.models import DEFAULT_EMBEDDING_MODEL


class Embedder(Protocol):
    """Minimal interface we rely on."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformersEmbedder:
    """Local embedder wrapping :class:`sentence_transformers.SentenceTransformer`.

    Provides the same ``embed_query`` / ``embed_documents`` surface the rest of
    the library expects, so it's a drop-in alternative to the hosted NVIDIA
    embedder for air-gapped deployments or experimenting with newer open
    models (e.g., Qwen3-Embedding-4B, nomic-embed-text-v2).
    """

    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover — exercised only when extra missing
            raise RuntimeError(
                "backend='sentence_transformers' requires the optional dependency. "
                "Install with `pip install ovgenai-retrieval[rerank]` "
                "(or `[all]`)."
            ) from e
        self._model_name = model
        self._impl = SentenceTransformer(model)

    def embed_query(self, text: str) -> list[float]:
        vec = self._impl.encode(text)
        # SentenceTransformer.encode returns a 1-D numpy array for a single input.
        return [float(x) for x in vec.tolist()]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        mat = self._impl.encode(list(texts))
        return [[float(x) for x in row] for row in mat.tolist()]


@dataclass
class EmbedderFactory:
    """Factory for pluggable embedders (NVIDIA hosted/NIM + sentence-transformers)."""

    @staticmethod
    def create(
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        backend: str = "nvidia",
    ) -> Embedder:
        """Return an Embedder for the requested backend.

        Args:
            model: Model identifier. For ``backend="nvidia"`` this is the
                NVIDIA model string (for example,
                ``"nvidia/nemotron-3-embed-1b"``); for
                ``backend="sentence_transformers"`` this is a HuggingFace repo
                ID or local path understood by
                :class:`sentence_transformers.SentenceTransformer`.
            api_key: NVIDIA API key. Falls back to ``NVIDIA_API_KEY`` env var.
                Ignored when ``backend="sentence_transformers"``. Optional when
                ``backend="nvidia"`` and ``base_url`` targets an auth-less NIM
                endpoint.
            base_url: Optional override for the NVIDIA endpoint (enables
                on-prem NIM). Forwarded to ``NVIDIAEmbeddings``. Ignored when
                ``backend="sentence_transformers"``.
            backend: ``"nvidia"`` (default, hosted/NIM) or
                ``"sentence_transformers"`` (local).
        """
        if backend == "sentence_transformers":
            return SentenceTransformersEmbedder(model)

        if backend != "nvidia":
            raise ValueError(f"Unknown embedder backend: {backend!r}. " "Expected 'nvidia' or 'sentence_transformers'.")

        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        key = api_key or os.environ.get("NVIDIA_API_KEY")
        # When base_url is set (on-prem NIM) the endpoint may be auth-less, so
        # a missing key is acceptable. For the hosted endpoint we still require
        # an explicit key — failing fast is better than hitting a 401 mid-query.
        if not key and not base_url:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Export it or pass api_key= to "
                "EmbedderFactory.create() (or pass base_url= for an auth-less "
                "on-prem NIM endpoint)."
            )
        kwargs: dict[str, Any] = {"model": model}
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        return NVIDIAEmbeddings(**kwargs)
