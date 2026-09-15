# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Get USD code example function implementation."""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from ..config import DEFAULT_RERANK_CODE, FAISS_CODE_INDEX_PATH, get_effective_api_key
from ..services.retrieval import Retriever, get_rag_context_code

logger = logging.getLogger(__name__)

_DEFAULT_CODE_SEARCH_TIMEOUT_SEC = 120.0

_code_retriever: Optional[Retriever] = None
_retriever_initialized = False


def _code_search_timeout_sec() -> float:
    raw = os.environ.get("USD_CODE_SEARCH_TIMEOUT_SEC", "").strip()
    if not raw:
        return _DEFAULT_CODE_SEARCH_TIMEOUT_SEC
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_CODE_SEARCH_TIMEOUT_SEC
    except ValueError:
        return _DEFAULT_CODE_SEARCH_TIMEOUT_SEC


def _initialize_retriever(embedding_config: Optional[Dict[str, Any]] = None):
    global _code_retriever, _retriever_initialized
    if _retriever_initialized and not embedding_config:
        return
    if FAISS_CODE_INDEX_PATH.exists():
        _code_retriever = Retriever(embedding_config=embedding_config, load_path=str(FAISS_CODE_INDEX_PATH))
    else:
        logger.warning(f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}")
    _retriever_initialized = True


def _blocking_search(
    request: str,
    rerank_k: int,
    embedding_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """FAISS code-index retrieval + embedder HTTP. Reranking handled via OVAI_RERANK*."""
    try:
        _initialize_retriever(embedding_config)
        if not FAISS_CODE_INDEX_PATH.exists():
            return {
                "outcome": "unavailable",
                "error": f"FAISS index not found at path: {FAISS_CODE_INDEX_PATH}. Please configure the path.",
            }
        if _code_retriever is None:
            return {
                "outcome": "unavailable",
                "error": "Code retriever could not be initialized. Please check the configuration.",
            }
        rag_context = get_rag_context_code(
            user_query=request,
            retriever=_code_retriever,
            rerank_k=rerank_k,
        )
        return {"outcome": "ok", "result": rag_context}
    except Exception as e:
        logger.exception("USD code example search failed in worker thread")
        return {"outcome": "error", "error": str(e)}


async def get_usd_code_example(
    request: str,
    rerank_k: int = DEFAULT_RERANK_CODE,
    enable_rerank: bool = True,
    embedding_config: Optional[Dict[str, Any]] = None,
    reranking_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieves relevant USD code examples using semantic vector search and optional reranking.

    Reranking is now controlled by env vars on the hybrid retriever
    (``OVAI_RERANK`` / ``OVAI_RERANK_BACKEND``, with legacy
    ``USD_CODE_RERANKER_BACKEND`` / ``KIT_RERANKER_BACKEND`` honored as a
    fallback). ``enable_rerank`` / ``reranking_config`` here are kept for
    backward compat but no longer flip rerank per-call; explicit
    ``enable_rerank=False`` logs a warning so the no-op isn't silent.
    """
    if not enable_rerank:
        logger.warning(
            "get_usd_code_example received enable_rerank=False; this kwarg is no "
            "longer wired to the hybrid retriever. Set OVAI_RERANK=false (and "
            "ensure no legacy *_RERANKER_BACKEND env var is set) to force-off."
        )
    try:
        timeout_sec = _code_search_timeout_sec()
        try:
            block_result: Dict[str, Any] = await asyncio.wait_for(
                asyncio.to_thread(_blocking_search, request, rerank_k, embedding_config),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            error_msg = (
                f"USD code example search timed out after {timeout_sec:.0f}s. "
                "Check NVIDIA_API_KEY, network, KIT_EMBEDDER_BACKEND, KIT_RERANKER_BACKEND, "
                "and USD_CODE_SEARCH_TIMEOUT_SEC."
            )
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        outcome = block_result.get("outcome")
        if outcome == "unavailable":
            error_msg = str(block_result.get("error", "Code search not available"))
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}
        if outcome == "error":
            error_msg = f"Error retrieving code examples: {block_result.get('error', 'unknown')}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        rag_context = block_result.get("result")
        if rag_context:
            logger.info(
                f"Retrieved code context for '{request}' with reranking: "
                f"{'enabled' if enable_rerank else 'disabled'}"
            )
            return {"success": True, "result": rag_context, "error": None}
        no_result_msg = "No relevant code examples found for your request."
        logger.info(no_result_msg)
        return {"success": True, "result": no_result_msg, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving code examples: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "result": ""}
