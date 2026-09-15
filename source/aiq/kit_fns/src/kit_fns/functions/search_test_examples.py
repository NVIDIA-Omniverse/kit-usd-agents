# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Function to search for Kit test examples using semantic search."""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_RERANK_CODE
from ..services.code_search_service import CodeSearchService
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Shares KIT_CODE_SEARCH_TIMEOUT_SEC with search_code_examples — the underlying
# CodeSearchService is the same singleton.
_DEFAULT_CODE_SEARCH_TIMEOUT_SEC = 120.0


def _code_search_timeout_sec() -> float:
    raw = os.environ.get("KIT_CODE_SEARCH_TIMEOUT_SEC", "").strip()
    if not raw:
        return _DEFAULT_CODE_SEARCH_TIMEOUT_SEC
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_CODE_SEARCH_TIMEOUT_SEC
    except ValueError:
        return _DEFAULT_CODE_SEARCH_TIMEOUT_SEC


def get_code_search_service() -> CodeSearchService:
    from .search_code_examples import get_code_search_service as _get_code_search_service

    return _get_code_search_service()


def _blocking_search(query: str, rerank_k: int) -> Dict[str, Any]:
    try:
        service = get_code_search_service()
        if not service.is_available():
            return {"outcome": "unavailable", "error": "Test search data is not available"}
        results = service.search_test_examples(query, rerank_k=rerank_k)
        return {"outcome": "ok", "results": results}
    except Exception as e:
        logger.exception("Test example search failed in worker thread")
        return {"outcome": "error", "error": str(e)}


async def search_test_examples(
    query: str,
    rerank_k: int = DEFAULT_RERANK_CODE,
    enable_rerank: bool = True,
    reranking_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find test implementations and patterns using semantic search and optional reranking.

    Reranking is controlled by env vars on the hybrid retriever (``OVAI_RERANK``
    / ``OVAI_RERANK_BACKEND``, with legacy ``KIT_RERANKER_BACKEND`` honored as a
    fallback). ``enable_rerank`` / ``reranking_config`` here are kept for
    backward compat but no longer flip rerank per-call; explicit
    ``enable_rerank=False`` logs a warning so the no-op isn't silent.
    """
    await ensure_telemetry_initialized()
    start_time = time.perf_counter()
    if not enable_rerank:
        logger.warning(
            "search_test_examples received enable_rerank=False; this kwarg is no "
            "longer wired to the hybrid retriever. Set OVAI_RERANK=false (and "
            "ensure no legacy *_RERANKER_BACKEND env var is set) to force-off."
        )
    telemetry_data = {
        "query": query,
        "rerank_k": rerank_k,
        "enable_rerank": enable_rerank,
        "has_reranking_config": reranking_config is not None,
    }
    success = True
    error_msg = None

    try:
        logger.info(f"Searching Kit test examples with query: '{query}'")

        if not query or not query.strip():
            error_msg = "query cannot be empty"
            return {"success": False, "error": error_msg, "result": ""}

        if rerank_k <= 0:
            error_msg = "rerank_k must be positive"
            return {"success": False, "error": error_msg, "result": ""}

        timeout_sec = _code_search_timeout_sec()
        try:
            block_result: Dict[str, Any] = await asyncio.wait_for(
                asyncio.to_thread(_blocking_search, query.strip(), rerank_k),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            error_msg = (
                f"Test example search timed out after {timeout_sec:.0f}s. "
                "Check NVIDIA_API_KEY, network, KIT_EMBEDDER_BACKEND, KIT_RERANKER_BACKEND, "
                "and KIT_CODE_SEARCH_TIMEOUT_SEC."
            )
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}

        outcome = block_result.get("outcome")
        if outcome == "unavailable":
            error_msg = str(block_result.get("error", "Test search data is not available"))
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}
        if outcome == "error":
            error_msg = f"Error searching test examples: {block_result.get('error', 'unknown')}"
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}

        search_results: List[Dict[str, Any]] = block_result.get("results") or []

        if not search_results:
            no_result_msg = f"No test examples found for query: '{query}'"
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

        result_lines = [f"# Kit Test Example Search Results for: '{query}'"]
        result_lines.append(f"\n**Found {len(search_results)} relevant test examples:**\n")
        for i, example in enumerate(search_results, 1):
            result_lines.append(f"## Test Example {i}: {example.get('title', 'Untitled')}")
            result_lines.append(f"**File:** `{example.get('file_path', 'unknown')}`")
            result_lines.append(f"**Extension:** `{example.get('extension_id', 'unknown')}`")
            result_lines.append(f"**Lines:** {example.get('line_start', 0)}-{example.get('line_end', 0)}")
            result_lines.append(f"**Relevance Score:** {example.get('relevance_score', 0):.2f}")
            result_lines.append(f"\n**Test Description:**")
            result_lines.append(f"{example.get('description', 'No description available')}")
            result_lines.append(f"\n**Test Code:**")
            result_lines.append(f"```python\n{example.get('code', 'No code available')}\n```")
            tags = example.get("tags", [])
            if tags:
                result_lines.append(f"\n**Test Categories:** {', '.join(tags)}")
            result_lines.append("\n---\n")
        result_lines.append("**Testing Tips:**")
        result_lines.append("- Use `omni.kit.test.AsyncTestCase` for asynchronous tests")
        result_lines.append("- Implement `setUp()` and `tearDown()` for test isolation")
        result_lines.append("- Use `get_instructions(instruction_sets='testing')` for comprehensive testing guidance")

        formatted_result = "\n".join(result_lines)
        logger.info(f"Successfully found {len(search_results)} test examples for query: '{query}'")
        return {"success": True, "result": formatted_result, "error": None}

    except Exception as e:
        error_msg = f"Error searching test examples: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        await telemetry.capture_call(
            function_name="search_test_examples",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
