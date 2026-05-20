# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Function to search for Isaac Sim code examples using semantic search."""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List

from ..services.code_search_service import CodeSearchService
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Default cap for blocking work: first call builds ``CodeSearchService`` (FAISS load,
# embedder init, optional large JSON fallback). Search then calls the embedding API.
# All of that must run off the asyncio event loop with a wall-clock bound.
_DEFAULT_CODE_SEARCH_TIMEOUT_SEC = 120.0

# Global code search service instance
_code_search_service = None


def _code_search_timeout_sec() -> float:
    raw = os.environ.get("ISAACSIM_CODE_SEARCH_TIMEOUT_SEC", "").strip()
    if not raw:
        return _DEFAULT_CODE_SEARCH_TIMEOUT_SEC
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_CODE_SEARCH_TIMEOUT_SEC
    except ValueError:
        return _DEFAULT_CODE_SEARCH_TIMEOUT_SEC


def get_code_search_service() -> CodeSearchService:
    """Get or create the global Code Search service instance.

    Returns:
        The Code Search service instance
    """
    global _code_search_service
    if _code_search_service is None:
        _code_search_service = CodeSearchService()
    return _code_search_service


def _blocking_code_example_search(query: str, top_k: int) -> Dict[str, Any]:
    """Run service construction (first call), availability check, and FAISS search in a worker thread.

    Must not be called from the asyncio event loop thread — it performs heavy synchronous I/O and HTTP.
    """
    try:
        service = get_code_search_service()
        if not service.is_available():
            return {"outcome": "unavailable", "error": "Code search data is not available"}
        results = service.search_code_examples(query, top_k)
        return {"outcome": "ok", "results": results}
    except Exception as e:
        logger.exception("Code example search failed in worker thread")
        return {"outcome": "error", "error": str(e)}


async def search_code_examples(query: str, top_k: int = 20) -> Dict[str, Any]:
    """Find relevant Isaac Sim code examples using semantic search.

    Args:
        query: Description of desired code functionality
        top_k: Number of results to return (default: 20)

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: Formatted code examples with file paths and implementation details
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {"query": query, "top_k": top_k}

    success = True
    error_msg = None

    try:
        logger.info(f"Searching Isaac Sim code examples with query: '{query}'")

        # Validate inputs
        if not query or not query.strip():
            error_msg = "query cannot be empty"
            return {"success": False, "error": error_msg, "result": ""}

        if top_k <= 0:
            error_msg = "top_k must be positive"
            return {"success": False, "error": error_msg, "result": ""}

        timeout_sec = _code_search_timeout_sec()
        try:
            block_result: Dict[str, Any] = await asyncio.wait_for(
                asyncio.to_thread(_blocking_code_example_search, query.strip(), top_k),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            error_msg = (
                f"Code example search timed out after {timeout_sec:.0f}s "
                "(includes first-time FAISS/embedder init and embedding HTTP). "
                "Check NVIDIA_API_KEY, network, KIT_EMBEDDER_BACKEND=local + KIT_LOCAL_EMBEDDER_URL, "
                "NVIDIA_EMBEDDING_REQUEST_TIMEOUT_SEC, and ISAACSIM_CODE_SEARCH_TIMEOUT_SEC."
            )
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}

        outcome = block_result.get("outcome")
        if outcome == "unavailable":
            error_msg = str(block_result.get("error", "Code search data is not available"))
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}
        if outcome == "error":
            error_msg = f"Error searching code examples: {block_result.get('error', 'unknown')}"
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}

        search_results: List[Dict[str, Any]] = block_result.get("results") or []

        if not search_results:
            no_result_msg = f"No code examples found for query: '{query}'"
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

        # Format the results
        result_lines = [f"# Isaac Sim Code Example Search Results for: '{query}'"]
        result_lines.append(f"\n**Found {len(search_results)} relevant examples:**\n")

        for i, example in enumerate(search_results, 1):
            result_lines.append(f"## Example {i}: {example.get('title', 'Untitled')}")
            result_lines.append(f"**File:** `{example.get('file_path', 'unknown')}`")
            result_lines.append(f"**Extension:** `{example.get('extension_id', 'unknown')}`")
            result_lines.append(f"**Lines:** {example.get('line_start', 0)}-{example.get('line_end', 0)}")
            result_lines.append(f"**Relevance Score:** {example.get('relevance_score', 0):.2f}")
            result_lines.append(f"\n**Description:**")
            result_lines.append(f"{example.get('description', 'No description available')}")
            result_lines.append(f"\n**Code:**")
            result_lines.append(f"```python\n{example.get('code', 'No code available')}\n```")

            # Show tags
            tags = example.get("tags", [])
            if tags:
                result_lines.append(f"\n**Tags:** {', '.join(tags)}")

            result_lines.append("\n---\n")

        # Add usage tip
        result_lines.append("*Use get_api_details for complete API documentation of specific classes/methods.*")

        formatted_result = "\n".join(result_lines)

        logger.info(f"Successfully found {len(search_results)} code examples for query: '{query}'")
        return {"success": True, "result": formatted_result, "error": None}

    except Exception as e:
        error_msg = f"Error searching code examples: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="search_code_examples",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
