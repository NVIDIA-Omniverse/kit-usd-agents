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

"""Function to search for Kit code examples using semantic search."""

import logging
import time
from typing import Any, Dict, Optional

from ..config import DEFAULT_RERANK_CODE
from ..services.code_search_service import CodeSearchService
from ..services.reranking import create_reranker_with_config
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Global code search service instance
_code_search_service = None
_reranker = None


def get_code_search_service() -> CodeSearchService:
    """Get or create the global Code Search service instance.

    Returns:
        The Code Search service instance
    """
    global _code_search_service
    if _code_search_service is None:
        _code_search_service = CodeSearchService()
    return _code_search_service


def _get_or_create_reranker(reranking_config: Optional[Dict[str, Any]] = None):
    """Lazily create and return the reranker."""
    global _reranker

    if _reranker is None or reranking_config:
        _reranker = create_reranker_with_config(reranking_config)
        if _reranker:
            logger.info("Reranker initialized for Kit code examples")

    return _reranker


async def search_code_examples(
    query: str,
    rerank_k: int = DEFAULT_RERANK_CODE,
    enable_rerank: bool = True,
    reranking_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find relevant Kit code examples using semantic search and optional reranking.

    This function performs a RAG (Retrieval-Augmented Generation) query against a curated database
    of Kit code examples. It uses FAISS vector similarity search with NVIDIA embeddings, followed by
    optional reranking for improved relevance.

    How it works:
    1. Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
    2. Performs semantic similarity search against pre-indexed Kit code examples
    3. Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
    4. Returns formatted code examples with their metadata

    Args:
        query: Description of desired code functionality
        rerank_k: Number of documents to keep after reranking (default: DEFAULT_RERANK_CODE)
        enable_rerank: Whether to enable reranking of search results (default: True)
        reranking_config: Optional configuration for reranking service

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
    telemetry_data = {
        "query": query,
        "rerank_k": rerank_k,
        "enable_rerank": enable_rerank,
        "has_reranking_config": reranking_config is not None,
    }

    success = True
    error_msg = None

    try:
        logger.info(f"Searching Kit code examples with query: '{query}'")

        # Validate inputs
        if not query or not query.strip():
            error_msg = "query cannot be empty"
            return {"success": False, "error": error_msg, "result": ""}

        if rerank_k <= 0:
            error_msg = "rerank_k must be positive"
            return {"success": False, "error": error_msg, "result": ""}

        code_search_service = get_code_search_service()

        if not code_search_service.is_available():
            error_msg = "Code search data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        start_time = time.time()

        # Get reranker only if reranking is enabled
        reranker_to_use = _get_or_create_reranker(reranking_config) if enable_rerank else None

        # Perform the search with reranking
        search_results = code_search_service.search_code_examples(
            query.strip(), reranker=reranker_to_use, rerank_k=rerank_k
        )

        if not search_results:
            no_result_msg = f"No code examples found for query: '{query}'"
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

        # Format the results
        result_lines = [f"# Kit Code Example Search Results for: '{query}'"]
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
