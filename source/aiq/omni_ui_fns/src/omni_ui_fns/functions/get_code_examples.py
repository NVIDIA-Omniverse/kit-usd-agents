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

"""Get code examples function implementation."""

import logging
import time
from typing import Any, Dict, Optional

from ..config import DEFAULT_RERANK_CODE, FAISS_CODE_INDEX_PATH
from ..services.reranking import create_reranker_with_config
from ..services.retrieval import Retriever, get_rag_context_omni_ui_code
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Global services - will be initialized on first use
_code_retriever: Optional[Retriever] = None
_reranker = None
_retriever_initialized = False


def _initialize_retriever(embedding_config: Optional[Dict[str, Any]] = None):
    """Initialize retriever if not already done."""
    global _code_retriever, _retriever_initialized

    if _retriever_initialized and not embedding_config:
        return

    # Initialize retriever with provided config
    if FAISS_CODE_INDEX_PATH.exists():
        _code_retriever = Retriever(embedding_config=embedding_config, load_path=str(FAISS_CODE_INDEX_PATH))
    else:
        logger.warning(f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}")

    _retriever_initialized = True


def _get_or_create_reranker(reranking_config: Optional[Dict[str, Any]] = None):
    """Lazily create and return the reranker."""
    global _reranker

    if _reranker is None or reranking_config:
        _reranker = create_reranker_with_config(reranking_config)
        if _reranker:
            logger.info("Reranker initialized for OmniUI code examples")

    return _reranker


async def get_code_examples(
    request: str,
    rerank_k: int = DEFAULT_RERANK_CODE,
    enable_rerank: bool = True,
    embedding_config: Optional[Dict[str, Any]] = None,
    reranking_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieves relevant OmniUI code examples using semantic vector search and optional reranking.

    This function performs a RAG (Retrieval-Augmented Generation) query against a curated database
    of OmniUI code examples. It uses FAISS vector similarity search with NVIDIA embeddings, followed by
    optional reranking for improved relevance.

    How it works:
    1. Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
    2. Performs semantic similarity search against pre-indexed OmniUI code examples
    3. Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
    4. Returns formatted code examples with their metadata

    Query matching: Your query is compared against OmniUI widget and component implementations,
    including examples like:
    - SearchField, SearchWordButton implementations
    - Widget styling and theming functions
    - UI component building patterns
    - Event handling and callback patterns

    Tips for better results:
    - Use specific OmniUI terminology (e.g., "SearchField", "ZStack", "VStack")
    - Frame queries as component or pattern names when possible
    - Include relevant UI operations (e.g., "build_ui", "style", "event handling")
    - Be specific about the UI component or pattern you want to implement

    Args:
        request: Your query describing the desired OmniUI code example.
                Examples: "How to create a search field?", "Button styling", "event handling"
        rerank_k: Number of documents to keep after reranking (default: DEFAULT_RERANK_CODE)
        enable_rerank: Whether to enable reranking of search results (default: True)
        embedding_config: Optional configuration for embedding service
        reranking_config: Optional configuration for reranking service

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: Formatted code examples with file paths and Python code snippets, or error message
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "request": request,
        "rerank_k": rerank_k,
        "enable_rerank": enable_rerank,
        "has_embedding_config": embedding_config is not None,
        "has_reranking_config": reranking_config is not None,
    }

    success = True
    error_msg = None

    try:
        # Debug logging
        logger.info(f"[DEBUG] get_code_examples called with request: {request}")
        logger.info(f"[DEBUG] embedding_config: {embedding_config}")
        logger.info(f"[DEBUG] reranking_config: {reranking_config}")
        logger.info(f"[DEBUG] FAISS_CODE_INDEX_PATH: {FAISS_CODE_INDEX_PATH}")
        logger.info(f"[DEBUG] FAISS_CODE_INDEX_PATH exists: {FAISS_CODE_INDEX_PATH.exists()}")

        # Initialize retriever if needed
        _initialize_retriever(embedding_config)

        if not FAISS_CODE_INDEX_PATH.exists():
            error_msg = f"FAISS index not found at path: {FAISS_CODE_INDEX_PATH}. Please configure the path."
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}

        if _code_retriever is None:
            error_msg = "Code retriever could not be initialized. Please check the configuration."
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}

        # Get reranker only if reranking is enabled
        reranker_to_use = _get_or_create_reranker(reranking_config) if enable_rerank else None

        # Get the RAG context using the utility function with reranking
        rag_context = get_rag_context_omni_ui_code(
            user_query=request,
            retriever=_code_retriever,
            reranker=reranker_to_use,
            rerank_k=rerank_k,  # Pass the rerank_k parameter
        )

        if rag_context:
            logger.info(
                f"Retrieved OmniUI code context for '{request}' with reranking: {'enabled' if enable_rerank else 'disabled'}"
            )
            return {"success": True, "result": rag_context, "error": None}
        else:
            no_result_msg = "No relevant OmniUI code examples found for your request."
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving OmniUI code examples: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_code_examples",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
