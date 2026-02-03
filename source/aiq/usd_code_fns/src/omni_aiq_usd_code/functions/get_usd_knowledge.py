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

"""Get USD knowledge function implementation."""

import logging
from typing import Any, Dict, Optional

from ..config import DEFAULT_RERANK_KNOWLEDGE, FAISS_KNOWLEDGE_INDEX_PATH, get_effective_api_key
from ..services.reranking import create_reranker_with_config
from ..services.retrieval import Retriever, get_rag_context_knowledge

logger = logging.getLogger(__name__)

# Global services - will be initialized on first use
_knowledge_retriever: Optional[Retriever] = None
_reranker = None
_retriever_initialized = False


def _initialize_retriever(embedding_config: Optional[Dict[str, Any]] = None):
    """Initialize retriever if not already done."""
    global _knowledge_retriever, _retriever_initialized

    if _retriever_initialized and not embedding_config:
        return

    # Initialize retriever with provided config
    if FAISS_KNOWLEDGE_INDEX_PATH.exists():
        _knowledge_retriever = Retriever(embedding_config=embedding_config, load_path=str(FAISS_KNOWLEDGE_INDEX_PATH))
    else:
        logger.warning(f"FAISS knowledge index not found at {FAISS_KNOWLEDGE_INDEX_PATH}")

    _retriever_initialized = True


def _get_or_create_reranker(reranking_config: Optional[Dict[str, Any]] = None):
    """Lazily create and return the reranker."""
    global _reranker

    if _reranker is None or reranking_config:
        _reranker = create_reranker_with_config(reranking_config)
        if _reranker:
            logger.info("Reranker initialized for knowledge")

    return _reranker


async def get_usd_knowledge(
    request: str,
    rerank_k: int = DEFAULT_RERANK_KNOWLEDGE,
    enable_rerank: bool = True,
    embedding_config: Optional[Dict[str, Any]] = None,
    reranking_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieves relevant USD documentation and knowledge using semantic vector search and optional reranking.

    This function performs a RAG (Retrieval-Augmented Generation) query against comprehensive USD
    documentation. It uses FAISS vector similarity search with NVIDIA embeddings, followed by
    optional reranking for improved relevance.

    How it works:
    1. Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
    2. Performs semantic similarity search against pre-indexed USD documentation
    3. Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
    4. Returns formatted documentation excerpts with titles and URLs

    Query matching: Your query is compared against the 'index_text' field of each document,
    which contains concise summaries like:
    - "UsdLuxGeometryLight lacks detailed specifications for consistent behavior across renderers"
    - "Different renderers have varying approaches to mesh lights"
    - "The design should cater to specific workflow requirements of geometry lights"

    Knowledge domains covered:
    - USD concepts and architecture (layers, prims, stages, composition)
    - Rendering and lighting (UsdLux, geometry lights, materials)
    - Animation and skeletal systems (UsdSkel)
    - Geometry and scenes (UsdGeom)
    - Workflows and best practices

    Tips for better results:
    - Use specific USD concepts (e.g., "layer composition", "prim inheritance")
    - Include relevant domains (e.g., "lighting", "animation", "geometry")
    - Ask about workflows, specifications, or behavioral details
    - Use USD schema names when relevant (e.g., "UsdLux", "UsdGeom", "UsdSkel")

    Args:
        request: Your query about USD concepts, workflows, or documentation.
                Examples: "What is layer composition?", "USD lighting workflow", "prim inheritance"
        rerank_k: Number of documents to keep after reranking (default: DEFAULT_RERANK_KNOWLEDGE)
        enable_rerank: Whether to enable reranking of search results (default: True)

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: Formatted documentation excerpts with titles, content, and source URLs, or error message
        - error: Error message if operation failed
    """
    try:
        # Initialize retriever if needed
        _initialize_retriever(embedding_config)

        if not FAISS_KNOWLEDGE_INDEX_PATH.exists():
            error_msg = (
                f"FAISS knowledge index not found at path: {FAISS_KNOWLEDGE_INDEX_PATH}. Please configure the path."
            )
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        if _knowledge_retriever is None:
            error_msg = "Knowledge retriever could not be initialized. Please check the configuration."
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Get reranker only if reranking is enabled
        reranker_to_use = _get_or_create_reranker(reranking_config) if enable_rerank else None

        # Get the RAG context using the utility function with reranking
        rag_context = get_rag_context_knowledge(
            user_query=request,
            retriever=_knowledge_retriever,
            reranker=reranker_to_use,
            rerank_k=rerank_k,  # Pass the rerank_k parameter
        )

        if rag_context:
            logger.info(
                f"Retrieved knowledge context for '{request}' with reranking: {'enabled' if enable_rerank else 'disabled'}"
            )
            return {"success": True, "result": rag_context, "error": None}
        else:
            no_result_msg = "No relevant USD knowledge found for your request."
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving USD knowledge: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "result": ""}
