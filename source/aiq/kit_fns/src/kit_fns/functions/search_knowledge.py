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

"""Search Kit knowledge function implementation."""

import logging
from typing import Any, Dict, Optional

from ..config import DEFAULT_RERANK_KNOWLEDGE, KNOWLEDGE_INDEX_PATH, get_effective_api_key
from ..services.knowledge_service import KnowledgeRetriever, get_rag_context_knowledge
from ..services.reranking import create_reranker_with_config

logger = logging.getLogger(__name__)

# Global services - will be initialized on first use
_knowledge_retriever: Optional[KnowledgeRetriever] = None
_reranker = None
_retriever_initialized = False


def _initialize_retriever(embedding_config: Optional[Dict[str, Any]] = None):
    """Initialize retriever if not already done."""
    global _knowledge_retriever, _retriever_initialized

    if _retriever_initialized and not embedding_config:
        return

    # Initialize retriever with provided config
    if KNOWLEDGE_INDEX_PATH.exists():
        _knowledge_retriever = KnowledgeRetriever(
            load_path=str(KNOWLEDGE_INDEX_PATH),
            embedding_config=embedding_config,
        )
    else:
        logger.warning(f"FAISS knowledge index not found at {KNOWLEDGE_INDEX_PATH}")

    _retriever_initialized = True


def _get_or_create_reranker(reranking_config: Optional[Dict[str, Any]] = None):
    """Get or create reranker instance."""
    global _reranker

    if _reranker is None:
        _reranker = create_reranker_with_config(reranking_config)

    return _reranker


async def search_kit_knowledge(
    request: str,
    rerank_k: int = DEFAULT_RERANK_KNOWLEDGE,
    enable_rerank: bool = True,
    embedding_config: Optional[Dict[str, Any]] = None,
    reranking_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieves relevant Kit documentation and knowledge using semantic vector search and optional reranking.

    This function performs a RAG (Retrieval-Augmented Generation) query against comprehensive Kit
    documentation. It uses FAISS vector similarity search with NVIDIA embeddings, followed by
    optional reranking for improved relevance.

    How it works:
    1. Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
    2. Performs semantic similarity search against pre-indexed Kit documentation
    3. Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
    4. Returns formatted documentation excerpts with titles and URLs

    Query matching: Your query is compared against the 'index_text' field of each document,
    which contains concise summaries of Kit concepts and documentation.

    Knowledge domains covered:
    - Kit framework architecture and extension system
    - UI development with omni.ui
    - USD integration and scene manipulation
    - Testing frameworks and patterns
    - Application lifecycle and settings

    Tips for better results:
    - Use specific Kit concepts (e.g., "extension lifecycle", "viewport rendering")
    - Include relevant domains (e.g., "UI", "settings", "testing")
    - Ask about workflows, patterns, or implementation details
    - Use Kit module names when relevant (e.g., "omni.ui", "omni.kit")

    Args:
        request: Your query about Kit concepts, workflows, or documentation.
                Examples: "How does extension lifecycle work?", "Kit viewport rendering", "omni.ui styling"
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

        if not KNOWLEDGE_INDEX_PATH.exists():
            error_msg = f"FAISS knowledge index not found at path: {KNOWLEDGE_INDEX_PATH}. Please configure the path."
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
            rerank_k=rerank_k,
        )

        if rag_context:
            logger.info(
                f"Retrieved knowledge context for '{request}' with reranking: {'enabled' if enable_rerank else 'disabled'}"
            )
            return {"success": True, "result": rag_context, "error": None}
        else:
            no_result_msg = "No relevant Kit knowledge found for your request."
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving Kit knowledge: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "result": ""}
