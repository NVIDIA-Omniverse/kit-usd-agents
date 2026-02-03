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

"""Knowledge retrieval service for Kit documentation using FAISS and reranking."""

import logging
from typing import Any, Dict, List, Optional

from ..config import (
    DEFAULT_RAG_LENGTH_KNOWLEDGE,
    DEFAULT_RAG_TOP_K_KNOWLEDGE,
    DEFAULT_RERANK_KNOWLEDGE,
    KNOWLEDGE_INDEX_PATH,
)
from .embedder_service import EmbedderFactory
from .reranking import Reranker

logger = logging.getLogger(__name__)

# Try to import FAISS (optional dependency)
try:
    from langchain_community.vectorstores import FAISS

    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Semantic search will be limited.")
    FAISS_AVAILABLE = False


class KnowledgeRetriever:
    """Service for retrieving Kit documentation knowledge using FAISS."""

    def __init__(
        self,
        load_path: str = None,
        top_k: int = DEFAULT_RAG_TOP_K_KNOWLEDGE,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Knowledge Retriever.

        Args:
            load_path: Path to the FAISS index
            top_k: Default number of results to retrieve
            embedding_config: Configuration for embeddings
        """
        self.top_k = top_k
        self.vectordb = None
        self.retriever = None

        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, knowledge search disabled")
            return

        # Create embedder using factory
        if embedding_config:
            self.embedder = EmbedderFactory.create(
                api_key=embedding_config.get("api_key"),
                model=embedding_config.get("model", "nvidia/nv-embedqa-e5-v5"),
            )
        else:
            self.embedder = EmbedderFactory.create(model="nvidia/nv-embedqa-e5-v5")

        if load_path:
            try:
                self.vectordb = FAISS.load_local(
                    load_path,
                    self.embedder,
                    allow_dangerous_deserialization=True,
                )
                self.retriever = self.vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                logger.info(f"Successfully loaded FAISS knowledge index from {load_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS knowledge index from {load_path}: {e}")
                raise

    def search(self, query: str, top_k: Optional[int] = None) -> List[Any]:
        """Search for relevant knowledge documents.

        Args:
            query: The search query
            top_k: Number of results to return (overrides default)

        Returns:
            List of relevant documents
        """
        if not self.retriever:
            logger.warning("Retriever not initialized - no FAISS index loaded")
            return []

        if top_k is not None and self.top_k != top_k:
            retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            return retriever.invoke(query)

        return self.retriever.invoke(query)


def get_rag_context_knowledge(
    user_query: str,
    retriever: KnowledgeRetriever,
    rag_max_size: int = DEFAULT_RAG_LENGTH_KNOWLEDGE,
    rag_top_k: int = DEFAULT_RAG_TOP_K_KNOWLEDGE,
    rerank_k: int = DEFAULT_RERANK_KNOWLEDGE,
    reranker: Optional[Reranker] = None,
) -> str:
    """Get RAG context for knowledge queries.

    Args:
        user_query: The user query
        retriever: The knowledge retriever
        rag_max_size: The maximum size of the RAG context
        rag_top_k: The top k RAG context to return
        rerank_k: The top k results to keep after reranking
        reranker: The reranker to use for reranking results

    Returns:
        The RAG context for knowledge
    """
    if retriever is None:
        return ""

    rag_results = retriever.search(user_query, top_k=rag_top_k)

    # Apply reranking if reranker is provided and we have results
    if reranker is not None and rag_results and rerank_k > 0:
        # Extract passages for reranking
        passages = []
        for rag_result in rag_results:
            # Combine title and content for reranking
            index_text = rag_result.metadata.get("index_text", "")
            passage_text = f"{index_text}\n{rag_result.page_content}"
            passages.append(passage_text)

        # Get reranked indices
        reranked_results = reranker.rerank(user_query, passages, top_k=rerank_k)

        # Clamp rerank_k to the number of available results
        effective_rerank_k = max(0, min(rerank_k, len(reranked_results)))
        logger.info(f"Effective rerank_k: {effective_rerank_k}")

        if effective_rerank_k == 0:
            rag_results = []
        else:
            # Keep only top effective_rerank_k results
            # Reorder rag_results based on reranking
            reranked_indices = [r["index"] for r in reranked_results[:effective_rerank_k]]
            rag_results = [rag_results[i] for i in reranked_indices if i < len(rag_results)]

    rag_tokens = 0
    if rag_results:
        rag_bit = ""
        for idx, rag_result in enumerate(rag_results):
            url_bit = ""
            if "url" in rag_result.metadata and rag_result.metadata["url"]:
                url_bit = f", URL '{rag_result.metadata['url']}'"

            index_text = rag_result.metadata.get("index_text", "Unknown")
            rag_bit += f"Title '{index_text}'{url_bit}\n---\n{rag_result.page_content}\n---\n\n\n"

            # Track token usage (approximate)
            index_text_tokens = rag_result.metadata.get("index_text_tokens", len(index_text.split()))
            content_tokens = rag_result.metadata.get("content_tokens", len(rag_result.page_content.split()))
            rag_tokens += index_text_tokens + content_tokens

            if rag_tokens > rag_max_size:
                break

        return rag_bit
    return ""
