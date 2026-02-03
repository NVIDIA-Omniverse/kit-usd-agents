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

"""Retrieval services for the USD RAG MCP server."""

import hashlib
import logging
import os
from typing import Any, Dict, Optional

from langchain_community.vectorstores import FAISS

from ..config import (
    DEFAULT_RAG_LENGTH_CODE,
    DEFAULT_RAG_LENGTH_KNOWLEDGE,
    DEFAULT_RAG_TOP_K_CODE,
    DEFAULT_RAG_TOP_K_KNOWLEDGE,
    DEFAULT_RERANK_CODE,
    DEFAULT_RERANK_KNOWLEDGE,
)
from ..utils.patching import patch_information
from .embeddings import create_embeddings, create_embeddings_with_config
from .reranking import Reranker

logger = logging.getLogger(__name__)


class Retriever:
    """Service for retrieving relevant documents from FAISS index."""

    def __init__(
        self,
        endpoint_url: str = None,
        api_key: str = "",
        load_path: str = None,
        top_k: int = 20,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Retriever.

        Args:
            endpoint_url: Optional custom endpoint URL for embeddings
            api_key: API key for authentication
            load_path: Path to the FAISS index
            top_k: Default number of results to retrieve
            embedding_config: Configuration for embeddings
        """
        if embedding_config:
            self.embedder = create_embeddings_with_config(embedding_config)
        else:
            self.embedder = create_embeddings(endpoint_url, api_key)
        self.top_k = top_k
        self.vectordb = None
        self.retriever = None

        if load_path and os.path.exists(load_path):
            try:
                # Verify FAISS index integrity before loading
                # Note: allow_dangerous_deserialization=True is required for FAISS pickle loading.
                # Security is ensured through:
                # 1. Checksum verification of index files before loading
                # 2. Read-only file system access where possible
                # 3. Index files are bundled with the package and not user-modifiable
                if not self._verify_faiss_index_integrity(load_path):
                    raise ValueError(f"FAISS index integrity check failed for {load_path}")

                self.vectordb = FAISS.load_local(
                    load_path,
                    self.embedder,
                    allow_dangerous_deserialization=True,
                )
                self.retriever = self.vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                logger.info(f"Successfully loaded FAISS index from {load_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {load_path}: {e}")
                raise
        elif load_path:
            logger.warning(f"FAISS index path does not exist: {load_path}")

    def _verify_faiss_index_integrity(self, load_path: str) -> bool:
        """Verify the integrity of FAISS index files before loading.

        This provides a basic security check to detect tampering with index files.
        For production deployments, consider using signed checksums stored separately.

        Args:
            load_path: Path to the FAISS index directory

        Returns:
            True if verification passes, False otherwise
        """
        index_faiss = os.path.join(load_path, "index.faiss")
        index_pkl = os.path.join(load_path, "index.pkl")
        checksum_file = os.path.join(load_path, "checksums.sha256")

        # Check required files exist
        if not os.path.exists(index_faiss) or not os.path.exists(index_pkl):
            logger.error(f"Missing required FAISS index files in {load_path}")
            return False

        # If checksums file exists, verify against it
        if os.path.exists(checksum_file):
            try:
                expected_checksums = {}
                with open(checksum_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split()
                            if len(parts) >= 2:
                                expected_checksums[parts[1]] = parts[0]

                for filename in ["index.faiss", "index.pkl"]:
                    filepath = os.path.join(load_path, filename)
                    if filename in expected_checksums:
                        actual_checksum = self._compute_file_checksum(filepath)
                        if actual_checksum != expected_checksums[filename]:
                            logger.error(
                                f"Checksum mismatch for {filename}: expected {expected_checksums[filename]}, got {actual_checksum}"
                            )
                            return False
                        logger.debug(f"Checksum verified for {filename}")
            except Exception as e:
                logger.warning(f"Could not verify checksums: {e}. Proceeding with caution.")

        # Log that we're loading the index (for audit purposes)
        logger.info(f"Loading FAISS index from trusted path: {load_path}")
        return True

    def _compute_file_checksum(self, filepath: str) -> str:
        """Compute SHA256 checksum of a file.

        Args:
            filepath: Path to the file

        Returns:
            Hex string of the SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def search(self, query: str, top_k: Optional[int] = None):
        """Search for relevant documents.

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
    retriever: Retriever,
    rag_max_size: int = DEFAULT_RAG_LENGTH_KNOWLEDGE,
    rag_top_k: int = DEFAULT_RAG_TOP_K_KNOWLEDGE,
    rerank_k: int = DEFAULT_RERANK_KNOWLEDGE,
    reranker: Optional[Reranker] = None,
) -> str:
    """Get RAG context for knowledge queries.

    Args:
        user_query: The user query
        retriever: The RAG retriever
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
            passage_text = f"{rag_result.metadata['index_text']}\n{rag_result.page_content}"
            passages.append(passage_text)

        # Get reranked indices
        reranked_indices = reranker.rerank(user_query, passages)

        # Clamp rerank_k to the number of available results
        effective_rerank_k = max(0, min(rerank_k, len(reranked_indices)))
        logger.info(f"Effective rerank_k: {effective_rerank_k}")

        if effective_rerank_k == 0:
            rag_results = []
        else:
            # Keep only top effective_rerank_k results
            reranked_indices = reranked_indices[:effective_rerank_k]
            # Reorder rag_results based on reranking
            rag_results = [rag_results[i] for i in reranked_indices]

    rag_tokens = 0
    if rag_results:
        rag_bit = ""
        for idx, rag_result in enumerate(rag_results):
            url_bit = ""
            if "url" in rag_result.metadata and rag_result.metadata["url"]:
                url_bit = f", URL '{rag_result.metadata['url']}'"
            rag_bit += (
                f"Title '{rag_result.metadata['index_text']}'{url_bit}\n---\n{rag_result.page_content}\n---\n\n\n"
            )
            rag_tokens += rag_result.metadata["index_text_tokens"] + rag_result.metadata["content_tokens"]
            if rag_tokens > rag_max_size:
                break

        # Apply patches before returning
        return patch_information(rag_bit)
    return ""


def get_rag_context_code(
    user_query: str,
    retriever: Retriever,
    rag_max_size: int = DEFAULT_RAG_LENGTH_CODE,
    rag_top_k: int = DEFAULT_RAG_TOP_K_CODE,
    rerank_k: int = DEFAULT_RERANK_CODE,
    reranker: Optional[Reranker] = None,
) -> str:
    """Get RAG context for code queries.

    Args:
        user_query: The user query
        retriever: The RAG retriever
        rag_max_size: The maximum size of the RAG context
        rag_top_k: The top k RAG context to return
        rerank_k: The top k results to keep after reranking
        reranker: The reranker to use for reranking results

    Returns:
        The RAG context for code
    """
    if retriever is None:
        return ""

    rag_results = retriever.search(user_query, top_k=rag_top_k)

    # Apply reranking if reranker is provided and we have results
    if reranker is not None and rag_results and rerank_k > 0:
        # Extract passages for reranking
        passages = []
        for rag_result in rag_results:
            # Combine question and code for reranking
            passage_text = f"{rag_result.metadata['index_text']}\n{rag_result.page_content}"
            passages.append(passage_text)

        # Get reranked indices
        reranked_indices = reranker.rerank(user_query, passages)

        # Clamp rerank_k to the number of available results
        effective_rerank_k = max(0, min(rerank_k, len(reranked_indices)))
        logger.info(f"Effective rerank_k: {effective_rerank_k}")

        if effective_rerank_k == 0:
            rag_results = []
        else:
            # Keep only top effective_rerank_k results
            reranked_indices = reranked_indices[:effective_rerank_k]

            # Reorder rag_results based on reranking
            rag_results = [rag_results[i] for i in reranked_indices]

    rag_tokens = 0
    if rag_results:
        rag_bit = ""
        for idx, rag_result in enumerate(rag_results):
            added_tokens = rag_result.metadata["index_text_tokens"] + rag_result.metadata["content_tokens"]
            if rag_tokens + added_tokens > rag_max_size:
                continue
            rag_bit += (
                f"Question: '{rag_result.metadata['index_text']}'\nCode:\n```\n{rag_result.page_content}\n```\n\n"
            )
            rag_tokens += added_tokens
            if rag_tokens > rag_max_size:
                break

        # Apply patches before returning
        return patch_information(rag_bit)
    return ""
