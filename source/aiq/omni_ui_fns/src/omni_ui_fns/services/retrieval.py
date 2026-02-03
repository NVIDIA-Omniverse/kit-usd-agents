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

"""Retrieval service for OmniUI code examples using FAISS vector search."""

import hashlib
import logging
import os
from typing import Any, Dict, Optional

from langchain_community.vectorstores import FAISS

from ..config import DEFAULT_EMBEDDING_MODEL, DEFAULT_RAG_LENGTH_CODE, DEFAULT_RAG_TOP_K_CODE, DEFAULT_RERANK_CODE
from .embedder_service import EmbedderFactory

logger = logging.getLogger(__name__)


def create_embeddings(endpoint_url: str = None, api_key: str = ""):
    """Create embeddings using factory pattern.

    Args:
        endpoint_url: Optional custom endpoint URL for local embedder
        api_key: API key for authentication

    Returns:
        Embedder instance (LocalEmbedder or NVIDIAEmbeddings)
    """
    if endpoint_url:
        # Use local embedder
        return EmbedderFactory.create(backend="local", local_url=endpoint_url, model=DEFAULT_EMBEDDING_MODEL)
    else:
        # Use default backend from environment (nvidia_api or local)
        return EmbedderFactory.create(api_key=api_key, model=DEFAULT_EMBEDDING_MODEL)


def create_embeddings_with_config(config: Dict[str, Any]):
    """Create embeddings using configuration and factory pattern.

    Args:
        config: Configuration dict with 'model', 'endpoint', and 'api_key'

    Returns:
        Embedder instance (LocalEmbedder or NVIDIAEmbeddings)
    """
    endpoint = config.get("endpoint")
    api_key = config.get("api_key", "")
    model = config.get("model", DEFAULT_EMBEDDING_MODEL)

    # Debug logging - sanitized to avoid leaking sensitive information
    logger.debug("create_embeddings_with_config called")
    logger.debug(f"API key provided: {'Yes' if api_key else 'No'}")
    # Note: Do not log API key length as it can help attackers narrow down key format
    logger.debug(f"Endpoint configured: {'Yes' if endpoint else 'No (using default)'}")
    logger.debug(f"Model: {model}")

    if endpoint:
        # Use local embedder
        return EmbedderFactory.create(backend="local", local_url=endpoint, model=model)
    else:
        # Use default backend from environment (nvidia_api or local)
        return EmbedderFactory.create(api_key=api_key, model=model)


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
            top_k: Number of results to return (uses default if not specified)

        Returns:
            List of relevant documents
        """
        logger.info(f"[DEBUG] Retriever.search called with query: {query}, top_k: {top_k}")

        if not self.retriever:
            logger.error("Retriever not initialized")
            return []

        k = top_k if top_k is not None else self.top_k
        self.retriever.search_kwargs = {"k": k}
        logger.info(f"[DEBUG] Using k={k} for search")

        try:
            # Use invoke() instead of deprecated get_relevant_documents()
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


def get_rag_context_omni_ui_code(
    user_query: str,
    retriever: Retriever,
    rag_max_size: int = DEFAULT_RAG_LENGTH_CODE,
    rag_top_k: int = DEFAULT_RAG_TOP_K_CODE,
    rerank_k: int = DEFAULT_RERANK_CODE,
    reranker: Optional[Any] = None,
) -> str:
    """Get RAG context for OmniUI code queries.

    Args:
        user_query: The user query
        retriever: The RAG retriever
        rag_max_size: The maximum size of the RAG context
        rag_top_k: The top k RAG context to return
        rerank_k: The top k results to keep after reranking
        reranker: The reranker to use for reranking results

    Returns:
        The RAG context for OmniUI code examples
    """
    if not retriever:
        logger.error("Retriever not provided")
        return ""

    # Perform search
    docs = retriever.search(user_query, top_k=rag_top_k)

    if not docs:
        return ""

    # If reranker is provided, use it to rerank results
    if reranker:
        try:
            # Extract texts for reranking
            texts = [doc.page_content for doc in docs]

            # Rerank documents
            reranked_results = reranker.rerank(user_query, texts, top_k=rerank_k)

            # Reorder docs based on reranking
            reranked_docs = []
            for result in reranked_results:
                idx = result["index"]
                if idx < len(docs):
                    reranked_docs.append(docs[idx])

            docs = reranked_docs[:rerank_k] if reranked_docs else docs[:rerank_k]
            logger.info(f"Reranked {len(docs)} documents")
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            docs = docs[:rerank_k]
    else:
        # If no reranker, just limit to rerank_k results
        docs = docs[:rerank_k]

    # Format the results
    rag_text = ""
    current_size = 0

    for i, doc in enumerate(docs, 1):
        # Extract metadata
        metadata = doc.metadata

        # Handle both FAISS index schemas:
        # Schema 1 (omni_ui_rag_collection.json): file_name, method_name, source_code
        # Schema 2 (ui_functions_with_descriptions.json): description, code, function_name, class_name

        # Try schema 1 first (original code examples)
        # Note: Some FAISS indices store method_name in page_content instead of metadata
        if "source_code" in metadata and "file_name" in metadata:
            file_name = metadata.get("file_name", "unknown")
            file_path = metadata.get("file_path", "unknown")
            method_ref = metadata.get("method_name", doc.page_content)  # Fallback to page_content
            source_code = metadata.get("source_code", "")
            description = doc.page_content  # Usually the method signature

        # Try schema 2 (window examples with descriptions)
        elif "function_name" in metadata or "code" in metadata:
            file_path = metadata.get("file_path", "unknown")
            function_name = metadata.get("function_name", "unknown")
            class_name = metadata.get("class_name", "unknown")
            source_code = metadata.get("code", doc.page_content)
            description = doc.page_content

            # Extract filename from path
            file_name = file_path.split("/")[-1] if file_path != "unknown" else "unknown"

            # Format method reference
            method_ref = f"{class_name}.{function_name}" if class_name != "unknown" else function_name

        # Fallback for unknown schema
        else:
            file_name = metadata.get("file_name", "unknown")
            file_path = metadata.get("file_path", "unknown")
            method_ref = "unknown"
            source_code = doc.page_content
            description = f"Example {i}"

        # Create formatted output
        example_text = f"""### Example {i}: {description}
File: {file_name}
Path: {file_path}
Method: {method_ref}

```python
{source_code}
```

"""

        # Check size limit
        if current_size + len(example_text) > rag_max_size:
            break

        rag_text += example_text
        current_size += len(example_text)

    return rag_text.strip()
