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

"""Retrieval service for UI window examples using FAISS vector search."""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .retrieval import create_embeddings, create_embeddings_with_config

logger = logging.getLogger(__name__)


class UIWindowExamplesRetriever:
    """Service for retrieving UI window examples from FAISS index."""

    def __init__(
        self,
        endpoint_url: str = None,
        api_key: str = "",
        faiss_index_path: str = None,
        top_k: int = 10,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the UI Window Examples Retriever.

        Args:
            endpoint_url: Optional custom endpoint URL for embeddings
            api_key: API key for authentication
            faiss_index_path: Path to the FAISS index
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

        if faiss_index_path and os.path.exists(faiss_index_path):
            try:
                self.vectordb = FAISS.load_local(
                    faiss_index_path,
                    self.embedder,
                    allow_dangerous_deserialization=True,
                )
                self.retriever = self.vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                logger.info(f"Successfully loaded UI window examples FAISS index from {faiss_index_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {faiss_index_path}: {e}")
                raise
        elif faiss_index_path:
            logger.warning(f"FAISS index path does not exist: {faiss_index_path}")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Search for relevant UI window examples.

        Args:
            query: The search query
            top_k: Number of results to return (uses default if not specified)

        Returns:
            List of relevant documents
        """
        logger.info(f"[DEBUG] UIWindowExamplesRetriever.search called with query: {query}, top_k: {top_k}")

        if not self.retriever:
            logger.error("UI Window Examples Retriever not initialized")
            return []

        k = top_k if top_k is not None else self.top_k
        self.retriever.search_kwargs = {"k": k}
        logger.info(f"[DEBUG] Using k={k} for UI window examples search")

        try:
            # Use invoke() instead of deprecated get_relevant_documents()
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"UI window examples search failed: {e}")
            return []

    def get_structured_results(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get structured results for UI window examples search.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of structured results with description, code, and file_path
        """
        docs = self.search(query, top_k)

        structured_results = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata

            result = {
                "rank": i,
                "description": metadata.get("description", ""),
                "code": metadata.get("code", ""),
                "file_path": metadata.get("file_path", ""),
                "function_name": metadata.get("function_name", "unknown"),
                "class_name": metadata.get("class_name", "unknown"),
                "line_number": metadata.get("line_number", 0),
                "similarity_score": getattr(doc, "similarity_score", None),  # If available from search
            }

            structured_results.append(result)

        return structured_results

    def get_formatted_results(
        self, query: str, top_k: Optional[int] = None, max_description_length: int = 200, max_code_length: int = 1000
    ) -> str:
        """Get formatted string results for UI window examples search.

        Args:
            query: The search query
            top_k: Number of results to return
            max_description_length: Maximum length for description preview
            max_code_length: Maximum length for code preview

        Returns:
            Formatted string with search results
        """
        results = self.get_structured_results(query, top_k)

        if not results:
            return "No UI window examples found for your query."

        formatted_output = f"**UI Window Examples Search Results for:** '{query}'\n"
        formatted_output += f"**Found {len(results)} matches:**\n\n"

        for result in results:
            description = result["description"]
            code = result["code"]

            # Truncate long descriptions and code
            if len(description) > max_description_length:
                description = description[:max_description_length] + "..."

            if len(code) > max_code_length:
                code = code[:max_code_length] + "\n... [truncated]"

            formatted_output += f"### Match {result['rank']}\n"
            formatted_output += f"**File:** `{result['file_path']}`\n"
            formatted_output += (
                f"**Function:** `{result['class_name']}.{result['function_name']}()` (Line {result['line_number']})\n\n"
            )
            formatted_output += f"**Description:**\n{description}\n\n"
            formatted_output += f"**Code:**\n```python\n{code}\n```\n\n"
            formatted_output += "---\n\n"

        return formatted_output.strip()


def get_ui_window_examples(
    user_query: str,
    retriever: UIWindowExamplesRetriever,
    top_k: int = 5,
    format_type: str = "structured",  # "structured", "formatted", "raw"
) -> Any:
    """Get UI window examples for a user query.

    Args:
        user_query: The user query
        retriever: The UI window examples retriever
        top_k: Number of results to return
        format_type: Type of formatting for results ("structured", "formatted", "raw")

    Returns:
        UI window examples in requested format
    """
    if not retriever:
        logger.error("UI Window Examples Retriever not provided")
        return [] if format_type == "structured" else ""

    try:
        if format_type == "structured":
            return retriever.get_structured_results(user_query, top_k)
        elif format_type == "formatted":
            return retriever.get_formatted_results(user_query, top_k)
        elif format_type == "raw":
            return retriever.search(user_query, top_k)
        else:
            logger.warning(f"Unknown format_type: {format_type}, defaulting to structured")
            return retriever.get_structured_results(user_query, top_k)

    except Exception as e:
        logger.error(f"Failed to get UI window examples: {e}")
        return [] if format_type in ["structured", "raw"] else "Error retrieving UI window examples."


def create_ui_window_examples_retriever(
    faiss_index_path: str = None, api_key: str = "", top_k: int = 10
) -> UIWindowExamplesRetriever:
    """Create a UI Window Examples Retriever with default paths.

    Args:
        faiss_index_path: Path to FAISS index (uses default if None)
        api_key: API key for embeddings
        top_k: Default number of results

    Returns:
        Configured UIWindowExamplesRetriever instance
    """
    if faiss_index_path is None:
        # Default path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
        faiss_index_path = os.path.join(data_dir, "ui_window_examples_faiss")

    return UIWindowExamplesRetriever(api_key=api_key, faiss_index_path=faiss_index_path, top_k=top_k)
