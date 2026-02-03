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

"""Code search service for finding Kit code examples and test patterns using FAISS."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_RAG_TOP_K_CODE, DEFAULT_RERANK_CODE, KIT_VERSION
from .embedder_service import EmbedderFactory

logger = logging.getLogger(__name__)

# Try to import FAISS (optional dependency)
try:
    from langchain_community.vectorstores import FAISS  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Semantic search will be limited.")
    FAISS_AVAILABLE = False

# Get the base path for data
DATA_BASE = Path(__file__).parent.parent / "data"
CODE_EXAMPLES_DATA_BASE = DATA_BASE / KIT_VERSION / "code_examples"

# Paths for mode-based FAISS databases
CODE_EXAMPLES_FAISS_PATHS = {
    "regular": CODE_EXAMPLES_DATA_BASE / "code_examples_faiss",
    "tests": CODE_EXAMPLES_DATA_BASE / "code_examples_faiss",  # Using regular for now
    "all": CODE_EXAMPLES_DATA_BASE / "code_examples_faiss",  # Using regular for now
}

# Paths for mode-based extracted methods (fallback)
EXTRACTED_METHODS_PATHS = {
    "regular": CODE_EXAMPLES_DATA_BASE / "extracted_methods_regular",
    "tests": CODE_EXAMPLES_DATA_BASE / "extracted_methods_regular",  # Using regular for now
    "all": CODE_EXAMPLES_DATA_BASE / "extracted_methods_regular",  # Using regular for now
}

# Legacy paths for backward compatibility
LEGACY_CODE_EXAMPLES_FAISS_PATH = CODE_EXAMPLES_DATA_BASE / "code_examples_faiss"
LEGACY_EXTRACTED_METHODS_PATH = CODE_EXAMPLES_DATA_BASE / "extracted_methods_regular"


class CodeSearchService:
    """Service for searching Kit code examples and test patterns using FAISS."""

    def __init__(self):
        """Initialize the Code Search service."""
        # Initialize vector stores
        self.code_vectorstore = None  # For regular/production code
        self.test_vectorstore = None  # For test code
        self.embedder = None

        # Initialize fallback data
        self.code_examples_data = None
        self.test_examples_data = None

        # Initialize FAISS and fallback data
        self._initialize_faiss()
        self._load_fallback_data()

    def _initialize_faiss(self) -> None:
        """Initialize FAISS vector stores for semantic search."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, semantic search disabled")
            return

        try:
            # Create embedder using factory
            self.embedder = EmbedderFactory.create(model="nvidia/nv-embedqa-e5-v5")

            # Try to load regular code FAISS index (priority order: regular, all, legacy)
            code_faiss_loaded = False
            for mode in ["regular", "all"]:
                faiss_path = CODE_EXAMPLES_FAISS_PATHS.get(mode)
                if faiss_path and faiss_path.exists():
                    try:
                        self.code_vectorstore = FAISS.load_local(
                            str(faiss_path), self.embedder, allow_dangerous_deserialization=True
                        )
                        logger.info(f"Successfully loaded code examples FAISS index from {faiss_path} (mode: {mode})")
                        code_faiss_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to load code examples FAISS index from {faiss_path}: {e}")

            # Try legacy path if no mode-based database found
            if not code_faiss_loaded and LEGACY_CODE_EXAMPLES_FAISS_PATH.exists():
                try:
                    self.code_vectorstore = FAISS.load_local(
                        str(LEGACY_CODE_EXAMPLES_FAISS_PATH), self.embedder, allow_dangerous_deserialization=True
                    )
                    logger.info(
                        f"Successfully loaded code examples FAISS index from legacy path: {LEGACY_CODE_EXAMPLES_FAISS_PATH}"
                    )
                    code_faiss_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load legacy code examples FAISS index: {e}")

            if not code_faiss_loaded:
                logger.warning("No code examples FAISS database found")

            # Try to load test examples FAISS index (priority order: tests, all, use code vectorstore)
            test_faiss_loaded = False
            for mode in ["tests", "all"]:
                faiss_path = CODE_EXAMPLES_FAISS_PATHS.get(mode)
                if faiss_path and faiss_path.exists():
                    try:
                        self.test_vectorstore = FAISS.load_local(
                            str(faiss_path), self.embedder, allow_dangerous_deserialization=True
                        )
                        logger.info(f"Successfully loaded test examples FAISS index from {faiss_path} (mode: {mode})")
                        test_faiss_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to load test examples FAISS index from {faiss_path}: {e}")

            # If no separate test database, use code vectorstore for tests as well
            if not test_faiss_loaded:
                if self.code_vectorstore:
                    self.test_vectorstore = self.code_vectorstore
                    logger.info("Using code examples FAISS database for test searches")
                else:
                    logger.warning("No test examples FAISS database found")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.code_vectorstore = None
            self.test_vectorstore = None
            self.embedder = None

    def _load_fallback_data(self) -> None:
        """Load fallback data from extracted methods JSON files."""
        # Load code examples from extracted methods
        self.code_examples_data = []
        self.test_examples_data = []

        # Try to load from mode-based directories first
        loaded_from_mode_dirs = False

        # Try regular methods directory
        regular_dir = EXTRACTED_METHODS_PATHS.get("regular")
        if regular_dir and regular_dir.exists():
            try:
                for json_file in regular_dir.glob("*.example.json"):
                    with open(json_file, "r") as f:
                        data = json.load(f)

                    extension_name = data.get("extension_name", json_file.stem.replace(".example", ""))
                    methods = data.get("methods", [])

                    for method in methods:
                        example = self._create_example_from_method(extension_name, method)
                        self.code_examples_data.append(example)

                loaded_from_mode_dirs = True
                logger.info(f"Loaded {len(self.code_examples_data)} code examples from {regular_dir}")
            except Exception as e:
                logger.error(f"Failed to load regular methods: {e}")

        # Try test methods directory
        tests_dir = EXTRACTED_METHODS_PATHS.get("tests")
        if tests_dir and tests_dir.exists():
            try:
                for json_file in tests_dir.glob("*.example.json"):
                    with open(json_file, "r") as f:
                        data = json.load(f)

                    extension_name = data.get("extension_name", json_file.stem.replace(".example", ""))
                    methods = data.get("methods", [])

                    for method in methods:
                        example = self._create_example_from_method(extension_name, method)
                        self.test_examples_data.append(example)

                loaded_from_mode_dirs = True
                logger.info(f"Loaded {len(self.test_examples_data)} test examples from {tests_dir}")
            except Exception as e:
                logger.error(f"Failed to load test methods: {e}")

        # If mode-based directories not found, try "all" directory
        if not loaded_from_mode_dirs:
            all_dir = EXTRACTED_METHODS_PATHS.get("all")
            if all_dir and all_dir.exists():
                try:
                    self._load_from_directory(all_dir)
                    loaded_from_mode_dirs = True
                    logger.info(f"Loaded examples from 'all' directory: {all_dir}")
                except Exception as e:
                    logger.error(f"Failed to load from 'all' directory: {e}")

        # Fallback to legacy directory if mode-based not found
        if not loaded_from_mode_dirs and LEGACY_EXTRACTED_METHODS_PATH.exists():
            try:
                self._load_from_directory(LEGACY_EXTRACTED_METHODS_PATH)
                logger.info(f"Loaded examples from legacy directory: {LEGACY_EXTRACTED_METHODS_PATH}")
            except Exception as e:
                logger.error(f"Failed to load from legacy directory: {e}")

    def _load_from_directory(self, directory: Path) -> None:
        """Load examples from a directory, separating into code and test examples."""
        for json_file in directory.glob("*.example.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                extension_name = data.get("extension_name", json_file.stem.replace(".example", ""))
                methods = data.get("methods", [])

                for method in methods:
                    # Determine if it's a test method
                    is_test = self._is_test_method(method)

                    # Create example entry
                    example = self._create_example_from_method(extension_name, method)

                    if is_test:
                        self.test_examples_data.append(example)
                    else:
                        self.code_examples_data.append(example)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        logger.info(
            f"Loaded {len(self.code_examples_data)} code and {len(self.test_examples_data)} test examples from {directory}"
        )

    def _is_test_method(self, method: Dict[str, Any]) -> bool:
        """Check if a method is a test method."""
        method_name = method.get("name", "").lower()
        module_name = method.get("module", "").lower()
        file_path = method.get("file_path", "").lower()

        return "test" in method_name or "test" in module_name or "test" in file_path or file_path.endswith("test.py")

    def _create_example_from_method(self, extension_name: str, method: Dict[str, Any]) -> Dict[str, Any]:
        """Create an example entry from a method."""
        return {
            "id": f"{extension_name}::{method.get('name', 'unknown')}",
            "title": method.get("name", "Unknown Method"),
            "description": self._generate_description(method),
            "file_path": method.get("file_path", ""),
            "extension_id": extension_name,
            "line_start": method.get("start_line", 0),
            "line_end": method.get("start_line", 0) + method.get("line_count", 0),
            "code": method.get("source_code", ""),
            "tags": self._extract_tags(method),
            "relevance_keywords": method.get("reasons", []),
        }

    def _generate_description(self, method: Dict[str, Any]) -> str:
        """Generate a description for a method based on its metadata."""
        parts = []

        if method.get("is_async"):
            parts.append("Async method")

        if method.get("complexity_score", 0) > 5:
            parts.append(f"Complex implementation (score: {method.get('complexity_score')})")

        if method.get("line_count", 0) > 50:
            parts.append(f"Large method ({method.get('line_count')} lines)")

        reasons = method.get("reasons", [])
        if reasons:
            parts.append(f"Features: {', '.join(reasons[:3])}")

        return ". ".join(parts) if parts else "Kit method implementation"

    def _extract_tags(self, method: Dict[str, Any]) -> List[str]:
        """Extract tags from method metadata."""
        tags = []

        # Add tags based on method characteristics
        if method.get("is_async"):
            tags.append("async")

        if method.get("decorators"):
            tags.append("decorated")
            for decorator in method.get("decorators", [])[:2]:
                tags.append(decorator.replace("@", ""))

        # Add tags from reasons
        reasons = method.get("reasons", [])
        for reason in reasons:
            if "recursion" in reason.lower():
                tags.append("recursion")
            if "error" in reason.lower():
                tags.append("error-handling")
            if "complex" in reason.lower():
                tags.append("complex")
            if "large" in reason.lower():
                tags.append("large")

        # Add module-based tags
        module = method.get("module", "")
        if "ui" in module.lower():
            tags.append("ui")
        if "usd" in module.lower():
            tags.append("usd")
        if "scene" in module.lower():
            tags.append("scene")
        if "test" in module.lower():
            tags.append("test")

        return list(set(tags))  # Remove duplicates

    def is_available(self) -> bool:
        """Check if code search data is available."""
        return (
            self.code_vectorstore is not None
            or (self.code_examples_data is not None and len(self.code_examples_data) > 0)
            or (self.test_examples_data is not None and len(self.test_examples_data) > 0)
        )

    def search_code_examples(
        self, query: str, top_k: int = DEFAULT_RAG_TOP_K_CODE, reranker=None, rerank_k: int = DEFAULT_RERANK_CODE
    ) -> List[Dict[str, Any]]:
        """Search for code examples using semantic search with FAISS and optional reranking.

        Args:
            query: Search query describing desired functionality
            top_k: Number of results to return (default: DEFAULT_RAG_TOP_K_CODE)
            reranker: Optional reranker instance for improving result relevance
            rerank_k: Number of results to keep after reranking (default: DEFAULT_RERANK_CODE)

        Returns:
            List of relevant code examples with scores
        """
        if not self.is_available():
            return []

        results = []

        # Use FAISS semantic search if available
        if self.code_vectorstore:
            try:
                # Perform similarity search
                docs_with_scores = self.code_vectorstore.similarity_search_with_score(query, k=top_k)

                # Collect all candidate results first
                candidate_results = []
                for doc, score in docs_with_scores:
                    metadata = doc.metadata

                    # Skip test methods when searching for code examples
                    if self._is_test_from_metadata(metadata):
                        continue

                    # Extract source code from metadata if available
                    source_code = metadata.get("source_code", "")
                    if not source_code:
                        # Try to extract from page content
                        content = doc.page_content
                        if "Code preview:" in content:
                            source_code = content.split("Code preview:")[-1].strip()

                    result = {
                        "id": metadata.get("method_key", ""),
                        "title": metadata.get("name", "Unknown Method"),
                        "description": self._create_description_from_metadata(metadata),
                        "file_path": metadata.get("file_path", ""),
                        "extension_id": metadata.get("extension", ""),
                        "line_start": metadata.get("line_count", 0),
                        "line_end": metadata.get("line_count", 0) + 50,  # Approximate
                        "code": source_code,
                        "tags": self._extract_tags_from_metadata(metadata),
                        "relevance_score": float(1.0 / (1.0 + score)),  # Convert distance to similarity
                        "page_content": doc.page_content,
                    }

                    candidate_results.append(result)

                # Apply reranking if available
                if reranker and candidate_results:
                    try:
                        # Extract texts for reranking
                        texts = [result["page_content"] for result in candidate_results]

                        # Rerank documents
                        reranked_results = reranker.rerank(query, texts, top_k=rerank_k)

                        # Reorder results based on reranking
                        reranked_candidate_indices = []
                        for rank_result in reranked_results:
                            idx = rank_result["index"]
                            if idx < len(candidate_results):
                                reranked_candidate_indices.append(idx)

                        # Use reranked results if we have them, otherwise fall back to original
                        results = (
                            list(map(lambda idx: candidate_results[idx], reranked_candidate_indices[:rerank_k]))
                            if reranked_candidate_indices
                            else candidate_results[:rerank_k]
                        )
                    except Exception as e:
                        logger.warning(f"Reranking failed: {e}, using original results")
                        results = candidate_results[:rerank_k]
                else:
                    # No reranking, use original results
                    results = candidate_results[:rerank_k]

            except Exception as e:
                logger.error(f"FAISS search failed: {e}")
                # Fall back to keyword search

        # Fallback to keyword-based search if FAISS not available or failed
        if not results and self.code_examples_data:
            results = self._keyword_search(query, self.code_examples_data, top_k)

        return results

    def search_test_examples(
        self, query: str, top_k: int = DEFAULT_RAG_TOP_K_CODE, reranker=None, rerank_k: int = DEFAULT_RERANK_CODE
    ) -> List[Dict[str, Any]]:
        """Search for test examples using semantic search with FAISS and optional reranking.

        Args:
            query: Search query describing test scenario
            top_k: Number of results to return (default: DEFAULT_RAG_TOP_K_CODE)
            reranker: Optional reranker instance for improving result relevance
            rerank_k: Number of results to keep after reranking (default: DEFAULT_RERANK_CODE)

        Returns:
            List of relevant test examples with scores
        """
        if not self.is_available():
            return []

        results = []

        # Use FAISS semantic search if available
        if self.test_vectorstore:
            try:
                # Perform similarity search
                docs_with_scores = self.test_vectorstore.similarity_search_with_score(query, k=top_k)

                # Collect all candidate results first
                candidate_results = []
                for doc, score in docs_with_scores:
                    metadata = doc.metadata

                    # If we have a dedicated test database, include all results
                    # If using shared database, filter for test methods
                    if self.test_vectorstore != self.code_vectorstore:
                        # Dedicated test database - include all
                        pass
                    else:
                        # Shared database - filter for tests
                        if not self._is_test_from_metadata(metadata):
                            continue

                    # Extract source code
                    source_code = metadata.get("source_code", "")
                    if not source_code:
                        content = doc.page_content
                        if "Code preview:" in content:
                            source_code = content.split("Code preview:")[-1].strip()

                    result = {
                        "id": metadata.get("method_key", ""),
                        "title": metadata.get("name", "Unknown Test"),
                        "description": self._create_description_from_metadata(metadata),
                        "file_path": metadata.get("file_path", ""),
                        "extension_id": metadata.get("extension", ""),
                        "line_start": metadata.get("line_count", 0),
                        "line_end": metadata.get("line_count", 0) + 50,
                        "code": source_code,
                        "tags": self._extract_tags_from_metadata(metadata),
                        "relevance_score": float(1.0 / (1.0 + score)),
                        "page_content": doc.page_content,
                    }

                    candidate_results.append(result)

                # Apply reranking if available
                if reranker and candidate_results:
                    try:
                        # Extract texts for reranking
                        texts = [result["page_content"] for result in candidate_results]

                        # Rerank documents
                        reranked_results = reranker.rerank(query, texts, top_k=rerank_k)

                        # Reorder results based on reranking
                        reranked_candidate_indices = []
                        for rank_result in reranked_results:
                            idx = rank_result["index"]
                            if idx < len(candidate_results):
                                reranked_candidate_indices.append(idx)

                        # Use reranked results if we have them, otherwise fall back to original
                        results = (
                            list(map(lambda idx: candidate_results[idx], reranked_candidate_indices[:rerank_k]))
                            if reranked_candidate_indices
                            else candidate_results[:rerank_k]
                        )
                    except Exception as e:
                        logger.warning(f"Reranking failed: {e}, using original results")
                        results = candidate_results[:rerank_k]
                else:
                    # No reranking, use original results
                    results = candidate_results[:rerank_k]

            except Exception as e:
                logger.error(f"FAISS test search failed: {e}")

        # Fallback to test examples data
        if not results and self.test_examples_data:
            results = self._keyword_search(query, self.test_examples_data, rerank_k)

        return results

    def _keyword_search(self, query: str, examples_data: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search on examples data."""
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        for example in examples_data:
            score = 0.0

            # Check in title and description
            if any(word in example.get("title", "").lower() for word in query_words):
                score += 1.0
            if any(word in example.get("description", "").lower() for word in query_words):
                score += 0.9

            # Check in relevance keywords
            for keyword in example.get("relevance_keywords", []):
                if any(word in keyword.lower() for word in query_words):
                    score += 0.8

            # Check in tags
            for tag in example.get("tags", []):
                if any(word in tag.lower() for word in query_words):
                    score += 0.6

            # Check in code content
            if any(word in example.get("code", "").lower() for word in query_words):
                score += 0.4

            if score > 0:
                result = example.copy()
                result["relevance_score"] = score
                results.append(result)

        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def _is_test_from_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata indicates a test method."""
        method_name = metadata.get("name", "").lower()
        module_name = metadata.get("module", "").lower()
        file_path = metadata.get("file_path", "").lower()

        return "test" in method_name or "test" in module_name or "test" in file_path or file_path.endswith("test.py")

    def _create_description_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Create a description from FAISS metadata."""
        parts = []

        if metadata.get("is_async"):
            parts.append("Async method")

        if metadata.get("complexity_score", 0) > 5:
            parts.append(f"Complex implementation (score: {metadata.get('complexity_score')})")

        if metadata.get("line_count", 0) > 50:
            parts.append(f"Large method ({metadata.get('line_count')} lines)")

        if metadata.get("reasons"):
            parts.append(metadata["reasons"])

        return ". ".join(parts) if parts else "Kit method implementation"

    def _extract_tags_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract tags from FAISS metadata."""
        tags = []

        # Add characteristic tags
        if metadata.get("is_async"):
            tags.append("async")
        if metadata.get("is_large"):
            tags.append("large")
        if metadata.get("is_complex"):
            tags.append("complex")
        if metadata.get("has_recursion"):
            tags.append("recursion")
        if metadata.get("has_error_handling"):
            tags.append("error-handling")

        # Add module-based tags
        module = metadata.get("module", "")
        if "ui" in module.lower():
            tags.append("ui")
        if "usd" in module.lower():
            tags.append("usd")
        if "scene" in module.lower():
            tags.append("scene")
        if "test" in module.lower():
            tags.append("test")

        return list(set(tags))
