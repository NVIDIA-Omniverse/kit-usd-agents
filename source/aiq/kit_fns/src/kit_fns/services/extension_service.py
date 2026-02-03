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

"""Extension service for managing Kit extension data with FAISS search."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from ..config import KIT_VERSION
from .embedder_service import EmbedderFactory
from .kit_exts_atlas import KitExtensionsAtlasService

logger = logging.getLogger(__name__)

# Try to import FAISS (optional dependency)
# These are optional - semantic search will work if available, otherwise fallback to keyword search
try:
    from langchain_community.vectorstores import FAISS  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Semantic search will be limited.")
    FAISS_AVAILABLE = False

# Get the path to FAISS database
FAISS_DB_PATH = Path(__file__).parent.parent / "data" / KIT_VERSION / "extensions" / "extensions_faiss"


class ExtensionService:
    """Service for managing Kit extension information and operations with FAISS search."""

    def __init__(self, faiss_db_path: str = None):
        """Initialize the Extension service.

        Args:
            faiss_db_path: Path to FAISS database directory
        """
        # Initialize Atlas service for metadata
        self.atlas_service = KitExtensionsAtlasService()

        # Initialize FAISS for semantic search
        self.faiss_db_path = Path(faiss_db_path) if faiss_db_path else FAISS_DB_PATH
        self.vectorstore = None
        self.embedder = None
        self._initialize_faiss()

        # Fallback to Atlas data if FAISS not available
        if not self.vectorstore:
            logger.info("Using Atlas service for extension data (FAISS not available)")

    def _initialize_faiss(self) -> None:
        """Initialize FAISS vector store for semantic search."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, semantic search disabled")
            return

        if not self.faiss_db_path.exists():
            logger.warning(f"FAISS database not found at {self.faiss_db_path}")
            return

        try:
            # Create embedder using factory
            self.embedder = EmbedderFactory.create(model="nvidia/nv-embedqa-e5-v5")

            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                str(self.faiss_db_path), self.embedder, allow_dangerous_deserialization=True
            )

            logger.info(f"Successfully loaded FAISS index from {self.faiss_db_path}")

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.vectorstore = None
            self.embedder = None

    def is_available(self) -> bool:
        """Check if extension data is available."""
        return self.atlas_service.is_available()

    def search_extensions(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search for extensions using semantic search with FAISS.

        Args:
            query: Search query string
            top_k: Number of results to return (default: 20)

        Returns:
            List of extension dictionaries with relevance scores
        """
        if not self.is_available():
            return []

        results = []

        # Use FAISS semantic search if available
        if self.vectorstore:
            try:
                # Perform similarity search
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query, k=top_k * 2
                )  # Get more for potential filtering

                for doc, score in docs_with_scores:
                    metadata = doc.metadata
                    ext_id = metadata.get("extension_id", "")

                    # Get full extension data from Atlas
                    ext_data = self.atlas_service.get_extension_metadata(ext_id)
                    if ext_data:
                        result = {
                            "id": ext_id,
                            "version": metadata.get("version", ext_data.get("version", "")),
                            "description": metadata.get("description", ext_data.get("description", "")),
                            "long_description": metadata.get("long_description", ext_data.get("long_description", "")),
                            "keywords": (
                                metadata.get("keywords", "").split(", ")
                                if metadata.get("keywords")
                                else ext_data.get("keywords", [])
                            ),
                            "features": [],  # Could extract from description
                            "dependencies": (
                                metadata.get("dependencies", "").split(", ")
                                if metadata.get("dependencies")
                                else ext_data.get("dependencies", [])
                            ),
                            "has_python_api": metadata.get("has_python_api", ext_data.get("has_python_api", False)),
                            "has_overview": metadata.get("has_overview", ext_data.get("has_overview", False)),
                            "total_classes": metadata.get("total_classes", ext_data.get("total_classes", 0)),
                            "total_methods": metadata.get("total_methods", ext_data.get("total_methods", 0)),
                            "relevance_score": float(1.0 / (1.0 + score)),  # Convert distance to similarity score
                        }

                        # Extract features from long description
                        if result["long_description"]:
                            # Simple feature extraction from description
                            features = []
                            desc_lower = result["long_description"].lower()
                            if "rendering" in desc_lower:
                                features.append("Rendering capabilities")
                            if "ui" in desc_lower or "interface" in desc_lower:
                                features.append("User interface components")
                            if "physics" in desc_lower:
                                features.append("Physics simulation")
                            if "usd" in desc_lower:
                                features.append("USD integration")
                            if "python" in desc_lower:
                                features.append("Python API support")
                            result["features"] = features[:4]  # Limit features

                        results.append(result)

                        if len(results) >= top_k:
                            break

            except Exception as e:
                logger.error(f"FAISS search failed: {e}")
                # Fall back to keyword search

        # Fallback to keyword-based search if FAISS not available or failed
        if not results:
            query_lower = query.lower()
            query_words = query_lower.split()

            for ext_id in self.atlas_service.get_extension_list():
                ext_data = self.atlas_service.get_extension_metadata(ext_id)
                if not ext_data:
                    continue

                # Simple scoring based on keyword matching
                score = 0.0

                # Check in extension ID
                if any(word in ext_id.lower() for word in query_words):
                    score += 1.5

                # Check in title/description
                title = ext_data.get("title", "").lower()
                description = ext_data.get("description", "").lower()
                long_desc = ext_data.get("long_description", "").lower()

                for word in query_words:
                    if word in ext_id.lower():
                        score += 1.0
                    if word in title:
                        score += 0.8
                    if word in description:
                        score += 0.6
                    if word in long_desc:
                        score += 0.4

                # Check in keywords
                keywords = ext_data.get("keywords", [])
                for keyword in keywords:
                    if any(word in keyword.lower() for word in query_words):
                        score += 0.5

                if score > 0:
                    result = {
                        "id": ext_id,
                        "version": ext_data.get("version", ""),
                        "description": ext_data.get("description", ""),
                        "long_description": ext_data.get("long_description", ""),
                        "keywords": ext_data.get("keywords", []),
                        "features": [],
                        "dependencies": ext_data.get("dependencies", []),
                        "has_python_api": ext_data.get("has_python_api", False),
                        "has_overview": ext_data.get("has_overview", False),
                        "total_classes": ext_data.get("total_classes", 0),
                        "total_methods": ext_data.get("total_methods", 0),
                        "relevance_score": score,
                    }
                    results.append(result)

            # Sort by relevance score and limit
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = results[:top_k]

        return results

    def get_extension_details(self, extension_ids: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information about specific extensions.

        Args:
            extension_ids: List of extension IDs to retrieve

        Returns:
            List of detailed extension information
        """
        if not self.is_available():
            return []

        results = []

        for ext_id in extension_ids:
            ext_data = self.atlas_service.get_extension_metadata(ext_id)

            if ext_data:
                # Build detailed result
                result = {
                    "id": ext_id,
                    "version": ext_data.get("version", ""),
                    "description": ext_data.get("description", ""),
                    "long_description": ext_data.get("long_description", ""),
                    "keywords": ext_data.get("keywords", []),
                    "dependencies": ext_data.get("dependencies", []),
                    "optional_dependencies": ext_data.get(
                        "optional_dependencies", []
                    ),  # Could be extracted if available
                    "has_python_api": ext_data.get("has_python_api", False),
                    "has_overview": ext_data.get("has_overview", False),
                    "total_modules": ext_data.get("total_modules", 0),
                    "total_classes": ext_data.get("total_classes", 0),
                    "total_methods": ext_data.get("total_methods", 0),
                    "codeatlas_token_count": ext_data.get("codeatlas_token_count", 0),
                    "api_docs_token_count": ext_data.get("api_docs_token_count", 0),
                    "overview_token_count": ext_data.get("overview_token_count", 0),
                    "storage_path": ext_data.get("storage_path", ""),
                }

                # Add API symbols if available
                api_symbols = self.atlas_service.get_api_symbols(ext_id)
                if api_symbols:
                    result["apis"] = [s["symbol"] for s in api_symbols[:10]]  # Sample of APIs
                    result["total_apis"] = len(api_symbols)

                # Extract features from description
                features = []
                desc_lower = (result.get("long_description", "") + " " + result.get("description", "")).lower()
                if "rendering" in desc_lower:
                    features.append("Rendering and visualization")
                if "ui" in desc_lower or "interface" in desc_lower or "widget" in desc_lower:
                    features.append("User interface components")
                if "physics" in desc_lower:
                    features.append("Physics simulation")
                if "usd" in desc_lower or "scene" in desc_lower:
                    features.append("USD/Scene management")
                if result["has_python_api"]:
                    features.append("Python API available")
                if result["has_overview"]:
                    features.append("Comprehensive documentation")
                result["features"] = features

                results.append(result)
            else:
                # Try fuzzy matching
                from ..config import DEFAULT_FUZZY_MATCH_THRESHOLD
                from ..utils.fuzzy_matching import find_best_match

                candidates = self.atlas_service.get_extension_list()
                match = find_best_match(ext_id, candidates, threshold=DEFAULT_FUZZY_MATCH_THRESHOLD)

                if match:
                    matched_id, score = match
                    matched_data = self.atlas_service.get_extension_metadata(matched_id)
                    if matched_data:
                        result = {
                            "id": matched_id,
                            "name": matched_data.get("title", matched_id),
                            "version": matched_data.get("version", ""),
                            "description": matched_data.get("description", ""),
                            "matched_query": ext_id,
                            "match_score": score,
                            "note": f"Did you mean '{matched_id}'?",
                        }
                        results.append(result)
                else:
                    results.append(
                        {
                            "error": f"Extension '{ext_id}' not found",
                            "suggestion": "Use search_extensions to find available extensions",
                        }
                    )

        return results

    def get_extension_dependencies(
        self, extension_id: str, depth: int = 2, include_optional: bool = False
    ) -> Dict[str, Any]:
        """Get extension dependency information.

        Args:
            extension_id: Extension ID to analyze
            depth: Dependency tree depth to explore
            include_optional: Include optional dependencies

        Returns:
            Dependency tree information
        """
        if not self.is_available():
            return {"error": "Extension data not available"}

        ext_data = self.atlas_service.get_extension_metadata(extension_id)
        if not ext_data:
            return {"error": f"Extension '{extension_id}' not found"}

        def get_dependencies_recursive(ext_id: str, current_depth: int, visited: set = None) -> Dict[str, Any]:
            if visited is None:
                visited = set()

            if current_depth >= depth or ext_id in visited:
                return {}

            visited.add(ext_id)

            ext = self.atlas_service.get_extension_metadata(ext_id)
            if not ext:
                return {"error": f"Dependency '{ext_id}' not found"}

            deps = {
                "required": ext.get("dependencies", []),
            }

            if include_optional:
                # Include optional dependencies if present in database
                deps["optional"] = ext.get("optional_dependencies", [])

            # Recursively get dependencies
            deps["children"] = {}
            for dep_id in deps["required"]:
                # Clean dependency name (remove version specs if any)
                clean_dep_id = dep_id.split("[")[0].split(">=")[0].split("==")[0].strip()
                if clean_dep_id and clean_dep_id not in visited:
                    deps["children"][clean_dep_id] = get_dependencies_recursive(
                        clean_dep_id, current_depth + 1, visited
                    )

            return deps

        return {
            "extension_id": extension_id,
            "name": ext_data.get("title", extension_id),
            "version": ext_data.get("version", ""),
            "dependencies": get_dependencies_recursive(extension_id, 0),
            "depth": depth,
            "include_optional": include_optional,
        }

    def get_extension_list(self) -> List[str]:
        """Get list of all available extension IDs."""
        if not self.is_available():
            return []
        return self.atlas_service.get_extension_list()
