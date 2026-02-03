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

"""Settings service for managing Kit settings data with FAISS search."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import KIT_VERSION
from .embedder_service import EmbedderFactory

logger = logging.getLogger(__name__)

# Try to import FAISS (optional dependency)
try:
    from langchain_community.vectorstores import FAISS  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Settings search will be limited.")
    FAISS_AVAILABLE = False

SETTINGS_DATA_BASE = Path(__file__).parent.parent / "data" / KIT_VERSION / "settings"
# Get the path to settings FAISS database
SETTINGS_FAISS_DB_PATH = SETTINGS_DATA_BASE / "settings_faiss"
SETTINGS_SUMMARY_PATH = SETTINGS_DATA_BASE / "setting_summary.json"


class SettingsService:
    """Service for managing Kit settings information with FAISS search."""

    def __init__(self, faiss_db_path: str = None, settings_summary_path: str = None):
        """Initialize the Settings service.

        Args:
            faiss_db_path: Path to FAISS database directory
            settings_summary_path: Path to settings summary JSON file
        """
        # Initialize paths
        self.faiss_db_path = Path(faiss_db_path) if faiss_db_path else SETTINGS_FAISS_DB_PATH
        self.settings_summary_path = Path(settings_summary_path) if settings_summary_path else SETTINGS_SUMMARY_PATH

        # Initialize components
        self.vectorstore = None
        self.embedder = None
        self.settings_data = None

        # Load settings summary
        self._load_settings_summary()

        # Initialize FAISS for semantic search
        self._initialize_faiss()

        # Fallback message if FAISS not available
        if not self.vectorstore:
            logger.info("Using keyword search for settings (FAISS not available)")

    def _load_settings_summary(self) -> None:
        """Load the settings summary data."""
        if self.settings_summary_path.exists():
            try:
                with open(self.settings_summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.settings_data = data.get("settings", {})
                    self.metadata = data.get("metadata", {})
                    logger.info(f"Loaded {len(self.settings_data)} settings from summary")
            except Exception as e:
                logger.error(f"Failed to load settings summary: {e}")
                self.settings_data = {}
                self.metadata = {}
        else:
            logger.warning(f"Settings summary not found at {self.settings_summary_path}")
            self.settings_data = {}
            self.metadata = {}

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

            logger.info(f"Successfully loaded FAISS index for settings from {self.faiss_db_path}")

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.vectorstore = None
            self.embedder = None

    def is_available(self) -> bool:
        """Check if settings data is available."""
        return bool(self.settings_data) or self.vectorstore is not None

    def search_settings(
        self, query: str, top_k: int = 20, prefix_filter: Optional[str] = None, type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for settings using semantic search with FAISS.

        Args:
            query: Search query string
            top_k: Number of results to return (default: 20)
            prefix_filter: Filter by setting prefix (exts, app, persistent, rtx)
            type_filter: Filter by setting type (bool, int, float, string, array, object)

        Returns:
            List of setting dictionaries with relevance scores
        """
        if not self.is_available():
            return []

        results = []

        # Build filter dict for FAISS
        filters = {}
        if prefix_filter:
            filters["prefix"] = prefix_filter
        if type_filter:
            filters["type"] = type_filter

        # Use FAISS semantic search if available
        if self.vectorstore:
            try:
                # Perform similarity search with filters
                if filters:
                    docs_with_scores = self.vectorstore.similarity_search_with_score(
                        query, k=top_k * 2, filter=filters  # Get more for potential filtering
                    )
                else:
                    docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)

                for doc, score in docs_with_scores:
                    metadata = doc.metadata
                    setting_key = metadata.get("setting_key", "")

                    # Get full setting data from summary
                    setting_data = self.settings_data.get(setting_key, {})

                    result = {
                        "setting": setting_key,
                        "type": metadata.get("type", setting_data.get("type", "unknown")),
                        "default_value": metadata.get("default_value", setting_data.get("default_value")),
                        "documentation": metadata.get("documentation", setting_data.get("documentation", "")),
                        "description": metadata.get("description", setting_data.get("description", "")),
                        "extensions": list(setting_data.get("extensions", [])),
                        "usage_count": metadata.get("usage_count", setting_data.get("usage_count", 0)),
                        "found_in": setting_data.get("found_in", [])[:3],  # Limit to first 3 locations
                        "has_documentation": metadata.get("has_documentation", bool(setting_data.get("documentation"))),
                        "prefix": metadata.get("prefix", self._extract_prefix(setting_key)),
                        "relevance_score": float(1.0 / (1.0 + score)),  # Convert distance to similarity score
                    }

                    results.append(result)

                    if len(results) >= top_k:
                        break

            except Exception as e:
                logger.error(f"FAISS search failed: {e}")
                # Fall back to keyword search

        # Fallback to keyword-based search if FAISS not available or failed
        if not results and self.settings_data:
            query_lower = query.lower()
            query_words = query_lower.split()

            for setting_key, setting_data in self.settings_data.items():
                # Apply filters
                if prefix_filter:
                    prefix = self._extract_prefix(setting_key)
                    if prefix != prefix_filter:
                        continue

                if type_filter:
                    if setting_data.get("type") != type_filter:
                        continue

                # Simple scoring based on keyword matching
                score = 0.0

                # Check in setting key
                key_lower = setting_key.lower()
                for word in query_words:
                    if word in key_lower:
                        score += 2.0

                # Check in documentation
                doc = (setting_data.get("documentation", "") or "").lower()
                desc = (setting_data.get("description", "") or "").lower()

                for word in query_words:
                    if word in doc:
                        score += 1.5
                    if word in desc:
                        score += 1.0

                # Check in extensions
                extensions = setting_data.get("extensions", [])
                for ext in extensions:
                    if any(word in ext.lower() for word in query_words):
                        score += 0.5

                if score > 0:
                    result = {
                        "setting": setting_key,
                        "type": setting_data.get("type", "unknown"),
                        "default_value": setting_data.get("default_value"),
                        "documentation": setting_data.get("documentation", ""),
                        "description": setting_data.get("description", ""),
                        "extensions": list(extensions),
                        "usage_count": setting_data.get("usage_count", 0),
                        "found_in": setting_data.get("found_in", [])[:3],
                        "has_documentation": bool(setting_data.get("documentation")),
                        "prefix": self._extract_prefix(setting_key),
                        "relevance_score": score,
                    }
                    results.append(result)

            # Sort by relevance score and limit
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = results[:top_k]

        return results

    def get_setting_details(self, setting_keys: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information about specific settings.

        Args:
            setting_keys: List of setting keys to retrieve

        Returns:
            List of detailed setting information
        """
        if not self.settings_data:
            return []

        results = []

        for setting_key in setting_keys:
            setting_data = self.settings_data.get(setting_key)

            if setting_data:
                # Build detailed result
                result = {
                    "setting": setting_key,
                    "type": setting_data.get("type", "unknown"),
                    "default_value": setting_data.get("default_value"),
                    "documentation": setting_data.get("documentation", ""),
                    "description": setting_data.get("description", ""),
                    "extensions": list(setting_data.get("extensions", [])),
                    "usage_count": setting_data.get("usage_count", 0),
                    "found_in": setting_data.get("found_in", []),
                    "has_documentation": bool(setting_data.get("documentation")),
                    "prefix": self._extract_prefix(setting_key),
                }

                results.append(result)
            else:
                # Try to find similar settings
                similar = self._find_similar_settings(setting_key)
                if similar:
                    results.append(
                        {
                            "error": f"Setting '{setting_key}' not found",
                            "suggestion": f"Did you mean one of these? {', '.join(similar[:3])}",
                        }
                    )
                else:
                    results.append(
                        {
                            "error": f"Setting '{setting_key}' not found",
                            "suggestion": "Use search_settings to find available settings",
                        }
                    )

        return results

    def get_settings_by_extension(self, extension_id: str) -> List[Dict[str, Any]]:
        """Get all settings used by a specific extension.

        Args:
            extension_id: Extension ID to get settings for

        Returns:
            List of settings used by the extension
        """
        if not self.settings_data:
            return []

        results = []

        for setting_key, setting_data in self.settings_data.items():
            extensions = setting_data.get("extensions", [])
            if extension_id in extensions:
                result = {
                    "setting": setting_key,
                    "type": setting_data.get("type", "unknown"),
                    "default_value": setting_data.get("default_value"),
                    "documentation": setting_data.get("documentation", ""),
                    "usage_count": setting_data.get("usage_count", 0),
                    "prefix": self._extract_prefix(setting_key),
                }
                results.append(result)

        # Sort by setting key for organization
        results.sort(key=lambda x: x["setting"])

        return results

    def get_settings_statistics(self) -> Dict[str, Any]:
        """Get statistics about the settings database.

        Returns:
            Dictionary with statistics
        """
        if not self.metadata:
            return {
                "total_settings": len(self.settings_data) if self.settings_data else 0,
                "faiss_available": self.vectorstore is not None,
            }

        stats = {
            "total_settings": self.metadata.get("total_settings", len(self.settings_data)),
            "total_extensions_scanned": self.metadata.get("total_extensions_scanned", 0),
            "scan_directory": self.metadata.get("scan_directory", "unknown"),
            "version": self.metadata.get("version", "unknown"),
            "faiss_available": self.vectorstore is not None,
        }

        # Add type distribution if we have data
        if self.settings_data:
            type_counts = {}
            prefix_counts = {}
            documented_count = 0

            for setting_data in self.settings_data.values():
                # Count types
                setting_type = setting_data.get("type", "unknown")
                type_counts[setting_type] = type_counts.get(setting_type, 0) + 1

                # Count prefixes
                for setting_key in self.settings_data.keys():
                    prefix = self._extract_prefix(setting_key)
                    if prefix:
                        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
                        break

                # Count documented
                if setting_data.get("documentation"):
                    documented_count += 1

            stats["type_distribution"] = type_counts
            stats["prefix_distribution"] = prefix_counts
            stats["documented_settings"] = documented_count
            stats["documentation_percentage"] = (
                (documented_count / len(self.settings_data) * 100) if self.settings_data else 0
            )

        return stats

    def _extract_prefix(self, setting_key: str) -> str:
        """Extract the prefix from a setting key.

        Args:
            setting_key: Full setting path

        Returns:
            Prefix (e.g., 'exts', 'app', 'persistent', 'rtx')
        """
        if setting_key.startswith("/"):
            parts = setting_key.split("/")
            if len(parts) > 1:
                return parts[1]
        return ""

    def _find_similar_settings(self, setting_key: str) -> List[str]:
        """Find settings with similar names.

        Args:
            setting_key: Setting key to find similar ones for

        Returns:
            List of similar setting keys
        """
        if not self.settings_data:
            return []

        similar = []
        key_lower = setting_key.lower()

        # Extract key parts for matching
        key_parts = set(key_lower.replace("/", " ").replace(".", " ").replace("_", " ").split())

        for existing_key in self.settings_data.keys():
            existing_lower = existing_key.lower()
            existing_parts = set(existing_lower.replace("/", " ").replace(".", " ").replace("_", " ").split())

            # Calculate similarity
            common_parts = key_parts.intersection(existing_parts)
            if len(common_parts) >= len(key_parts) * 0.5:  # At least 50% match
                similar.append(existing_key)

        return similar[:5]  # Return top 5 similar
