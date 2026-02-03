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

"""API service for managing Kit API documentation and information using Code Atlas data."""

import logging
import os
from typing import Any, Dict, List, Optional

from .kit_exts_atlas import KitExtensionsAtlasService

logger = logging.getLogger(__name__)

# Maximum cache size - configurable via environment variable
_MAX_API_CACHE_SIZE = int(os.getenv("KIT_MCP_API_CACHE_SIZE", "1000"))


class APIService:
    """Service for managing Kit API information and documentation using Code Atlas."""

    def __init__(self):
        """Initialize the API service."""
        self.atlas_service = KitExtensionsAtlasService()
        # Use LRU cache with bounded size to prevent unbounded memory growth
        self._api_cache: Dict[str, Any] = {}
        self._cache_order: List[str] = []  # Track insertion order for LRU eviction
        self._max_cache_size = _MAX_API_CACHE_SIZE

    def _cache_set(self, key: str, value: Any) -> None:
        """Set a value in the cache with LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        # If key already exists, remove it from order list
        if key in self._api_cache:
            self._cache_order.remove(key)

        # Evict oldest entries if cache is at capacity
        while len(self._cache_order) >= self._max_cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._api_cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")

        # Add new entry
        self._api_cache[key] = value
        self._cache_order.append(key)

    def _cache_get(self, key: str) -> Optional[Any]:
        """Get a value from the cache and update its LRU position.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self._api_cache:
            # Move to end of order list (most recently used)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._api_cache[key]
        return None

    def is_available(self) -> bool:
        """Check if API data is available."""
        return self.atlas_service.is_available()

    def get_extension_apis(self, extension_ids: List[str]) -> List[Dict[str, Any]]:
        """Get all APIs provided by specified extensions.

        Args:
            extension_ids: List of extension IDs to get APIs for

        Returns:
            List of API information grouped by extension
        """
        if not self.is_available():
            return []

        results = []

        for ext_id in extension_ids:

            # Get API symbols from atlas data
            api_symbols = self.atlas_service.get_api_symbols(ext_id)

            if api_symbols:
                # Group APIs by type
                classes = [s for s in api_symbols if s["type"] == "class"]
                methods = [s for s in api_symbols if s["type"] in ["method", "function"]]

                # Format API listing
                api_list = []

                # Add classes
                for cls in classes:
                    docstring = cls.get("docstring", "") or ""
                    api_list.append(
                        {
                            "api_reference": cls["api_reference"],
                            "symbol": cls["symbol"],
                            "full_name": cls["full_name"],
                            "type": "class",
                            "docstring": docstring[:200] + "..." if len(docstring) > 200 else docstring,
                        }
                    )

                # Add standalone functions/methods
                for method in methods:
                    if not method.get("parent_class"):  # Only standalone functions
                        docstring = method.get("docstring", "") or ""
                        api_list.append(
                            {
                                "api_reference": method["api_reference"],
                                "symbol": method["symbol"],
                                "full_name": method["full_name"],
                                "type": "function",
                                "docstring": docstring[:200] + "..." if len(docstring) > 200 else docstring,
                            }
                        )

                results.append(
                    {
                        "extension_id": ext_id,
                        "api_count": len(api_list),
                        "class_count": len(classes),
                        "function_count": len([m for m in methods if not m.get("parent_class")]),
                        "apis": api_list,
                    }
                )
            else:
                # Check if extension exists
                ext_metadata = self.atlas_service.get_extension_metadata(ext_id)
                if ext_metadata:
                    results.append(
                        {
                            "extension_id": ext_id,
                            "api_count": 0,
                            "apis": [],
                            "note": f"Extension '{ext_id}' exists but has no Python APIs or documentation",
                        }
                    )
                else:
                    results.append(
                        {"extension_id": ext_id, "api_count": 0, "apis": [], "error": f"Extension '{ext_id}' not found"}
                    )

        return results

    def get_api_details(self, api_references: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information about specific APIs.

        Args:
            api_references: List of API references in format 'extension_id@symbol'

        Returns:
            List of detailed API information
        """
        if not self.is_available():
            return []

        results = []

        for api_ref in api_references:
            # Check cache first (with LRU update)
            cached_result = self._cache_get(api_ref)
            if cached_result is not None:
                results.append(cached_result)
                continue

            # Parse the API reference
            if "@" not in api_ref:
                results.append(
                    {
                        "api_reference": api_ref,
                        "error": f"Invalid API reference format: '{api_ref}'",
                        "suggestion": "Use format 'extension_id@symbol' (e.g., 'omni.ui@Window')",
                    }
                )
                continue

            parts = api_ref.split("@", 1)
            extension_id = parts[0]
            symbol = parts[1]

            # Load Code Atlas for the extension
            codeatlas = self.atlas_service.load_codeatlas(extension_id)
            api_docs = self.atlas_service.load_api_docs(extension_id)

            if not codeatlas and not api_docs:
                results.append(
                    {
                        "api_reference": api_ref,
                        "error": f"No API data found for extension '{extension_id}'",
                        "suggestion": "Extension may not exist or have no Python APIs",
                    }
                )
                continue

            # Try to find the API symbol
            api_info = None

            # First try API docs (cleaner format)
            if api_docs:
                # Check classes
                for class_name, class_info in api_docs.get("classes", {}).items():
                    if class_name == symbol:
                        # Found class
                        api_info = self._format_class_details(extension_id, class_name, class_info, from_api_docs=True)
                        break

                    # Check class methods
                    if "." in symbol:
                        method_parts = symbol.split(".", 1)
                        if method_parts[0] == class_name:
                            method_name = method_parts[1]
                            # Methods is a list of strings in our generated api_docs
                            methods = class_info.get("methods", [])
                            if isinstance(methods, list):
                                for method_item in methods:
                                    if isinstance(method_item, str):
                                        # It's just a method name
                                        if method_item == method_name:
                                            # Create a minimal method info
                                            api_info = self._format_method_details(
                                                extension_id,
                                                method_name,
                                                {"name": method_name},
                                                parent_class=class_name,
                                                from_api_docs=True,
                                            )
                                            break
                                    elif isinstance(method_item, dict):
                                        # It's a method info dict
                                        if method_item.get("name") == method_name:
                                            api_info = self._format_method_details(
                                                extension_id,
                                                method_name,
                                                method_item,
                                                parent_class=class_name,
                                                from_api_docs=True,
                                            )
                                            break
                            elif isinstance(methods, dict):
                                # Handle dict format if needed
                                if method_name in methods:
                                    api_info = self._format_method_details(
                                        extension_id,
                                        method_name,
                                        (
                                            methods[method_name]
                                            if isinstance(methods[method_name], dict)
                                            else {"name": method_name}
                                        ),
                                        parent_class=class_name,
                                        from_api_docs=True,
                                    )
                                    break
                            if api_info:
                                break

                # Check module-level functions
                if not api_info:
                    functions = api_docs.get("functions", [])
                    if isinstance(functions, list):
                        for func_info in functions:
                            if func_info.get("name") == symbol:
                                api_info = self._format_function_details(
                                    extension_id, symbol, func_info, from_api_docs=True
                                )
                                break
                    elif isinstance(functions, dict):
                        # Handle dict format if needed
                        for func_name, func_info in functions.items():
                            if func_name == symbol:
                                api_info = self._format_function_details(
                                    extension_id, func_name, func_info, from_api_docs=True
                                )
                                break

            # Fallback to Code Atlas
            if not api_info and codeatlas:
                # Check classes
                for class_key, class_info in codeatlas.get("classes", {}).items():
                    if class_info.get("name") == symbol or symbol in class_info.get("full_name", ""):
                        api_info = self._format_class_details(extension_id, symbol, class_info)
                        break

                # Check methods
                if not api_info:
                    for method_key, method_info in codeatlas.get("methods", {}).items():
                        method_name = method_info.get("name", "")
                        full_name = method_info.get("full_name", "")

                        if method_name == symbol or symbol in full_name:
                            api_info = self._format_method_details(extension_id, method_name, method_info)
                            break

            if api_info:
                # Cache the result with LRU eviction
                self._cache_set(api_ref, api_info)
                results.append(api_info)
            else:
                # Try fuzzy matching
                from ..config import DEFAULT_FUZZY_MATCH_THRESHOLD
                from ..utils.fuzzy_matching import find_best_match

                # Get all available symbols for this extension
                api_symbols = self.atlas_service.get_api_symbols(extension_id)
                if api_symbols:
                    candidates = [s["symbol"] for s in api_symbols]
                    match = find_best_match(symbol, candidates, threshold=DEFAULT_FUZZY_MATCH_THRESHOLD)

                    if match:
                        matched_symbol, score = match
                        results.append(
                            {
                                "api_reference": api_ref,
                                "error": f"API symbol '{symbol}' not found in extension '{extension_id}'",
                                "suggestion": f"Did you mean '{extension_id}@{matched_symbol}'?",
                                "match_score": score,
                            }
                        )
                    else:
                        results.append(
                            {
                                "api_reference": api_ref,
                                "error": f"API symbol '{symbol}' not found in extension '{extension_id}'",
                                "suggestion": "Use get_extension_apis to see available APIs",
                            }
                        )
                else:
                    results.append({"api_reference": api_ref, "error": f"No APIs found for extension '{extension_id}'"})

        return results

    def _format_class_details(
        self, extension_id: str, class_name: str, class_info: Dict[str, Any], from_api_docs: bool = False
    ) -> Dict[str, Any]:
        """Format class details for API response."""
        result = {
            "extension_id": extension_id,
            "symbol": class_name,
            "full_name": class_info.get("full_name", f"{extension_id}.{class_name}"),
            "type": "class",
            "docstring": class_info.get("docstring", ""),
            "api_reference": f"{extension_id}@{class_name}",
        }

        if from_api_docs:
            # Format from API docs (methods is a list of strings in our generated api_docs)
            methods = []
            methods_data = class_info.get("methods", [])
            if isinstance(methods_data, list):
                for method_item in methods_data:
                    if isinstance(method_item, str):
                        # It's just a method name
                        methods.append(
                            {
                                "name": method_item,
                                "signature": f"{method_item}(...)",
                                "docstring": "",
                                "return_type": "Any",
                            }
                        )
                    elif isinstance(method_item, dict):
                        # It's a method info dict
                        methods.append(
                            {
                                "name": method_item.get("name", ""),
                                "signature": method_item.get("signature", ""),
                                "docstring": method_item.get("docstring", ""),
                                "return_type": method_item.get("return_type", "Any"),
                            }
                        )
            elif isinstance(methods_data, dict):
                # Handle dict format if needed
                for method_name, method_info in methods_data.items():
                    methods.append(
                        {
                            "name": method_name,
                            "signature": (
                                method_info.get("signature", f"{method_name}(...)")
                                if isinstance(method_info, dict)
                                else f"{method_name}(...)"
                            ),
                            "docstring": method_info.get("docstring", "") if isinstance(method_info, dict) else "",
                            "return_type": (
                                method_info.get("return_type", "Any") if isinstance(method_info, dict) else "Any"
                            ),
                        }
                    )

            result["methods"] = methods
            result["properties"] = []  # Could extract if available

        else:
            # Format from Code Atlas
            result["parent_classes"] = class_info.get("parent_classes", [])
            result["line_number"] = class_info.get("line_number")
            result["module_name"] = class_info.get("module_name", "")

            # Get method names
            method_names = class_info.get("methods", [])
            result["methods"] = [{"name": m, "signature": f"{m}(...)"} for m in method_names]

        return result

    def _format_method_details(
        self,
        extension_id: str,
        method_name: str,
        method_info: Dict[str, Any],
        parent_class: str = None,
        from_api_docs: bool = False,
    ) -> Dict[str, Any]:
        """Format method details for API response."""
        result = {
            "extension_id": extension_id,
            "symbol": method_name if not parent_class else f"{parent_class}.{method_name}",
            "full_name": method_info.get(
                "full_name",
                f"{extension_id}.{parent_class}.{method_name}" if parent_class else f"{extension_id}.{method_name}",
            ),
            "type": "method" if parent_class else "function",
            "docstring": method_info.get("docstring", ""),
            "api_reference": (
                f"{extension_id}@{parent_class}.{method_name}" if parent_class else f"{extension_id}@{method_name}"
            ),
        }

        if from_api_docs:
            # Format from API docs
            result["signature"] = method_info.get("signature", f"{method_name}(...)")
            result["return_type"] = method_info.get("return_type", "Any")
            result["parameters"] = method_info.get("arguments", [])

        else:
            # Format from Code Atlas
            result["parent_class"] = method_info.get("parent_class", parent_class)
            result["line_number"] = method_info.get("line_number")
            result["module_name"] = method_info.get("module_name", "")
            result["return_type"] = method_info.get("return_type", "Any")
            result["is_async_method"] = method_info.get("is_async_method", False)
            result["is_class_method"] = method_info.get("is_class_method", False)
            result["is_static_method"] = method_info.get("is_static_method", False)

            # Format arguments
            arguments = method_info.get("arguments", [])
            parameters = []
            for arg in arguments:
                param = {"name": arg.get("name", ""), "type": arg.get("type", "Any"), "default": arg.get("default")}
                parameters.append(param)
            result["parameters"] = parameters

            # Build signature
            param_strs = []
            for p in parameters:
                param_str = p["name"]
                if p.get("type") and p["type"] != "Any":
                    param_str += f": {p['type']}"
                if p.get("default") is not None:
                    param_str += f" = {p['default']}"
                param_strs.append(param_str)

            signature = f"{method_name}({', '.join(param_strs)})"
            if result.get("return_type") and result["return_type"] != "Any":
                signature += f" -> {result['return_type']}"
            result["signature"] = signature

        return result

    def _format_function_details(
        self, extension_id: str, func_name: str, func_info: Dict[str, Any], from_api_docs: bool = False
    ) -> Dict[str, Any]:
        """Format function details for API response."""
        # Functions are like methods without a parent class
        return self._format_method_details(
            extension_id, func_name, func_info, parent_class=None, from_api_docs=from_api_docs
        )

    def get_api_list(self) -> List[str]:
        """Get list of all available API references."""
        if not self.is_available():
            return []

        api_refs = []

        # Iterate through all extensions
        for ext_id in self.atlas_service.get_extension_list():
            api_symbols = self.atlas_service.get_api_symbols(ext_id)
            for symbol in api_symbols:
                api_refs.append(symbol["api_reference"])

        return sorted(api_refs)
