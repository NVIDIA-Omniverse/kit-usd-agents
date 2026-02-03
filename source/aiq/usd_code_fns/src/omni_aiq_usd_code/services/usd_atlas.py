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

"""USD Atlas data service for the USD RAG MCP server."""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import USD_ATLAS_FILE_PATH
from ..utils.patching import patch_information

logger = logging.getLogger(__name__)


class USDAtlasService:
    """Service for managing USD Atlas data operations."""

    def __init__(self, atlas_file_path: str = USD_ATLAS_FILE_PATH):
        """Initialize the USD Atlas service.

        Args:
            atlas_file_path: Path to the USD Atlas JSON file
        """
        self.atlas_file_path = atlas_file_path
        self.atlas_data = None
        self._load_atlas_data()

    def _load_atlas_data(self) -> None:
        """Load USD Atlas data from file."""
        try:
            with open(self.atlas_file_path, "r") as f:
                raw_content = f.read()

            # Apply patches before JSON decoding
            patched_content = patch_information(raw_content)

            self.atlas_data = json.loads(patched_content)
            logger.info(f"Successfully loaded USD Atlas data from {self.atlas_file_path}")
        except FileNotFoundError:
            logger.warning(f"USD Atlas file not found: {self.atlas_file_path}")
            self.atlas_data = None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.atlas_file_path}: {e}")
            self.atlas_data = None

    def is_available(self) -> bool:
        """Check if USD Atlas data is available."""
        return self.atlas_data is not None

    def get_modules(self) -> Dict[str, Any]:
        """Get all USD modules.

        Returns:
            Dictionary containing module information and summary
        """
        if not self.is_available():
            return {"error": "USD Atlas data is not available."}

        if "modules" not in self.atlas_data:
            return {"error": "No modules section found in USD Atlas data."}

        modules = self.atlas_data["modules"]
        result = {
            "modules": [],
            "total_count": 0,
            "summary": {
                "total_modules": 0,
                "modules_with_classes": 0,
                "modules_with_functions": 0,
                "total_classes": 0,
                "total_functions": 0,
            },
        }

        for module_key, module_info in modules.items():
            name = module_info.get("name", "Unknown")

            # Skip __DOC modules for cleaner output
            if name == "__DOC":
                continue

            full_name = module_info.get("full_name", module_key)
            file_path = module_info.get("file_path", "")
            class_names = module_info.get("class_names", [])
            function_names = module_info.get("function_names", [])

            module_data = {
                "name": name,
                "full_name": full_name,
                "file_path": file_path,
                "class_count": len(class_names),
                "function_count": len(function_names),
                "class_names": class_names,
                "function_names": function_names,
            }

            result["modules"].append(module_data)
            result["summary"]["total_modules"] += 1
            result["summary"]["total_classes"] += len(class_names)
            result["summary"]["total_functions"] += len(function_names)

            if class_names:
                result["summary"]["modules_with_classes"] += 1
            if function_names:
                result["summary"]["modules_with_functions"] += 1

        result["total_count"] = len(result["modules"])
        return result

    def get_classes(self) -> Dict[str, Any]:
        """Get all USD classes.

        Returns:
            Dictionary containing class information and summary
        """
        if not self.is_available():
            return {"error": "USD Atlas data is not available."}

        if "classes" not in self.atlas_data:
            return {"error": "No classes section found in USD Atlas data."}

        classes = self.atlas_data["classes"]
        result = {
            "classes": [],
            "total_count": 0,
            "summary": {"total_classes": 0, "classes_with_methods": 0, "total_methods": 0, "modules": set()},
        }

        for class_key, class_info in classes.items():
            name = class_info.get("name", "Unknown")
            full_name = class_info.get("full_name", class_key)
            module_name = class_info.get("module_name", "Unknown")
            methods = class_info.get("methods", [])

            class_data = {
                "name": name,
                "full_name": full_name,
                "module_name": module_name,
                "method_count": len(methods),
                "methods": methods,
                "docstring": class_info.get("docstring", ""),
                "parent_classes": class_info.get("parent_classes", []),
            }

            result["classes"].append(class_data)
            result["summary"]["total_classes"] += 1
            result["summary"]["total_methods"] += len(methods)
            result["summary"]["modules"].add(module_name)

            if methods:
                result["summary"]["classes_with_methods"] += 1

        result["total_count"] = len(result["classes"])
        result["summary"]["modules"] = list(result["summary"]["modules"])
        result["summary"]["unique_modules"] = len(result["summary"]["modules"])

        return result

    def get_module_detail(self, module_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific module.

        Args:
            module_name: Name of the module to look up

        Returns:
            Dictionary containing detailed module information
        """
        if not self.is_available():
            return {"error": "USD Atlas data is not available."}

        if "modules" not in self.atlas_data:
            return {"error": "No modules section found in USD Atlas data."}

        modules = self.atlas_data["modules"]

        # Use fuzzy matching to find the best module match
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            module_name, modules, key_func=lambda k, v: [k, v.get("name", ""), v.get("full_name", "")]
        )

        if not matches:
            return {
                "error": f"Module '{module_name}' not found in USD Atlas data.",
                "suggestion": "Try using get_usd_modules() to see available modules.",
            }

        # Get the best match
        target_key, target_module, match_score = matches[0]

        result = {
            "module": {
                "name": target_module.get("name", "Unknown"),
                "full_name": target_module.get("full_name", target_key),
                "file_path": target_module.get("file_path", "Unknown"),
                "match_score": match_score,
            },
            "classes": target_module.get("class_names", []),
            "functions": target_module.get("function_names", []),
            "summary": {
                "class_count": len(target_module.get("class_names", [])),
                "function_count": len(target_module.get("function_names", [])),
            },
        }

        # If there are multiple good matches, include them as alternatives
        if len(matches) > 1 and matches[1][2] > 0.5:
            result["alternatives"] = []
            for alt_key, alt_module, alt_score in matches[1:6]:  # Top 5 alternatives
                if alt_score > 0.5:
                    result["alternatives"].append(
                        {
                            "name": alt_module.get("name", "Unknown"),
                            "full_name": alt_module.get("full_name", alt_key),
                            "match_score": alt_score,
                        }
                    )

        return result

    def _find_class_by_name(self, class_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Find a class by name using fuzzy matching.

        Args:
            class_name: Name of the class to find

        Returns:
            Tuple of (class_key, class_info) if found, None otherwise
        """
        if not self.is_available() or "classes" not in self.atlas_data:
            return None

        classes = self.atlas_data["classes"]
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            class_name, classes, key_func=lambda k, v: [k, v.get("name", ""), v.get("full_name", "")]
        )

        if matches and matches[0][2] > 0.5:  # Good match threshold
            return matches[0][0], matches[0][1]
        return None

    def _collect_ancestor_methods(self, class_info: Dict[str, Any], visited: Set[str] = None) -> List[str]:
        """Recursively collect methods from ancestor classes.

        Args:
            class_info: Class information dictionary
            visited: Set of visited class names to prevent infinite recursion

        Returns:
            List of method names from ancestor classes
        """
        if visited is None:
            visited = set()

        ancestor_methods = []
        parent_classes = class_info.get("parent_classes", [])

        for parent_class in parent_classes:
            if parent_class in visited:
                continue
            visited.add(parent_class)

            parent_info = self._find_class_by_name(parent_class)
            if parent_info:
                parent_key, parent_data = parent_info
                # Add parent's methods
                parent_methods = parent_data.get("methods", [])
                ancestor_methods.extend(parent_methods)

                # Recursively collect from grandparents
                grandparent_methods = self._collect_ancestor_methods(parent_data, visited)
                ancestor_methods.extend(grandparent_methods)

        return ancestor_methods

    def _collect_ancestor_variables(self, class_info: Dict[str, Any], visited: Set[str] = None) -> Dict[str, str]:
        """Recursively collect class variables from ancestor classes.

        Args:
            class_info: Class information dictionary
            visited: Set of visited class names to prevent infinite recursion

        Returns:
            Dictionary of variable names to type annotations from ancestor classes
        """
        if visited is None:
            visited = set()

        ancestor_variables = {}
        parent_classes = class_info.get("parent_classes", [])

        for parent_class in parent_classes:
            if parent_class in visited:
                continue
            visited.add(parent_class)

            parent_info = self._find_class_by_name(parent_class)
            if parent_info:
                parent_key, parent_data = parent_info

                # Add parent's class variables
                raw_parent_variables = parent_data.get("class_variables", [])
                for var in raw_parent_variables:
                    if isinstance(var, dict):
                        var_name = var.get("name", "Unknown")
                        type_annotation = var.get("type_annotation", "")
                        ancestor_variables[var_name] = type_annotation

                # Recursively collect from grandparents
                grandparent_variables = self._collect_ancestor_variables(parent_data, visited)
                ancestor_variables.update(grandparent_variables)

        return ancestor_variables

    def _search_method_in_ancestors(
        self, method_name: str, class_name: str, visited: Set[str] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search for a method in ancestor classes.

        Args:
            method_name: Name of the method to search for
            class_name: Name of the class whose ancestors to search
            visited: Set of visited class names to prevent infinite recursion

        Returns:
            List of tuples (method_key, method_info, match_score) from ancestor classes
        """
        if visited is None:
            visited = set()

        if class_name in visited:
            return []
        visited.add(class_name)

        ancestor_matches = []

        # Find the class
        class_info = self._find_class_by_name(class_name)
        if not class_info:
            return []

        class_key, class_data = class_info
        parent_classes = class_data.get("parent_classes", [])

        if not self.atlas_data or "methods" not in self.atlas_data:
            return []

        methods = self.atlas_data["methods"]
        from ..utils.fuzzy_matching import find_best_matches, fuzzy_match_score

        for parent_class in parent_classes:
            # Search for method in this parent class
            parent_filtered_methods = {}
            for method_key, method_info in methods.items():
                # Use improved fuzzy matching for class name comparison
                class_name_match = fuzzy_match_score(parent_class, method_info.get("class_name", "")) > 0.3

                # Also check if class name appears in the method key using fuzzy matching
                method_key_match = fuzzy_match_score(parent_class, method_key) > 0.3

                if class_name_match or method_key_match:
                    parent_filtered_methods[method_key] = method_info

            parent_matches = find_best_matches(
                method_name,
                parent_filtered_methods,
                key_func=lambda k, v: [v.get("name", ""), k.split(".")[-1] if "." in k else k],
            )

            ancestor_matches.extend(parent_matches)

            # Recursively search in grandparents
            grandparent_matches = self._search_method_in_ancestors(method_name, parent_class, visited)
            ancestor_matches.extend(grandparent_matches)

        return ancestor_matches

    def get_class_detail(self, class_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific class.

        Args:
            class_name: Name of the class to look up

        Returns:
            Dictionary containing detailed class information including inherited methods and variables
        """
        if not self.is_available():
            return {"error": "USD Atlas data is not available."}

        if "classes" not in self.atlas_data:
            return {"error": "No classes section found in USD Atlas data."}

        classes = self.atlas_data["classes"]

        # Use fuzzy matching to find the best class match
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            class_name, classes, key_func=lambda k, v: [k, v.get("name", ""), v.get("full_name", "")]
        )

        if not matches:
            return {
                "error": f"Class '{class_name}' not found in USD Atlas data.",
                "suggestion": "Try using get_usd_classes() to see available classes.",
            }

        # Get the best match
        target_key, target_class, match_score = matches[0]

        # Process class variables (own class)
        own_class_variables = {}
        raw_class_variables = target_class.get("class_variables", [])
        for var in raw_class_variables:
            if isinstance(var, dict):
                var_name = var.get("name", "Unknown")
                # Use type_annotation if it exists, otherwise empty string
                type_annotation = var.get("type_annotation", "")
                own_class_variables[var_name] = type_annotation

        # Collect methods and variables from ancestor classes
        own_methods = target_class.get("methods", [])
        ancestor_methods = self._collect_ancestor_methods(target_class)
        ancestor_variables = self._collect_ancestor_variables(target_class)

        # Combine all methods and variables
        all_methods = own_methods + ancestor_methods
        all_class_variables = {**ancestor_variables, **own_class_variables}  # Own variables override ancestor ones

        result = {
            "class": {
                "name": target_class.get("name", "Unknown"),
                "full_name": target_class.get("full_name", target_key),
                "module_name": target_class.get("module_name", "Unknown"),
                "match_score": match_score,
                "docstring": target_class.get("docstring", ""),
                "parent_classes": target_class.get("parent_classes", []),
            },
            "methods": {"own": own_methods, "inherited": ancestor_methods, "all": all_methods},
            "class_variables": {
                "own": own_class_variables,
                "inherited": ancestor_variables,
                "all": all_class_variables,
            },
            "summary": {
                "own_method_count": len(own_methods),
                "inherited_method_count": len(ancestor_methods),
                "total_method_count": len(all_methods),
                "own_class_variable_count": len(own_class_variables),
                "inherited_class_variable_count": len(ancestor_variables),
                "total_class_variable_count": len(all_class_variables),
                "has_docstring": bool(target_class.get("docstring", "")),
                "has_parent_classes": bool(target_class.get("parent_classes", [])),
                "has_inheritance": bool(ancestor_methods or ancestor_variables),
            },
        }

        # If there are multiple good matches, include them as alternatives
        if len(matches) > 1 and matches[1][2] > 0.5:
            result["alternatives"] = []
            for alt_key, alt_class, alt_score in matches[1:6]:  # Top 5 alternatives
                if alt_score > 0.5:
                    result["alternatives"].append(
                        {
                            "name": alt_class.get("name", "Unknown"),
                            "full_name": alt_class.get("full_name", alt_key),
                            "module_name": alt_class.get("module_name", "Unknown"),
                            "match_score": alt_score,
                        }
                    )

        return result

    def get_method_detail(self, method_name: str, class_name: str = "") -> Dict[str, Any]:
        """Get detailed information about a specific method.

        Args:
            method_name: Name of the method to look up
            class_name: Optional class name to narrow down the search

        Returns:
            Dictionary containing detailed method information including inherited methods
        """
        if not self.is_available():
            return {"error": "USD Atlas data is not available."}

        if "methods" not in self.atlas_data:
            return {"error": "No methods section found in USD Atlas data."}

        methods = self.atlas_data["methods"]

        # Use fuzzy matching to find methods
        from ..utils.fuzzy_matching import find_best_matches, fuzzy_match_score

        matches = []

        if class_name:
            # Filter by class first, then match method name
            class_filtered_methods = {}
            for method_key, method_info in methods.items():
                # Use improved fuzzy matching for class name comparison
                class_name_match = fuzzy_match_score(class_name, method_info.get("class_name", "")) > 0.3

                # Also check if class name appears in the method key using fuzzy matching
                method_key_match = fuzzy_match_score(class_name, method_key) > 0.3

                if class_name_match or method_key_match:
                    class_filtered_methods[method_key] = method_info

            matches = find_best_matches(
                method_name,
                class_filtered_methods,
                key_func=lambda k, v: [v.get("name", ""), k.split(".")[-1] if "." in k else k],
            )

            # If no matches found in the specified class, search in ancestor classes
            if not matches:
                ancestor_matches = self._search_method_in_ancestors(method_name, class_name)
                matches = ancestor_matches
        else:
            matches = find_best_matches(
                method_name, methods, key_func=lambda k, v: [v.get("name", ""), k.split(".")[-1] if "." in k else k]
            )

        if not matches:
            search_info = f" in class '{class_name}' or its ancestors" if class_name else ""
            return {
                "error": f"Method '{method_name}'{search_info} not found in USD Atlas data.",
                "suggestion": "Try using get_usd_class_detail() to see available methods for a class.",
            }

        result = {
            "query": {
                "method_name": method_name,
                "class_name": class_name,
                "total_matches": len(matches),
                "searched_ancestors": bool(class_name),
            },
            "methods": [],
        }

        # Include top matches (up to 5)
        for i, (method_key, method_info, match_score) in enumerate(matches[:5]):
            # Extract class name from full_name (e.g., "pxr.Usd.Attribute.Clear" -> "Attribute")
            full_name = method_info.get("full_name", method_key)
            extracted_class_name = "Unknown"
            if full_name and "." in full_name:
                parts = full_name.split(".")
                if len(parts) >= 2:
                    # For "pxr.Usd.Attribute.Clear", we want "Attribute"
                    extracted_class_name = parts[-2]  # Second to last part

            # Determine if this method is from the target class or an ancestor
            is_inherited = False
            if class_name:
                is_inherited = fuzzy_match_score(class_name, extracted_class_name) < 0.5

            method_data = {
                "full_name": full_name,
                "name": method_info.get("name", "Unknown"),
                "class_name": extracted_class_name,
                "module_name": method_info.get("module_name", "Unknown"),
                "match_score": match_score,
                "docstring": method_info.get("docstring", ""),
                "signature": method_info.get("signature", ""),
                "arguments": method_info.get("arguments", []),
                "return_type": method_info.get("return_type", ""),
                "is_primary_match": i == 0,
                "is_inherited": is_inherited,
            }

            result["methods"].append(method_data)

        # Add summary
        result["summary"] = {
            "primary_match": result["methods"][0]["full_name"] if result["methods"] else None,
            "has_multiple_matches": len(matches) > 1,
            "methods_with_docstring": sum(1 for m in result["methods"] if m["docstring"]),
            "methods_with_signature": sum(1 for m in result["methods"] if m["signature"]),
            "methods_with_return_type": sum(1 for m in result["methods"] if m["return_type"]),
            "inherited_methods_count": sum(1 for m in result["methods"] if m["is_inherited"]),
            "own_methods_count": sum(1 for m in result["methods"] if not m["is_inherited"]),
        }

        return result
