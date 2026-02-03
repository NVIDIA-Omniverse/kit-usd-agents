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

"""OmniUI Atlas data service for the OmniUI MCP server."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Get the path to ui_atlas.json
UI_ATLAS_FILE_PATH = Path(__file__).parent.parent / "data" / "ui_atlas.json"


class OmniUIAtlasService:
    """Service for managing OmniUI Atlas data operations."""

    def __init__(self, atlas_file_path: str = None):
        """Initialize the OmniUI Atlas service.

        Args:
            atlas_file_path: Path to the UI Atlas JSON file
        """
        self.atlas_file_path = Path(atlas_file_path) if atlas_file_path else UI_ATLAS_FILE_PATH
        self.atlas_data = None
        self._load_atlas_data()

    def _load_atlas_data(self) -> None:
        """Load OmniUI Atlas data from file."""
        try:
            with open(self.atlas_file_path, "r", encoding="utf-8") as f:
                self.atlas_data = json.load(f)
            logger.info(f"Successfully loaded OmniUI Atlas data from {self.atlas_file_path}")
        except FileNotFoundError:
            logger.warning(f"OmniUI Atlas file not found: {self.atlas_file_path}")
            self.atlas_data = None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.atlas_file_path}: {e}")
            self.atlas_data = None
        except Exception as e:
            logger.error(f"Unexpected error loading Atlas data: {e}")
            self.atlas_data = None

    def is_available(self) -> bool:
        """Check if OmniUI Atlas data is available."""
        return self.atlas_data is not None

    def get_modules(self) -> Dict[str, Any]:
        """Get all OmniUI modules.

        Returns:
            Dictionary containing module information and summary
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "modules" not in self.atlas_data:
            return {"error": "No modules section found in OmniUI Atlas data."}

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
                "extensions": set(),
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
            extension_name = module_info.get("extension_name", "")

            module_data = {
                "name": name,
                "full_name": full_name,
                "file_path": file_path,
                "extension_name": extension_name,
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
            if extension_name:
                result["summary"]["extensions"].add(extension_name)

        result["total_count"] = len(result["modules"])
        result["summary"]["extensions"] = list(result["summary"]["extensions"])
        result["summary"]["unique_extensions"] = len(result["summary"]["extensions"])

        return result

    def get_classes(self) -> Dict[str, Any]:
        """Get all OmniUI classes.

        Returns:
            Dictionary containing class information and summary
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "classes" not in self.atlas_data:
            return {"error": "No classes section found in OmniUI Atlas data."}

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
            return {"error": "OmniUI Atlas data is not available."}

        if "modules" not in self.atlas_data:
            return {"error": "No modules section found in OmniUI Atlas data."}

        modules = self.atlas_data["modules"]

        # Use fuzzy matching to find the best module match
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            query=module_name,
            candidates=modules,
            key_func=lambda x: x,
            value_func=lambda x: modules[x],
            name_func=lambda x: x.get("full_name", ""),
            threshold=0.5,
        )

        if not matches:
            return {
                "error": f"Module '{module_name}' not found in OmniUI Atlas data.",
                "suggestion": "Try using get_omni_ui_modules() to see available modules.",
            }

        # Get the best match
        target_key, target_module, match_score = matches[0]

        # Get associated classes
        module_classes = []
        if "classes" in self.atlas_data:
            for class_key, class_info in self.atlas_data["classes"].items():
                if class_info.get("module_name") == target_module.get("full_name"):
                    module_classes.append(
                        {
                            "name": class_info.get("name"),
                            "full_name": class_info.get("full_name"),
                            "method_count": len(class_info.get("methods", [])),
                        }
                    )

        # Get associated functions
        module_functions = []
        if "functions" in self.atlas_data:
            for func_key, func_info in self.atlas_data["functions"].items():
                if func_info.get("module_name") == target_module.get("full_name"):
                    module_functions.append(
                        {
                            "name": func_info.get("name"),
                            "full_name": func_info.get("full_name"),
                            "docstring": func_info.get("docstring", ""),
                        }
                    )

        return {
            "name": target_module.get("name"),
            "full_name": target_module.get("full_name"),
            "file_path": target_module.get("file_path"),
            "extension_name": target_module.get("extension_name", ""),
            "class_names": target_module.get("class_names", []),
            "function_names": target_module.get("function_names", []),
            "classes": module_classes,
            "functions": module_functions,
            "match_score": match_score,
            "total_classes": len(module_classes),
            "total_functions": len(module_functions),
        }

    def get_class_detail(self, class_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific class.

        Args:
            class_name: Name of the class to look up

        Returns:
            Dictionary containing detailed class information
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "classes" not in self.atlas_data:
            return {"error": "No classes section found in OmniUI Atlas data."}

        classes = self.atlas_data["classes"]

        # Use fuzzy matching to find the best class match
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            query=class_name,
            candidates=classes,
            key_func=lambda x: x,
            value_func=lambda x: classes[x],
            name_func=lambda x: x.get("full_name", ""),
            threshold=0.5,
        )

        if not matches:
            return {
                "error": f"Class '{class_name}' not found in OmniUI Atlas data.",
                "suggestion": "Try using get_omni_ui_classes() to see available classes.",
            }

        # Get the best match
        target_key, target_class, match_score = matches[0]

        # Get method details with defensive handling for invalid data
        method_details = []
        methods = target_class.get("methods", [])

        if "methods" in self.atlas_data:
            for method_name in methods:
                try:
                    # Find the method in the atlas
                    class_full_name = target_class.get("full_name", "")
                    if not class_full_name:
                        continue
                    method_key = f"{class_full_name}.{method_name}"
                    if method_key in self.atlas_data["methods"]:
                        method_info = self.atlas_data["methods"][method_key]
                        if not isinstance(method_info, dict):
                            continue
                        method_details.append(
                            {
                                "name": method_info.get("name"),
                                "full_name": method_info.get("full_name"),
                                "parameters": method_info.get("parameters", []),
                                "return_type": method_info.get("return_type", ""),
                                "docstring": method_info.get("docstring", ""),
                                "is_static": method_info.get("is_static", False),
                                "is_classmethod": method_info.get("is_classmethod", False),
                            }
                        )
                except (KeyError, TypeError, AttributeError):
                    # Skip invalid method entries gracefully
                    continue

        return {
            "name": target_class.get("name"),
            "full_name": target_class.get("full_name"),
            "module_name": target_class.get("module_name"),
            "docstring": target_class.get("docstring", ""),
            "parent_classes": target_class.get("parent_classes", []),
            "methods": methods,
            "method_details": method_details,
            "match_score": match_score,
            "total_methods": len(methods),
        }

    def get_method_detail(self, method_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific method.

        Args:
            method_name: Name of the method to look up (can be partial or full name)

        Returns:
            Dictionary containing detailed method information
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "methods" not in self.atlas_data:
            return {"error": "No methods section found in OmniUI Atlas data."}

        methods = self.atlas_data["methods"]

        # Use fuzzy matching to find the best method match
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            query=method_name,
            candidates=methods,
            key_func=lambda x: x,
            value_func=lambda x: methods[x],
            name_func=lambda x: x.get("full_name", ""),
            threshold=0.5,
        )

        if not matches:
            return {
                "error": f"Method '{method_name}' not found in OmniUI Atlas data.",
                "suggestion": "Try using get_omni_ui_class_detail() to see methods for a specific class.",
            }

        # Get the best match
        target_key, target_method, match_score = matches[0]

        # Parse the class name from the full method name
        full_name = target_method.get("full_name", "")
        class_name = ".".join(full_name.split(".")[:-1]) if "." in full_name else ""

        return {
            "name": target_method.get("name"),
            "full_name": full_name,
            "class_name": class_name,
            "parameters": target_method.get("parameters", []),
            "return_type": target_method.get("return_type", ""),
            "docstring": target_method.get("docstring", ""),
            "is_static": target_method.get("is_static", False),
            "is_classmethod": target_method.get("is_classmethod", False),
            "is_property": target_method.get("is_property", False),
            "match_score": match_score,
        }

    def get_class_names_list(self) -> List[str]:
        """Get a simple list of all class full names.

        Returns:
            List of class full names sorted alphabetically
        """
        if not self.is_available():
            return []

        if "classes" not in self.atlas_data:
            return []

        class_names = []
        for class_key, class_info in self.atlas_data["classes"].items():
            full_name = class_info.get("full_name", class_key)
            class_names.append(full_name)

        return sorted(class_names)

    def get_module_names_list(self) -> List[str]:
        """Get a simple list of all module full names.

        Returns:
            List of module full names sorted alphabetically
        """
        if not self.is_available():
            return []

        if "modules" not in self.atlas_data:
            return []

        module_names = []
        for module_key, module_info in self.atlas_data["modules"].items():
            name = module_info.get("name", "")
            # Skip __DOC modules
            if name == "__DOC":
                continue
            full_name = module_info.get("full_name", module_key)
            module_names.append(full_name)

        return sorted(module_names)

    def get_method_list(self) -> List[str]:
        """Get a simple list of all method full names.

        Returns:
            List of method full names sorted alphabetically
        """
        if not self.is_available():
            return []

        if "methods" not in self.atlas_data:
            return []

        method_names = []
        for method_key, method_info in self.atlas_data["methods"].items():
            full_name = method_info.get("full_name", method_key)
            method_names.append(full_name)

        return sorted(method_names)
