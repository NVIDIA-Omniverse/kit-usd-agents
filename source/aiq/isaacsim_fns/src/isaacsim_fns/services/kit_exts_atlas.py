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

"""Isaac Sim Extensions Atlas service for managing Code Atlas data."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..config import ISAACSIM_VERSION

logger = logging.getLogger(__name__)

# Get the paths to our generated data files
DATA_BASE_PATH = Path(__file__).parent.parent / "data" / ISAACSIM_VERSION / "extensions"
EXTENSIONS_DATABASE_FILE = DATA_BASE_PATH / "extensions_database.json"
CODEATLAS_DIR = DATA_BASE_PATH / "codeatlas"


class KitExtensionsAtlasService:
    """Service for managing Isaac Sim Extensions Atlas data operations."""

    def __init__(self, database_file_path: str = None, codeatlas_dir: str = None):
        """Initialize the Isaac Sim Extensions Atlas service.

        Args:
            database_file_path: Path to the extensions database JSON file
            codeatlas_dir: Path to the directory containing Code Atlas files
        """
        self.database_file_path = Path(database_file_path) if database_file_path else EXTENSIONS_DATABASE_FILE
        self.codeatlas_dir = Path(codeatlas_dir) if codeatlas_dir else CODEATLAS_DIR

        self.database = None
        self.extensions = {}
        self._cached_codeatlas = {}  # Cache for loaded Code Atlas files
        self._load_database()

    def _load_database(self) -> None:
        """Load extensions database from file."""
        try:
            if not self.database_file_path.exists():
                logger.warning(f"Extensions database file not found: {self.database_file_path}")
                self.database = None
                return

            with open(self.database_file_path, "r", encoding="utf-8") as f:
                self.database = json.load(f)

            # Extract extensions data
            if "extensions" in self.database:
                self.extensions = self.database["extensions"]
            else:
                # Fallback for old format
                self.extensions = {
                    k: v
                    for k, v in self.database.items()
                    if isinstance(v, dict) and k not in ["database_version", "generated_at", "total_extensions"]
                }

            logger.info(f"Successfully loaded {len(self.extensions)} extensions from database")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.database_file_path}: {e}")
            self.database = None
            self.extensions = {}
        except Exception as e:
            logger.error(f"Unexpected error loading database: {e}")
            self.database = None
            self.extensions = {}

    def is_available(self) -> bool:
        """Check if extensions database is available."""
        return self.database is not None and len(self.extensions) > 0

    def get_extension_list(self) -> List[str]:
        """Get list of all available extension IDs."""
        if not self.is_available():
            return []
        return sorted(self.extensions.keys())

    def get_extension_metadata(self, extension_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific extension.

        Args:
            extension_id: Extension ID to look up

        Returns:
            Extension metadata or None if not found
        """
        if not self.is_available():
            return None
        return self.extensions.get(extension_id)

    def load_codeatlas(self, extension_id: str) -> Optional[Dict[str, Any]]:
        """Load Code Atlas data for a specific extension.

        Args:
            extension_id: Extension ID to load Code Atlas for

        Returns:
            Code Atlas data or None if not found
        """
        # Check cache first
        if extension_id in self._cached_codeatlas:
            return self._cached_codeatlas[extension_id]

        # Get extension metadata to find the file
        metadata = self.get_extension_metadata(extension_id)
        if not metadata:
            return None

        # Construct file path
        version = metadata.get("version", "")
        codeatlas_file = self.codeatlas_dir / f"{extension_id}-{version}.codeatlas.json"

        if not codeatlas_file.exists():
            # Try without version
            codeatlas_file = self.codeatlas_dir / f"{extension_id}.codeatlas.json"
            if not codeatlas_file.exists():
                logger.debug(f"Code Atlas file not found for {extension_id}")
                return None

        try:
            with open(codeatlas_file, "r", encoding="utf-8") as f:
                codeatlas_data = json.load(f)

            # Cache the loaded data
            self._cached_codeatlas[extension_id] = codeatlas_data
            logger.debug(f"Loaded Code Atlas for {extension_id}")
            return codeatlas_data

        except Exception as e:
            logger.error(f"Error loading Code Atlas for {extension_id}: {e}")
            return None

    def get_modules(self, extension_id: str = None) -> Dict[str, Any]:
        """Get modules information.

        Args:
            extension_id: Optional extension ID to filter by

        Returns:
            Dictionary containing module information
        """
        if not self.is_available():
            return {"error": "Extensions database is not available."}

        result = {
            "modules": [],
            "total_count": 0,
            "summary": {
                "total_modules": 0,
                "modules_with_classes": 0,
                "modules_with_functions": 0,
                "total_classes": 0,
                "total_methods": 0,
                "extensions": set(),
            },
        }

        # If specific extension requested
        if extension_id:
            codeatlas = self.load_codeatlas(extension_id)
            if not codeatlas:
                return {"error": f"No Code Atlas data found for extension '{extension_id}'"}

            modules = codeatlas.get("modules", {})
            for module_key, module_info in modules.items():
                module_data = {
                    "name": module_info.get("name", "Unknown"),
                    "full_name": module_info.get("full_name", module_key),
                    "file_path": module_info.get("file_path", ""),
                    "extension_name": extension_id,
                    "class_count": len(module_info.get("class_names", [])),
                    "function_count": len(module_info.get("function_names", [])),
                    "class_names": module_info.get("class_names", []),
                    "function_names": module_info.get("function_names", []),
                }
                result["modules"].append(module_data)
                result["summary"]["total_modules"] += 1
                result["summary"]["total_classes"] += module_data["class_count"]
                result["summary"]["extensions"].add(extension_id)

                if module_data["class_count"] > 0:
                    result["summary"]["modules_with_classes"] += 1
                if module_data["function_count"] > 0:
                    result["summary"]["modules_with_functions"] += 1

        else:
            # Get modules from all extensions
            for ext_id in self.get_extension_list():
                metadata = self.get_extension_metadata(ext_id)
                if metadata:
                    result["summary"]["total_modules"] += metadata.get("total_modules", 0)
                    result["summary"]["total_classes"] += metadata.get("total_classes", 0)
                    result["summary"]["total_methods"] += metadata.get("total_methods", 0)
                    result["summary"]["extensions"].add(ext_id)

        result["total_count"] = result["summary"]["total_modules"]
        result["summary"]["extensions"] = list(result["summary"]["extensions"])
        result["summary"]["unique_extensions"] = len(result["summary"]["extensions"])

        return result

    def get_classes(self, extension_id: str = None) -> Dict[str, Any]:
        """Get classes information.

        Args:
            extension_id: Optional extension ID to filter by

        Returns:
            Dictionary containing class information
        """
        if not self.is_available():
            return {"error": "Extensions database is not available."}

        result = {
            "classes": [],
            "total_count": 0,
            "summary": {
                "total_classes": 0,
                "classes_with_methods": 0,
                "total_methods": 0,
                "extensions": set(),
            },
        }

        # If specific extension requested
        if extension_id:
            codeatlas = self.load_codeatlas(extension_id)
            if not codeatlas:
                return {"error": f"No Code Atlas data found for extension '{extension_id}'"}

            classes = codeatlas.get("classes", {})
            for class_key, class_info in classes.items():
                methods = class_info.get("methods", [])
                class_data = {
                    "name": class_info.get("name", "Unknown"),
                    "full_name": class_info.get("full_name", class_key),
                    "module_name": class_info.get("module_name", ""),
                    "extension_name": extension_id,
                    "method_count": len(methods),
                    "methods": methods,
                    "docstring": class_info.get("docstring", ""),
                    "parent_classes": class_info.get("parent_classes", []),
                    "line_number": class_info.get("line_number"),
                }
                result["classes"].append(class_data)
                result["summary"]["total_classes"] += 1
                result["summary"]["total_methods"] += len(methods)
                result["summary"]["extensions"].add(extension_id)

                if methods:
                    result["summary"]["classes_with_methods"] += 1

        else:
            # Get summary from all extensions
            for ext_id in self.get_extension_list():
                metadata = self.get_extension_metadata(ext_id)
                if metadata:
                    result["summary"]["total_classes"] += metadata.get("total_classes", 0)
                    result["summary"]["total_methods"] += metadata.get("total_methods", 0)
                    result["summary"]["extensions"].add(ext_id)

        result["total_count"] = result["summary"]["total_classes"]
        result["summary"]["extensions"] = list(result["summary"]["extensions"])
        result["summary"]["unique_extensions"] = len(result["summary"]["extensions"])

        return result

    def get_class_detail(self, extension_id: str, class_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific class.

        Args:
            extension_id: Extension ID containing the class
            class_name: Name of the class to look up

        Returns:
            Detailed class information or None if not found
        """
        codeatlas = self.load_codeatlas(extension_id)
        if not codeatlas:
            return None

        classes = codeatlas.get("classes", {})

        # Try exact match first
        for class_key, class_info in classes.items():
            if class_info.get("name") == class_name or class_info.get("full_name") == class_name:
                return class_info

        # Try partial match
        for class_key, class_info in classes.items():
            if class_name in class_info.get("full_name", ""):
                return class_info

        return None

    def get_method_detail(self, extension_id: str, method_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific method.

        Args:
            extension_id: Extension ID containing the method
            method_name: Name of the method to look up

        Returns:
            Detailed method information or None if not found
        """
        codeatlas = self.load_codeatlas(extension_id)
        if not codeatlas:
            return None

        methods = codeatlas.get("methods", {})

        # Try exact match first
        for method_key, method_info in methods.items():
            if method_info.get("name") == method_name or method_info.get("full_name") == method_name:
                return method_info

        # Try partial match
        for method_key, method_info in methods.items():
            if method_name in method_info.get("full_name", ""):
                return method_info

        return None
