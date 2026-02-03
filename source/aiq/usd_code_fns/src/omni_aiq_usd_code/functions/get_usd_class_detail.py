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

"""Get USD class detail function implementation."""

import json
import logging
import threading
from typing import Any, Dict, Optional

from ..services.usd_atlas import USDAtlasService
from ..utils.input_validation import InputValidationError, validate_list_items

logger = logging.getLogger(__name__)

# Global service - will be initialized on first use with thread safety
_usd_atlas_service: Optional[USDAtlasService] = None
_service_lock = threading.Lock()


def _initialize_service():
    """Initialize USD Atlas service if not already done.

    Uses double-checked locking pattern for thread-safe lazy initialization.
    """
    global _usd_atlas_service

    if _usd_atlas_service is None:
        with _service_lock:
            # Double-check after acquiring lock
            if _usd_atlas_service is None:
                _usd_atlas_service = USDAtlasService()
                if _usd_atlas_service.is_available():
                    logger.info("USD Atlas service initialized successfully")
                else:
                    logger.warning("USD Atlas data is not available")


async def get_usd_class_detail(class_names: str) -> Dict[str, Any]:
    """Return detailed information about specific USD classes, including their methods.

    Args:
        class_names: Comma-separated list of class names (e.g., "UsdStage,UsdPrim" or "pxr.Usd.Stage,pxr.Usd.Prim")

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: JSON string with detailed information about the classes including their methods and docstrings
        - error: Error message if operation failed
    """
    try:
        # Validate input
        try:
            class_list = validate_list_items(
                class_names,
                field_name="class_names",
                max_items=50,  # Limit number of classes per request
                max_item_length=500,  # Max length per class name
            )
        except InputValidationError as e:
            return {"success": False, "error": str(e), "result": ""}

        # Initialize service if needed
        _initialize_service()

        if not _usd_atlas_service or not _usd_atlas_service.is_available():
            error_msg = "USD Atlas data is not available. Please check if the data file is loaded correctly."
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Get details for each class
        results = {}
        for class_name in class_list:
            try:
                result = _usd_atlas_service.get_class_detail(class_name)
                results[class_name] = result
            except Exception as e:
                results[class_name] = {"error": f"Failed to retrieve details: {str(e)}"}

        result_json = json.dumps(results, indent=2)

        logger.info(f"Successfully retrieved details for {len(class_list)} classes")
        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving class details: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "result": ""}
