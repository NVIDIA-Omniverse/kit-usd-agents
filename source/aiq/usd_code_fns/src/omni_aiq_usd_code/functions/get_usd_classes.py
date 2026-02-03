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

"""Get USD classes function implementation."""

import json
import logging
import threading
from typing import Any, Dict, Optional

from ..services.usd_atlas import USDAtlasService

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


async def get_usd_classes() -> Dict[str, Any]:
    """Return a list of all USD class full names from the USD Atlas data.

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing all USD class full names
        - error: Error message if operation failed
    """
    try:
        # Initialize service if needed
        _initialize_service()

        if not _usd_atlas_service or not _usd_atlas_service.is_available():
            error_msg = "USD Atlas data is not available. Please check if the data file is loaded correctly."
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        result = _usd_atlas_service.get_classes()

        # Check if there's an error in the result
        if "error" in result:
            return {"success": False, "error": result["error"], "result": ""}

        # Extract just the full names from the classes
        class_full_names = []
        for class_data in result.get("classes", []):
            full_name = class_data.get("full_name", "")
            if full_name:
                class_full_names.append(full_name)

        # Return the simplified result with just full names
        simplified_result = {"class_full_names": sorted(class_full_names), "total_count": len(class_full_names)}

        result_json = json.dumps(simplified_result, indent=2)

        logger.info(f"Successfully retrieved {len(class_full_names)} USD classes")
        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving USD classes: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "result": ""}
