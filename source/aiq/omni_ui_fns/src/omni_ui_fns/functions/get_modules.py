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

"""
Function to retrieve OmniUI module information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.telemetry import ensure_telemetry_initialized, telemetry
from ..utils import get_atlas_service

logger = logging.getLogger(__name__)


async def get_modules() -> Dict[str, Any]:
    """Return a list of all OmniUI module names from the Atlas data.

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing all OmniUI module names
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data (no parameters for this function)
    telemetry_data = {}

    success = True
    error_msg = None

    try:
        logger.info("Retrieving OmniUI modules from Atlas data")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Get the list of module names
        module_names = atlas_service.get_module_names_list()

        # Return the simplified result with just module names
        simplified_result = {
            "module_names": module_names,
            "total_count": len(module_names),
            "description": "OmniUI modules from Atlas data",
        }

        result_json = json.dumps(simplified_result, indent=2)

        logger.info(f"Successfully retrieved {len(module_names)} OmniUI modules from Atlas")
        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving OmniUI modules: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_modules",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
