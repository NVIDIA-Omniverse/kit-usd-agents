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

"""Function to retrieve APIs provided by Kit extensions."""

import json
import logging
import time
from typing import Any, Dict

from ..services.api_service import APIService
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Global API service instance
_api_service = None


def get_api_service() -> APIService:
    """Get or create the global API service instance.

    Returns:
        The API service instance
    """
    global _api_service
    if _api_service is None:
        _api_service = APIService()
    return _api_service


async def get_extension_apis(extension_ids) -> Dict[str, Any]:
    """List all APIs provided by specified extensions.

    Args:
        extension_ids: List of extension IDs to get APIs for, or None to get available extensions info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing API listings grouped by extension
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "extension_ids": extension_ids,
        "is_none": extension_ids is None,
        "is_string": isinstance(extension_ids, str),
        "is_list": isinstance(extension_ids, list),
        "count": (
            len(extension_ids) if isinstance(extension_ids, list) else (1 if isinstance(extension_ids, str) else 0)
        ),
    }

    success = True
    error_msg = None

    try:
        # Handle None input (get available APIs info)
        if extension_ids is None:
            # Return information about available API references
            api_service = get_api_service()
            if not api_service.is_available():
                return {"success": False, "error": "API data is not available", "result": ""}

            # Get basic info about available APIs
            available_apis = api_service.get_api_list()
            result = {
                "available_api_references": available_apis,
                "total_count": len(available_apis),
                "usage": "Provide specific extension IDs to get their APIs",
                "format": "Use format 'extension_id@symbol' for detailed API information",
            }
            return {"success": True, "result": json.dumps(result, indent=2), "error": None}

        # Handle list input
        elif isinstance(extension_ids, list):
            if len(extension_ids) == 0:
                return {"success": False, "error": "extension_ids array cannot be empty", "result": ""}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(extension_ids) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"extension_ids contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }
        # Handle string input
        elif isinstance(extension_ids, str):
            if not extension_ids.strip():
                return {"success": False, "error": "extension_id cannot be empty or whitespace", "result": ""}
            extension_ids = [extension_ids]
        else:
            actual_type = type(extension_ids).__name__
            return {
                "success": False,
                "error": f"extension_ids must be None or a list of strings, but got {actual_type}: {extension_ids}",
                "result": "",
            }

        logger.info(f"Retrieving APIs for {len(extension_ids)} Kit extension(s): {extension_ids}")

        api_service = get_api_service()

        if not api_service.is_available():
            error_msg = "API data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Get API information for extensions
        api_results = api_service.get_extension_apis(extension_ids)

        # Return single result if only one extension was requested
        if len(extension_ids) == 1:
            if api_results and len(api_results) > 0:
                result_json = json.dumps(api_results[0], indent=2)
            else:
                return {"success": False, "error": f"No APIs found for extension: {extension_ids[0]}", "result": ""}
        else:
            # Return array of results for multiple extensions
            result_json = json.dumps(
                {
                    "extensions": api_results,
                    "total_requested": len(extension_ids),
                    "extensions_with_apis": len([r for r in api_results if r.get("api_count", 0) > 0]),
                    "total_apis": sum(r.get("api_count", 0) for r in api_results),
                },
                indent=2,
            )

        logger.info(f"Successfully retrieved APIs for {len(extension_ids)} extensions")

        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving extension APIs: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_extension_apis",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
