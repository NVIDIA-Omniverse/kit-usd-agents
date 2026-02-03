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

"""Function to retrieve detailed Kit extension information."""

import json
import logging
import time
from typing import Any, Dict

from ..services.extension_service import ExtensionService
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)


def get_extension_service() -> ExtensionService:
    """Get or create the global Extension service instance.

    Returns:
        The Extension service instance
    """
    from .search_extensions import get_extension_service as _get_extension_service

    return _get_extension_service()


async def get_extension_details(extension_ids) -> Dict[str, Any]:
    """Get detailed information about one or more Kit extensions.

    Args:
        extension_ids: List of extension IDs to look up, or None to get available extensions info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing detailed extension information (single or array)
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
        # Handle None input (get available extensions info)
        if extension_ids is None:
            # Return information about available extensions
            extension_service = get_extension_service()
            if not extension_service.is_available():
                return {"success": False, "error": "Extension data is not available", "result": ""}

            # Get basic info about available extensions
            available_extensions = extension_service.get_extension_list()
            result = {
                "available_extensions": available_extensions,
                "total_count": len(available_extensions),
                "usage": "Provide specific extension IDs to get detailed information",
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

        logger.info(f"Retrieving detailed info for {len(extension_ids)} Kit extension(s): {extension_ids}")

        extension_service = get_extension_service()

        if not extension_service.is_available():
            error_msg = "Extension data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Get detailed extension information
        extension_details = extension_service.get_extension_details(extension_ids)

        # Return single result if only one extension was requested
        if len(extension_ids) == 1:
            if extension_details and "error" not in extension_details[0]:
                result_json = json.dumps(extension_details[0], indent=2)
            else:
                error = extension_details[0].get("error", "Unknown error") if extension_details else "No data returned"
                return {"success": False, "error": error, "result": ""}
        else:
            # Return array of results for multiple extensions
            successful_results = [r for r in extension_details if "error" not in r]
            failed_results = [r for r in extension_details if "error" in r]

            result_json = json.dumps(
                {
                    "extensions": extension_details,
                    "total_requested": len(extension_ids),
                    "successful": len(successful_results),
                    "failed": len(failed_results),
                },
                indent=2,
            )

        logger.info(
            f"Successfully retrieved details for {len([r for r in extension_details if 'error' not in r])} extensions"
        )

        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving extension details: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_extension_details",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
