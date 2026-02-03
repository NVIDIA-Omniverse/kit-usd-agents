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

"""Function to retrieve detailed Kit API documentation."""

import json
import logging
import time
from typing import Any, Dict

from ..services.api_service import APIService
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)


def get_api_service() -> APIService:
    """Get or create the global API service instance.

    Returns:
        The API service instance
    """
    from .get_extension_apis import get_api_service as _get_api_service

    return _get_api_service()


async def get_api_details(api_references) -> Dict[str, Any]:
    """Get detailed documentation for specific APIs.

    Args:
        api_references: List of API references in format 'extension_id@symbol', or None to get available APIs info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing detailed API information (single or array)
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "api_references": api_references,
        "is_none": api_references is None,
        "is_string": isinstance(api_references, str),
        "is_list": isinstance(api_references, list),
        "count": (
            len(api_references) if isinstance(api_references, list) else (1 if isinstance(api_references, str) else 0)
        ),
    }

    success = True
    error_msg = None

    try:
        # Handle None input (get available APIs info)
        if api_references is None:
            # Return information about available API references
            api_service = get_api_service()
            if not api_service.is_available():
                return {"success": False, "error": "API data is not available", "result": ""}

            # Get basic info about available API references
            available_apis = api_service.get_api_list()
            result = {
                "available_api_references": available_apis,
                "total_count": len(available_apis),
                "usage": "Provide specific API references to get detailed information",
                "format": "Use format 'extension_id@symbol' (e.g., 'omni.ui@Window', 'omni.ui@Button')",
            }
            return {"success": True, "result": json.dumps(result, indent=2), "error": None}

        # Handle list input
        elif isinstance(api_references, list):
            if len(api_references) == 0:
                return {"success": False, "error": "api_references array cannot be empty", "result": ""}
            # Check for empty strings in the list
            empty_refs = [i for i, ref in enumerate(api_references) if not isinstance(ref, str) or not ref.strip()]
            if empty_refs:
                return {
                    "success": False,
                    "error": f"api_references contains empty or non-string values at indices: {empty_refs}",
                    "result": "",
                }
            # Validate format
            for i, ref in enumerate(api_references):
                if "@" not in ref:
                    return {
                        "success": False,
                        "error": f"Invalid API reference format at index {i}: '{ref}'. Use format 'extension_id@symbol'",
                        "result": "",
                    }
        # Handle string input
        elif isinstance(api_references, str):
            if not api_references.strip():
                return {"success": False, "error": "api_reference cannot be empty or whitespace", "result": ""}
            if "@" not in api_references:
                return {
                    "success": False,
                    "error": f"Invalid API reference format: '{api_references}'. Use format 'extension_id@symbol'",
                    "result": "",
                }
            api_references = [api_references]
        else:
            actual_type = type(api_references).__name__
            return {
                "success": False,
                "error": f"api_references must be None or a list of strings, but got {actual_type}: {api_references}",
                "result": "",
            }

        logger.info(f"Retrieving detailed info for {len(api_references)} Kit API(s): {api_references}")

        api_service = get_api_service()

        if not api_service.is_available():
            error_msg = "API data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Get detailed API information
        api_details = api_service.get_api_details(api_references)

        # Return single result if only one API was requested
        if len(api_references) == 1:
            if api_details and "error" not in api_details[0]:
                result_json = json.dumps(api_details[0], indent=2)
            else:
                error = api_details[0].get("error", "Unknown error") if api_details else "No data returned"
                return {"success": False, "error": error, "result": ""}
        else:
            # Return array of results for multiple APIs
            successful_results = [r for r in api_details if "error" not in r]
            failed_results = [r for r in api_details if "error" in r]

            result_json = json.dumps(
                {
                    "apis": api_details,
                    "total_requested": len(api_references),
                    "successful": len(successful_results),
                    "failed": len(failed_results),
                },
                indent=2,
            )

        logger.info(f"Successfully retrieved details for {len([r for r in api_details if 'error' not in r])} APIs")

        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving API details: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_api_details",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
