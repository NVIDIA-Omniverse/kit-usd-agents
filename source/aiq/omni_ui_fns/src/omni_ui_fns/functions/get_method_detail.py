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
Function to retrieve detailed OmniUI method information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.telemetry import ensure_telemetry_initialized, telemetry
from ..utils import get_atlas_service

logger = logging.getLogger(__name__)


async def get_method_detail(method_names) -> Dict[str, Any]:
    """Get detailed information about one or more OmniUI methods.

    Args:
        method_names: List of method names to look up, or None to get available methods info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing detailed method information (single or array)
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "method_names": method_names,
        "is_none": method_names is None,
        "is_string": isinstance(method_names, str),
        "is_list": isinstance(method_names, list),
        "count": len(method_names) if isinstance(method_names, list) else (1 if isinstance(method_names, str) else 0),
    }

    success = True
    error_msg = None

    try:
        # Handle None input (get available methods info)
        if method_names is None:
            # Return information about available methods
            atlas_service = get_atlas_service()
            if not atlas_service.is_available():
                return {"success": False, "error": "OmniUI Atlas data is not available", "result": ""}

            # Get basic info about available methods
            available_methods = atlas_service.get_method_list()
            result = {
                "available_methods": available_methods,
                "total_count": len(available_methods),
                "usage": "Provide specific method names to get detailed information",
            }
            return {"success": True, "result": json.dumps(result, indent=2), "error": None}

        # Handle list input
        elif isinstance(method_names, list):
            if len(method_names) == 0:
                return {"success": False, "error": "method_names array cannot be empty", "result": ""}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(method_names) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"method_names contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }
        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(method_names, str):
            if not method_names.strip():
                return {"success": False, "error": "method_name cannot be empty or whitespace", "result": ""}
            method_names = [method_names]
        else:
            actual_type = type(method_names).__name__
            return {
                "success": False,
                "error": f"method_names must be None or a list of strings, but got {actual_type}: {method_names}",
                "result": "",
            }

        logger.info(f"Retrieving detailed info for {len(method_names)} OmniUI method(s): {method_names}")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        results = []
        errors = []

        for method_name in method_names:
            try:
                # Get detailed method information
                method_detail = atlas_service.get_method_detail(method_name)

                if "error" in method_detail:
                    error_msg = f"{method_name}: {method_detail['error']}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    # Include error in results for partial success
                    results.append({"method_name": method_name, "error": method_detail["error"]})
                else:
                    results.append(method_detail)
                    logger.info(
                        f"Successfully retrieved details for method: {method_detail.get('full_name', method_name)}"
                    )

            except Exception as e:
                error_msg = f"{method_name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                results.append({"method_name": method_name, "error": str(e)})

        # Return single result if only one method was requested
        if len(method_names) == 1:
            if results and "error" not in results[0]:
                result_json = json.dumps(results[0], indent=2)
            else:
                return {"success": False, "error": errors[0] if errors else "Unknown error", "result": ""}
        else:
            # Return array of results for multiple methods
            result_json = json.dumps(
                {
                    "methods": results,
                    "total_requested": len(method_names),
                    "successful": len([r for r in results if "error" not in r]),
                    "failed": len([r for r in results if "error" in r]),
                },
                indent=2,
            )

        # Determine overall success (at least one succeeded)
        overall_success = any("error" not in r for r in results)

        if not overall_success:
            return {"success": False, "error": "; ".join(errors), "result": ""}

        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving method details: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_method_detail",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
