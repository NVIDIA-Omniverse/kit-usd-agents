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
Function to retrieve detailed OmniUI module information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.telemetry import ensure_telemetry_initialized, telemetry
from ..utils import get_atlas_service

logger = logging.getLogger(__name__)


async def get_module_detail(module_names) -> Dict[str, Any]:
    """Get detailed information about one or more OmniUI modules.

    Args:
        module_names: List of module names to look up, or None to get available modules info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing detailed module information (single or array)
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "module_names": module_names,
        "is_none": module_names is None,
        "is_string": isinstance(module_names, str),
        "is_list": isinstance(module_names, list),
        "count": len(module_names) if isinstance(module_names, list) else (1 if isinstance(module_names, str) else 0),
    }

    success = True
    error_msg = None

    try:
        # Handle None input (get available modules info)
        if module_names is None:
            # Return information about available modules
            atlas_service = get_atlas_service()
            if not atlas_service.is_available():
                return {"success": False, "error": "OmniUI Atlas data is not available", "result": ""}

            # Get basic info about available modules
            available_modules = atlas_service.get_module_names_list()
            result = {
                "available_modules": available_modules,
                "total_count": len(available_modules),
                "usage": "Provide specific module names to get detailed information",
            }
            return {"success": True, "result": json.dumps(result, indent=2), "error": None}

        # Handle list input
        elif isinstance(module_names, list):
            if len(module_names) == 0:
                return {"success": False, "error": "module_names array cannot be empty", "result": ""}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(module_names) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"module_names contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }
            # Sanitize input: reject path traversal attempts and other suspicious patterns
            import re

            suspicious_patterns = [r"\.\./", r"\.\.", r"/etc/", r"\\", r"<script", r"javascript:"]
            for i, name in enumerate(module_names):
                for pattern in suspicious_patterns:
                    if re.search(pattern, name, re.IGNORECASE):
                        return {
                            "success": False,
                            "error": f"Invalid module name at index {i}: contains suspicious pattern. "
                            f"Module names should be valid Python module identifiers like 'omni.ui'.",
                            "result": "",
                        }
        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(module_names, str):
            if not module_names.strip():
                return {"success": False, "error": "module_name cannot be empty or whitespace", "result": ""}
            module_names = [module_names]
        else:
            actual_type = type(module_names).__name__
            return {
                "success": False,
                "error": f"module_names must be None or a list of strings, but got {actual_type}: {module_names}",
                "result": "",
            }

        logger.info(f"Retrieving detailed info for {len(module_names)} OmniUI module(s): {module_names}")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        results = []
        errors = []

        for module_name in module_names:
            try:
                # Get detailed module information
                module_detail = atlas_service.get_module_detail(module_name)

                if "error" in module_detail:
                    error_msg = f"{module_name}: {module_detail['error']}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    # Include error in results for partial success
                    results.append({"module_name": module_name, "error": module_detail["error"]})
                else:
                    results.append(module_detail)
                    logger.info(
                        f"Successfully retrieved details for module: {module_detail.get('full_name', module_name)}"
                    )

            except Exception as e:
                error_msg = f"{module_name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                results.append({"module_name": module_name, "error": str(e)})

        # Return single result if only one module was requested
        if len(module_names) == 1:
            if results and "error" not in results[0]:
                result_json = json.dumps(results[0], indent=2)
            else:
                return {"success": False, "error": errors[0] if errors else "Unknown error", "result": ""}
        else:
            # Return array of results for multiple modules
            result_json = json.dumps(
                {
                    "modules": results,
                    "total_requested": len(module_names),
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
        error_msg = f"Error retrieving module details: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_module_detail",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
