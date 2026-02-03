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
Function to retrieve detailed OmniUI class information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.telemetry import ensure_telemetry_initialized, telemetry
from ..utils import get_atlas_service

logger = logging.getLogger(__name__)


async def get_class_detail(class_names) -> Dict[str, Any]:
    """Get detailed information about one or more OmniUI classes.

    Args:
        class_names: List of class names to look up, or None to get available classes info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing detailed class information (single or array)
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "class_names": class_names,
        "is_none": class_names is None,
        "is_string": isinstance(class_names, str),
        "is_list": isinstance(class_names, list),
        "count": len(class_names) if isinstance(class_names, list) else (1 if isinstance(class_names, str) else 0),
    }

    success = True
    error_msg = None

    try:
        # Handle None input (get available classes info)
        if class_names is None:
            # Return information about available classes
            atlas_service = get_atlas_service()
            if not atlas_service.is_available():
                return {"success": False, "error": "OmniUI Atlas data is not available", "result": ""}

            # Get basic info about available classes
            available_classes = atlas_service.get_class_names_list()
            result = {
                "available_classes": available_classes,
                "total_count": len(available_classes),
                "usage": "Provide specific class names to get detailed information",
            }
            return {"success": True, "result": json.dumps(result, indent=2), "error": None}

        # Handle list input
        elif isinstance(class_names, list):
            if len(class_names) == 0:
                return {"success": False, "error": "class_names array cannot be empty", "result": ""}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(class_names) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"class_names contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }
            # Sanitize input: reject path traversal attempts and other suspicious patterns
            import re

            suspicious_patterns = [r"\.\./", r"\.\.", r"/etc/", r"\\", r"<script", r"javascript:"]
            for i, name in enumerate(class_names):
                for pattern in suspicious_patterns:
                    if re.search(pattern, name, re.IGNORECASE):
                        return {
                            "success": False,
                            "error": f"Invalid class name at index {i}: contains suspicious pattern. "
                            f"Class names should be valid Python identifiers like 'Button' or 'omni.ui.Button'.",
                            "result": "",
                        }
        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(class_names, str):
            if not class_names.strip():
                return {"success": False, "error": "class_name cannot be empty or whitespace", "result": ""}
            class_names = [class_names]
        else:
            actual_type = type(class_names).__name__
            return {
                "success": False,
                "error": f"class_names must be None or a list of strings, but got {actual_type}: {class_names}",
                "result": "",
            }

        logger.info(f"Retrieving detailed info for {len(class_names)} OmniUI class(es): {class_names}")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        results = []
        errors = []

        for class_name in class_names:
            try:
                # Get detailed class information
                class_detail = atlas_service.get_class_detail(class_name)

                if "error" in class_detail:
                    error_msg = f"{class_name}: {class_detail['error']}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    # Include error in results for partial success
                    results.append({"class_name": class_name, "error": class_detail["error"]})
                else:
                    results.append(class_detail)
                    logger.info(
                        f"Successfully retrieved details for class: {class_detail.get('full_name', class_name)}"
                    )

            except Exception as e:
                error_msg = f"{class_name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                results.append({"class_name": class_name, "error": str(e)})

        # Return single result if only one class was requested
        if len(class_names) == 1:
            if results and "error" not in results[0]:
                result_json = json.dumps(results[0], indent=2)
            else:
                return {"success": False, "error": errors[0] if errors else "Unknown error", "result": ""}
        else:
            # Return array of results for multiple classes
            result_json = json.dumps(
                {
                    "classes": results,
                    "total_requested": len(class_names),
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
        error_msg = f"Error retrieving class details: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_class_detail",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
