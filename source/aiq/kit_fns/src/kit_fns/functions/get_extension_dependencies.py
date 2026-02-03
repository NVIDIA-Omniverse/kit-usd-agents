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

"""Function to analyze Kit extension dependencies."""

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


async def get_extension_dependencies(
    extension_id: str, depth: int = 2, include_optional: bool = False
) -> Dict[str, Any]:
    """Analyze and visualize extension dependency graphs.

    Args:
        extension_id: Extension ID to analyze dependencies for
        depth: Dependency tree depth to explore (default: 2)
        include_optional: Include optional dependencies (default: False)

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing dependency tree information
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {"extension_id": extension_id, "depth": depth, "include_optional": include_optional}

    success = True
    error_msg = None

    try:
        logger.info(f"Analyzing dependencies for extension: {extension_id}")

        extension_service = get_extension_service()

        if not extension_service.is_available():
            error_msg = "Extension data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Validate inputs
        if not extension_id or not extension_id.strip():
            error_msg = "extension_id cannot be empty"
            return {"success": False, "error": error_msg, "result": ""}

        if depth < 0:
            error_msg = "depth must be non-negative"
            return {"success": False, "error": error_msg, "result": ""}

        # Get dependency analysis
        dependency_info = extension_service.get_extension_dependencies(
            extension_id=extension_id.strip(), depth=depth, include_optional=include_optional
        )

        if "error" in dependency_info:
            error_msg = dependency_info["error"]
            logger.warning(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Format the result as JSON
        result_json = json.dumps(dependency_info, indent=2)

        logger.info(f"Successfully analyzed dependencies for extension: {extension_id}")
        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error analyzing extension dependencies: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_extension_dependencies",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
