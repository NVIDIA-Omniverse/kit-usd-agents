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

"""Function to search for Kit settings using semantic search."""

import logging
import time
from typing import Any, Dict, Optional

from ..services.settings_service import SettingsService
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Global settings service instance
_settings_service = None


def get_settings_service() -> SettingsService:
    """Get or create the global Settings service instance.

    Returns:
        The Settings service instance
    """
    global _settings_service
    if _settings_service is None:
        _settings_service = SettingsService()
    return _settings_service


async def search_settings(
    query: str, top_k: int = 20, prefix_filter: Optional[str] = None, type_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Search for Kit configuration settings using semantic search.

    This function searches through over 1,000 Kit settings from 400+ extensions,
    using semantic understanding to find relevant configuration options.

    Args:
        query: Search query describing the settings you're looking for
        top_k: Number of results to return (default: 20)
        prefix_filter: Filter by setting prefix ('exts', 'app', 'persistent', 'rtx')
        type_filter: Filter by setting type ('bool', 'int', 'float', 'string', 'array', 'object')

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: Formatted search results with settings and metadata
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {"query": query, "top_k": top_k, "prefix_filter": prefix_filter, "type_filter": type_filter}

    success = True
    error_msg = None

    try:
        logger.info(f"Searching Kit settings with query: '{query}'")

        settings_service = get_settings_service()

        if not settings_service.is_available():
            error_msg = "Settings data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Perform the search
        search_results = settings_service.search_settings(
            query, top_k, prefix_filter=prefix_filter, type_filter=type_filter
        )

        if not search_results:
            no_result_msg = f"No settings found for query: '{query}'"
            if prefix_filter:
                no_result_msg += f" with prefix filter: {prefix_filter}"
            if type_filter:
                no_result_msg += f" with type filter: {type_filter}"
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

        # Format the results
        result_lines = []
        result_lines.append(f"**Found {len(search_results)} matching settings:**\n")

        for i, setting in enumerate(search_results, 1):
            result_lines.append(f"## {i}. `{setting['setting']}`")

            # Basic info
            result_lines.append(f"**Type:** {setting.get('type', 'unknown')}")
            result_lines.append(f"**Default:** `{setting.get('default_value', 'None')}`")
            result_lines.append(f"**Relevance Score:** {setting.get('relevance_score', 0):.2f}")

            # Documentation
            documentation = setting.get("documentation", "")
            description = setting.get("description", "")

            if documentation:
                result_lines.append(f"**Documentation:** {documentation}")
            elif description:
                result_lines.append(f"**Description:** {description}")
            else:
                result_lines.append("**Documentation:** *No documentation available*")

            # Extensions using this setting
            extensions = setting.get("extensions", [])
            if extensions:
                ext_list = ", ".join(extensions[:3])
                if len(extensions) > 3:
                    ext_list += f" (and {len(extensions) - 3} more)"
                result_lines.append(f"**Used by:** {ext_list}")

            # Usage info
            usage_count = setting.get("usage_count", 0)
            if usage_count > 1:
                result_lines.append(f"**Usage count:** {usage_count} times across extensions")

            # Found in locations (for reference)
            found_in = setting.get("found_in", [])
            if found_in:
                locations = ", ".join(found_in[:2])
                if len(found_in) > 2:
                    locations += f" (and {len(found_in) - 2} more)"
                result_lines.append(f"**Found in:** {locations}")

            result_lines.append("---\n")

        # Add summary
        result_lines.append("\n*Use get_setting_details to get complete information about specific settings.*")

        if prefix_filter:
            result_lines.append(f"*Filtered by prefix: {prefix_filter}*")
        if type_filter:
            result_lines.append(f"*Filtered by type: {type_filter}*")

        formatted_result = "\n".join(result_lines)

        logger.info(f"Successfully found {len(search_results)} settings for query: '{query}'")
        return {"success": True, "result": formatted_result, "error": None}

    except Exception as e:
        error_msg = f"Error searching settings: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="search_settings",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
