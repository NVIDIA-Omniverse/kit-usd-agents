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

"""Function to search for Kit extensions using semantic search."""

import logging
import time
from typing import Any, Dict

from ..services.extension_service import ExtensionService
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Global extension service instance
_extension_service = None


def get_extension_service() -> ExtensionService:
    """Get or create the global Extension service instance.

    Returns:
        The Extension service instance
    """
    global _extension_service
    if _extension_service is None:
        _extension_service = ExtensionService()
    return _extension_service


async def search_extensions(query: str, top_k: int = 20) -> Dict[str, Any]:
    """Search for Kit extensions using semantic search.

    Args:
        query: Search query for finding relevant extensions
        top_k: Number of results to return (default: 20)

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: Formatted search results with extensions and scores
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {"query": query, "top_k": top_k}

    success = True
    error_msg = None

    try:
        logger.info(f"Searching Kit extensions with query: '{query}'")

        extension_service = get_extension_service()

        if not extension_service.is_available():
            error_msg = "Extension data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Perform the search
        search_results = extension_service.search_extensions(query, top_k)

        if not search_results:
            no_result_msg = f"No extensions found for query: '{query}'"
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

        # Format the results
        result_lines = [f"# Kit Extension Search Results for: '{query}'"]
        result_lines.append(f"\n**Found {len(search_results)} matching extensions:**\n")

        for i, ext in enumerate(search_results, 1):
            result_lines.append(f"## {i}. {ext['id']}")
            result_lines.append(f"**Version:** {ext.get('version', 'Unknown')}")
            result_lines.append(f"**Relevance Score:** {ext.get('relevance_score', 0):.2f}")
            result_lines.append(f"**Description:** {ext.get('description', 'No description available')}")

            # Show key features
            features = ext.get("features", [])
            if features:
                result_lines.append("**Key Features:**")
                for feature in features[:3]:  # Limit to top 3 features
                    result_lines.append(f"  - {feature}")

            # Show dependencies
            deps = ext.get("dependencies", [])
            if deps:
                result_lines.append(f"**Dependencies:** {', '.join(deps[:3])}")
                if len(deps) > 3:
                    result_lines.append(f"  *(and {len(deps)-3} more)*")

            result_lines.append("---\n")

        # Add summary
        result_lines.append(f"*Use get_extension_details to get complete information about specific extensions.*")

        formatted_result = "\n".join(result_lines)

        logger.info(f"Successfully found {len(search_results)} extensions for query: '{query}'")
        return {"success": True, "result": formatted_result, "error": None}

    except Exception as e:
        error_msg = f"Error searching extensions: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="search_extensions",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
