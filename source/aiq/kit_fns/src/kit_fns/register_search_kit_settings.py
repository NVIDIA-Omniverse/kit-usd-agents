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

"""Registration wrapper for search_settings function."""

import logging
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_settings import search_settings
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


# Define input schema for search_settings function
class SearchKitSettingsInput(BaseModel):
    """Input for search_settings function."""

    query: str = Field(description="Search query for finding relevant Kit configuration settings")
    top_k: int = Field(default=20, description="Number of setting results to return (default: 20)")
    prefix_filter: Optional[str] = Field(
        None,
        description="Filter by setting prefix: 'exts' (extension settings), 'app' (application settings), 'persistent' (saved settings), 'rtx' (rendering settings)",
    )
    type_filter: Optional[str] = Field(
        None, description="Filter by setting type: 'bool', 'int', 'float', 'string', 'array', 'object'"
    )


# Tool description
SEARCH_KIT_SETTINGS_DESCRIPTION = """Search for NVIDIA Omniverse Kit configuration settings using semantic search across 1,000+ settings from 400+ extensions.

WHAT IT DOES:
- Searches through Kit's hierarchical settings system (Carbonite settings)
- Uses semantic understanding to find relevant configuration options
- Provides setting metadata including types, defaults, and documentation
- Shows which extensions use each setting
- Supports filtering by prefix and type

SETTING PREFIXES:
Kit settings follow a path-based structure with common prefixes:
- /exts/: Extension-specific settings (e.g., /exts/omni.kit.viewport.window/enabled)
- /app/: Application-level settings (e.g., /app/window/title)
- /persistent/: Settings saved between sessions (e.g., /persistent/app/viewport/camMoveVelocity)
- /rtx/: RTX rendering settings (e.g., /rtx/rendermode)
- /renderer/: General renderer settings
- /physics/: Physics simulation settings

SEARCH CAPABILITIES:
Your query is matched against:
- Setting paths and names
- Documentation and descriptions
- Extension names that use the settings
- Setting types and categories

ARGUMENTS:
- query (str): Natural language search query describing desired settings
- top_k (int, optional): Number of results to return (default: 20)
- prefix_filter (str, optional): Filter by setting prefix ('exts', 'app', 'persistent', 'rtx')
- type_filter (str, optional): Filter by type ('bool', 'int', 'float', 'string', 'array', 'object')

RETURNS:
Formatted search results with:
- Full setting paths
- Data types and default values
- Documentation (when available)
- Extensions using each setting
- Usage counts across codebase
- Source file locations

USAGE EXAMPLES:
search_settings("viewport rendering settings")
search_settings("enable debug mode", type_filter="bool")
search_settings("window configuration", prefix_filter="app")
search_settings("saved user preferences", prefix_filter="persistent")
search_settings("ray tracing quality", prefix_filter="rtx")

TIPS FOR BETTER RESULTS:
- Use specific terminology (e.g., "viewport", "timeline", "grid", "fps")
- Include setting purpose (e.g., "enable", "disable", "configure", "adjust")
- Specify data types when known (e.g., "boolean flags", "integer limits")
- Use prefix filters to narrow down to specific categories
- Search for extension names to find their settings"""


class SearchKitSettingsConfig(FunctionBaseConfig, name="search_kit_settings"):
    """Configuration for search_settings function."""

    name: str = "search_kit_settings"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=SearchKitSettingsConfig, framework_wrappers=[])
async def register_search_kit_settings(config: SearchKitSettingsConfig, builder: Builder):
    """Register search_settings function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering search_kit_settings in verbose mode")

    async def search_kit_settings_wrapper(input: SearchKitSettingsInput) -> str:
        """Search for Kit configuration settings using semantic search."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {
            "query": input.query,
            "top_k": input.top_k,
            "prefix_filter": input.prefix_filter,
            "type_filter": input.type_filter,
        }
        error_msg = None
        success = True

        try:
            # Sanitize user input before sending to external APIs
            from kit_fns.utils.input_sanitization import sanitize_query

            sanitized_query = sanitize_query(input.query)

            # Call the async function directly
            result = await search_settings(
                query=sanitized_query,
                top_k=input.top_k or 20,
                prefix_filter=input.prefix_filter,
                type_filter=input.type_filter,
            )

            # Use config fields to modify behavior
            if verbose:
                logger.debug(
                    f"Searched settings for query: '{input.query}', "
                    f"top_k: {input.top_k}, prefix: {input.prefix_filter}, type: {input.type_filter}"
                )

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to search settings - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_kit_settings",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_settings: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        search_kit_settings_wrapper,
        description=SEARCH_KIT_SETTINGS_DESCRIPTION,
        input_schema=SearchKitSettingsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
