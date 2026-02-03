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

"""Registration wrapper for search_extensions function."""

import logging
from typing import List, Optional

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_extensions import search_extensions
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


# Define input schema for search_extensions function
class SearchKitExtensionsInput(BaseModel):
    """Input for search_extensions function."""

    query: str = Field(description="Search query for finding relevant Kit extensions")
    top_k: int = Field(default=10, description="Number of extension results to return (default: 10)")
    # categories: List[str] = Field(
    #     None,
    #     description="Filter by extension categories (ui, rendering, physics, development, usd, etc.)"
    # )


# Tool description
SEARCH_KIT_EXTENSIONS_DESCRIPTION = """Search for Kit extensions using semantic search across 400+ available extensions.

WHAT IT DOES:
- Converts your query to embeddings for semantic understanding
- Searches across Kit extension metadata, descriptions, and features
- Ranks results by relevance to your specific needs
- Supports category filtering for targeted searches
- Returns extensions with descriptions, features, and dependencies

SEARCH CAPABILITIES:
Your query is matched against:
- Extension names and descriptions
- Feature lists and capabilities
- Category classifications
- Dependency information
- Use case descriptions

ARGUMENTS:
- query (str): Search query describing what you're looking for
- top_k (int, optional): Number of results to return (default: 10)

RETURNS:
Formatted search results with:
- Extension names and IDs
- Relevance scores
- Brief descriptions
- Key features (top 3)
- Dependencies
- Version information

USAGE EXAMPLES:
search_extensions("window management tools", top_k=10)
search_extensions("ui widgets and controls", top_k=5)
search_extensions("physics simulation", top_k=10)
search_extensions("viewport and camera", top_k=3)

TIPS FOR BETTER RESULTS:
- Use specific terminology (e.g., "viewport", "console", "manipulator")
- Include functionality keywords (e.g., "window", "widget", "render", "physics")
- Specify use cases (e.g., "debugging tools", "ui components", "scene editing")
- Use category filters to narrow results to relevant domains"""


class SearchKitExtensionsConfig(FunctionBaseConfig, name="search_kit_extensions"):
    """Configuration for search_extensions function."""

    name: str = "search_kit_extensions"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=SearchKitExtensionsConfig, framework_wrappers=[])
async def register_search_kit_extensions(config: SearchKitExtensionsConfig, builder: Builder):
    """Register search_extensions function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering search_kit_extensions in verbose mode")

    async def search_kit_extensions_wrapper(input: SearchKitExtensionsInput) -> str:
        """Search for Kit extensions using semantic search."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {
            "query": input.query,
            "top_k": input.top_k,
        }
        error_msg = None
        success = True

        try:
            # Sanitize user input before sending to external APIs
            from kit_fns.utils.input_sanitization import sanitize_query

            sanitized_query = sanitize_query(input.query)

            # Call the async function directly
            result = await search_extensions(
                query=sanitized_query,
                top_k=input.top_k or 10,
            )

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Searched extensions for query: '{input.query}', top_k: {input.top_k}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to search extensions - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_kit_extensions",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_extensions: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        search_kit_extensions_wrapper,
        description=SEARCH_KIT_EXTENSIONS_DESCRIPTION,
        input_schema=SearchKitExtensionsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
