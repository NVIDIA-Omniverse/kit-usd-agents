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

"""Registration wrapper for search_app_examples function."""

import logging
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_app_examples import search_app_examples
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


class SearchKitAppTemplatesInput(BaseModel):
    """Input for search_app_examples function."""

    query: str = Field(
        description="""Search query for finding relevant Kit application templates. Examples:
        - "large scale visualization" - Find apps for visualizing large environments
        - "streaming cloud" - Find streaming-capable applications
        - "content creation" - Find authoring and editing applications
        - "factory warehouse" - Find industrial visualization apps
        - "collaboration" - Find apps with multi-user support"""
    )

    top_k: int = Field(default=5, description="Number of top results to return (default: 5)")

    category_filter: str = Field(
        default="",
        description="""Optional category filter to narrow results:
        - "editor" - Interactive editing applications
        - "authoring" - Content creation and authoring apps
        - "visualization" - Viewing and exploration apps
        - "streaming" - Cloud and streaming optimized apps
        - "configuration" - Configuration layers and settings""",
    )

    model_config = {"extra": "forbid"}


# Tool description
SEARCH_KIT_APP_TEMPLATES_DESCRIPTION = """Search for Kit application templates using semantic search to find the right starting point for your project.

🔍 INTELLIGENT SEARCH: Matches against descriptions, use cases, features, and categories

PARAMETERS:
- query: Natural language search query describing your needs
- top_k: Number of results to return (default: 5)
- category_filter: Optional category to narrow search

SEARCH EXAMPLES:
✓ "I need to visualize a large factory" → USD Explorer
✓ "streaming application for cloud" → USD Viewer
✓ "professional content creation" → USD Composer
✓ "simple 3D editor" → Kit Base Editor
✓ "collaborative design review" → Templates with collaboration

CATEGORIES:
- **editor**: Interactive 3D editing applications
- **authoring**: Professional content creation tools
- **visualization**: Large-scale viewing and exploration
- **streaming**: Cloud-optimized streaming applications
- **configuration**: Streaming configuration layers

RETURNS:
- Ranked list of matching templates with relevance scores
- Brief description and key features for each match
- Template IDs for retrieving full details
- Use case summaries to help selection

USAGE WORKFLOW:
1. Search for templates: search_app_examples("factory visualization")
2. Review results and scores
3. Get full details: get_app_examples("usd_explorer")
4. Access complete README and .kit configuration

SEARCH TIPS:
- Use specific keywords for better matches
- Combine terms like "large scale industrial"
- Filter by category for focused results
- Check streaming support in results

The search uses intelligent matching to understand:
- Technical requirements (streaming, collaboration)
- Scale needs (large environments, many assets)
- Use case patterns (authoring, viewing, editing)
- Industry terms (factory, warehouse, industrial)"""


class SearchKitAppTemplatesConfig(FunctionBaseConfig, name="search_kit_app_templates"):
    """Configuration for search_app_examples function."""

    name: str = "search_kit_app_templates"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=SearchKitAppTemplatesConfig, framework_wrappers=[])
async def register_search_kit_app_templates(config: SearchKitAppTemplatesConfig, builder: Builder):
    """Register search_app_examples function with AIQ."""

    verbose = config.verbose

    if verbose:
        logger.info("Registering search_kit_app_templates in verbose mode")

    async def search_kit_app_templates_wrapper(input: SearchKitAppTemplatesInput) -> str:
        """Search for Kit application templates."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        parameters = {"query": input.query, "top_k": input.top_k, "category_filter": input.category_filter}

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await search_app_examples(
                query=input.query, top_k=input.top_k, category_filter=input.category_filter
            )

            if verbose:
                logger.debug(f"Search for '{input.query}' returned {result.get('total_found', 0)} results")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to search app examples - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_kit_app_templates",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_app_examples: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        search_kit_app_templates_wrapper,
        description=SEARCH_KIT_APP_TEMPLATES_DESCRIPTION,
        input_schema=SearchKitAppTemplatesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
