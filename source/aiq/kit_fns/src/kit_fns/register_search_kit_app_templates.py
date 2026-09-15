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

from kit_fns.utils._retrieval_compat import empty_result_sentinel as _empty_result_sentinel
from kit_fns.utils._retrieval_compat import sanitize_query as _sanitize_query
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_app_examples import search_app_examples
from .utils.usage_logging import get_usage_logger


# Audit R4/R5: strip CR/LF and cap length on user-provided strings before
# embedding them in log lines or MCP error responses, so a query with
# ``\r\n`` can't forge structured-log records or leak arbitrary text
# back to the caller. Not a sanitizer for search purposes — only for
# *rendering* into logs/errors.
def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


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
SEARCH_KIT_APP_TEMPLATES_DESCRIPTION = """Match a project description against the kit-app-template catalog to find the right application starting point.

WHEN TO USE THIS TOOL:
- "I want to build a Kit app that does X — which template should I start from?"
- Comparing USD Composer vs Explorer vs Viewer vs Base Editor for a use case.
- Looking for streaming / authoring / visualization starter kits.

ARGUMENTS:
- query (str): natural-language description of the project/use case.
- top_k (int, optional): number of templates to return (default 5).
- category_filter (str, optional): 'editor' | 'authoring' | 'visualization' | 'streaming' | 'configuration'.

RETURNS:
Ranked templates with relevance scores, short descriptions, key features, and template IDs. Follow up with get_kit_app_template_details for the full README + .kit file.

USAGE EXAMPLES:
search_kit_app_templates "large factory visualization"
search_kit_app_templates "streaming cloud viewer"
search_kit_app_templates "content authoring tool"

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- You already know the template ID and want the full README / .kit file → use get_kit_app_template_details.
- General "how do I build a Kit app?" question → use search_kit_knowledge.
- Looking for individual extensions rather than a whole app → use search_kit_extensions.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand."""


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
            # Sanitize user input before sending to external APIs
            sanitized_query = _sanitize_query(input.query)

            # Call the async function directly
            result = await search_app_examples(
                query=sanitized_query, top_k=input.top_k, category_filter=input.category_filter
            )

            if verbose:
                logger.debug(
                    f"Search for '{_sanitize_log(input.query)}' returned {result.get('total_found', 0)} results"
                )

            if result["success"]:
                text = result["result"] or ""
                if not text.strip():
                    return _empty_result_sentinel()
                return text
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {_sanitize_log(error_msg, cap=500)}"

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
