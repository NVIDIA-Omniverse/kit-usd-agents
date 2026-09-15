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

from kit_fns.utils._retrieval_compat import empty_result_sentinel as _empty_result_sentinel
from kit_fns.utils._retrieval_compat import sanitize_query
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_extensions import search_extensions
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
SEARCH_KIT_EXTENSIONS_DESCRIPTION = """PRIMARY tool for discovering "is there a Kit extension that does X?". Semantic search across the Kit extension catalog.

WHEN TO USE THIS TOOL:
- "Is there an extension that provides <feature>?"
- "Which Kit extension handles viewport / console / manipulators / physics?"
- Comparing several extensions to pick the right one for a use case.

ARGUMENTS:
- query (str): natural-language description of the desired capability.
- top_k (int, optional): number of results to return (default 10).

RETURNS:
Ranked extensions with names, IDs, relevance scores, short descriptions, key features, dependencies, and version info.

USAGE EXAMPLES:
search_kit_extensions "window management tools"
search_kit_extensions "ui widgets and controls"
search_kit_extensions "viewport and camera"

COVERAGE CAVEAT:
Coverage is limited to extensions shipped with recent Kit SDK builds (Kit 105.1–110.x). Vendor or lab-only extensions may be missing — fall back to `search_kit_knowledge` for conceptual questions.

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- You already know the extension name and want full metadata → use get_kit_extension_details.
- You need the extension's API surface → use get_kit_extension_apis.
- You need the extension's dependency tree → use get_kit_extension_dependencies.
- General "how does Kit X work?" → use search_kit_knowledge.
- Isaac Sim extension discovery → use the Isaac Sim MCP's search_isaac_sim_extensions.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand."""


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

            sanitized_query = sanitize_query(input.query)

            # Call the async function directly
            result = await search_extensions(
                query=sanitized_query,
                top_k=input.top_k or 10,
            )

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Searched extensions for query: '{_sanitize_log(input.query)}', top_k: {input.top_k}")

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
