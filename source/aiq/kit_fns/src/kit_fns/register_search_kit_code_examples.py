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

"""Registration wrapper for search_code_examples function."""

import logging

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_code_examples import search_code_examples
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


class SearchKitCodeExamplesInput(BaseModel):
    """Input for search_code_examples function."""

    query: str = Field(description="Description of desired Kit code functionality")
    top_k: int = Field(default=10, description="Number of code examples to return")


# Tool description
SEARCH_KIT_CODE_EXAMPLES_DESCRIPTION = """Find relevant Kit code examples using semantic search and optional reranking.

WHAT IT DOES:
- Searches across curated Kit code examples and implementations
- Uses semantic matching to find relevant code patterns
- Optionally reranks results using NVIDIA's reranking model for improved relevance
- Returns complete code examples with context and file paths
- Includes implementation details and usage patterns

SEARCH CAPABILITIES:
Your query is matched against:
- Extension implementations and patterns
- UI component creation examples
- USD operation examples
- Widget usage and styling patterns
- Layout and container implementations
- Event handling and callback patterns

ARGUMENTS:
- query (str): Description of desired code functionality
- top_k (int, optional): Number of code examples to return (default: 10)

RETURNS:
Formatted code examples with:
- Complete implementation code
- File paths and line numbers
- Extension IDs and context
- Descriptions and use cases
- Relevance scores
- Associated tags

USAGE EXAMPLES:
search_code_examples("create window with buttons")
search_code_examples("USD stage operations", top_k=5)
search_code_examples("ui layout containers")
search_code_examples("extension lifecycle management")
search_code_examples("3d scene ui elements")

TIPS FOR BETTER RESULTS:
- Use Kit-specific terminology (e.g., "extension", "omni.ui", "USD stage")
- Include component types (e.g., "window", "button", "layout", "viewport")
- Reference patterns (e.g., "lifecycle", "callback", "event handling")
- Specify frameworks (e.g., "omni.ui", "omni.usd", "omni.kit")"""


class SearchKitCodeExamplesConfig(FunctionBaseConfig, name="search_kit_code_examples"):
    """Configuration for search_code_examples function."""

    name: str = "search_kit_code_examples"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    enable_rerank: bool = Field(default=True, description="Whether to enable reranking for improved relevance")


@register_function(config_type=SearchKitCodeExamplesConfig, framework_wrappers=[])
async def register_search_kit_code_examples(config: SearchKitCodeExamplesConfig, builder: Builder):
    """Register search_code_examples function with AIQ."""

    verbose = config.verbose

    if verbose:
        logger.info(f"Registering search_kit_code_examples in verbose mode")

    async def search_kit_code_examples_wrapper(input: SearchKitCodeExamplesInput) -> str:
        """Search for Kit code examples."""
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

            result = await search_code_examples(
                query=sanitized_query,
                rerank_k=input.top_k,
                enable_rerank=config.enable_rerank,
            )

            if verbose:
                logger.debug(f"Searched code examples for: '{input.query}', top_k: {input.top_k}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to search code examples - {error_msg}"
        finally:
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_kit_code_examples",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_code_examples: {log_error}")

    function_info = FunctionInfo.from_fn(
        search_kit_code_examples_wrapper,
        description=SEARCH_KIT_CODE_EXAMPLES_DESCRIPTION,
        input_schema=SearchKitCodeExamplesInput,
    )

    function_info.metadata = {"mcp_exposed": True}

    yield function_info
