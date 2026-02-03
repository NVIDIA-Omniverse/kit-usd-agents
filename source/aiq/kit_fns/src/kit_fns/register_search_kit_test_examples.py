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

"""Registration wrapper for search_test_examples function."""

import logging

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_test_examples import search_test_examples
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


class SearchKitTestExamplesInput(BaseModel):
    """Input for search_test_examples function."""

    query: str = Field(description="Test scenario or functionality to find examples for")
    top_k: int = Field(default=10, description="Number of test examples to return")


# Tool description
SEARCH_KIT_TEST_EXAMPLES_DESCRIPTION = """Find Kit test implementations and patterns using semantic search and optional reranking.

WHAT IT DOES:
- Searches across Kit test code and testing patterns
- Uses semantic matching to find relevant test implementations
- Optionally reranks results using NVIDIA's reranking model for improved relevance
- Returns complete test examples with setup and validation
- Includes testing best practices and frameworks

SEARCH CAPABILITIES:
Your query is matched against:
- Test method implementations
- Test setup and teardown patterns
- UI testing examples
- USD stage testing patterns
- Extension lifecycle testing
- Performance and integration tests

ARGUMENTS:
- query (str): Test scenario or functionality to find examples for
- top_k (int, optional): Number of test examples to return (default: 10)

RETURNS:
Test examples with:
- Complete test method code
- Setup and teardown patterns
- Test assertions and validation
- File paths and test locations
- Testing framework usage
- Test categories and tags

USAGE EXAMPLES:
search_test_examples("ui widget testing")
search_test_examples("usd stage operations test", top_k=5)
search_test_examples("extension lifecycle test")
search_test_examples("async test patterns")
search_test_examples("button click testing")

TIPS FOR BETTER RESULTS:
- Use testing terminology (e.g., "test", "assert", "setup", "teardown")
- Include component types (e.g., "widget test", "stage test", "extension test")
- Specify testing patterns (e.g., "async test", "mock test", "integration test")
- Reference specific functionality (e.g., "button click", "window creation", "prim operations")"""


class SearchKitTestExamplesConfig(FunctionBaseConfig, name="search_kit_test_examples"):
    """Configuration for search_test_examples function."""

    name: str = "search_kit_test_examples"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    enable_rerank: bool = Field(default=True, description="Whether to enable reranking for improved relevance")


@register_function(config_type=SearchKitTestExamplesConfig, framework_wrappers=[])
async def register_search_kit_test_examples(config: SearchKitTestExamplesConfig, builder: Builder):
    """Register search_test_examples function with AIQ."""

    verbose = config.verbose

    if verbose:
        logger.info(f"Registering search_kit_test_examples in verbose mode")

    async def search_kit_test_examples_wrapper(input: SearchKitTestExamplesInput) -> str:
        """Search for Kit test examples."""
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

            result = await search_test_examples(
                query=sanitized_query,
                rerank_k=input.top_k,
                enable_rerank=config.enable_rerank,
            )

            if verbose:
                logger.debug(f"Searched test examples for: '{input.query}', top_k: {input.top_k}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to search test examples - {error_msg}"
        finally:
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_kit_test_examples",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_test_examples: {log_error}")

    function_info = FunctionInfo.from_fn(
        search_kit_test_examples_wrapper,
        description=SEARCH_KIT_TEST_EXAMPLES_DESCRIPTION,
        input_schema=SearchKitTestExamplesInput,
    )

    function_info.metadata = {"mcp_exposed": True}

    yield function_info
