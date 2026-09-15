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

from kit_fns.utils._retrieval_compat import empty_result_sentinel as _empty_result_sentinel
from kit_fns.utils._retrieval_compat import sanitize_query
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_test_examples import search_test_examples
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


class SearchKitTestExamplesInput(BaseModel):
    """Input for search_test_examples function."""

    query: str = Field(description="Test scenario or functionality to find examples for")
    top_k: int = Field(default=10, description="Number of test examples to return")


# Tool description
SEARCH_KIT_TEST_EXAMPLES_DESCRIPTION = """Narrow-scope tool for finding Kit test code — setup/teardown, assertions, async-test patterns, and Kit-specific test harnesses.

WHEN TO USE THIS TOOL:
- "Show me how to write a Kit unit/UI test for X."
- Looking for `omni.kit.test` patterns, async test fixtures, or test harness boilerplate.
- Need an example of testing extension lifecycle, widgets, or USD stages in Kit.

ARGUMENTS:
- query (str): test scenario or functionality to find examples for.
- top_k (int, optional): number of test examples to return (default 10).

RETURNS:
Test examples with complete test method code, setup/teardown patterns, assertions, file paths, and framework usage.

USAGE EXAMPLES:
search_kit_test_examples "ui widget testing"
search_kit_test_examples "async test patterns"
search_kit_test_examples "extension lifecycle test"

COVERAGE CAVEAT:
Indexed from tests shipped with Kit SDK source trees; vendor test suites or private tests will not appear.

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Non-test / production code examples → use search_kit_code_examples.
- Conceptual questions about Kit's testing framework → use search_kit_knowledge.
- Testing omni.ui widget behavior specifically → prefer the OmniUI MCP's search_ui_code_examples first.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand."""


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

            sanitized_query = sanitize_query(input.query)

            result = await search_test_examples(
                query=sanitized_query,
                rerank_k=input.top_k,
                enable_rerank=config.enable_rerank,
            )

            if verbose:
                logger.debug(f"Searched test examples for: '{_sanitize_log(input.query)}', top_k: {input.top_k}")

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
