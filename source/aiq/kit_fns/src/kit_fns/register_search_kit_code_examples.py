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

from kit_fns.utils._retrieval_compat import empty_result_sentinel as _empty_result_sentinel
from kit_fns.utils._retrieval_compat import sanitize_query
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_code_examples import search_code_examples
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


class SearchKitCodeExamplesInput(BaseModel):
    """Input for search_code_examples function."""

    query: str = Field(description="Description of desired Kit code functionality")
    top_k: int = Field(default=10, description="Number of code examples to return")


# Tool description
SEARCH_KIT_CODE_EXAMPLES_DESCRIPTION = """Find copy-pastable Kit code snippets harvested from Kit SDK source, extensions, and samples.

WHEN TO USE THIS TOOL:
- "Show me how to implement X in Kit" where X is a concrete code pattern.
- "Give me an example of a Kit extension lifecycle / callback / subscription."
- Looking for boilerplate for omni.kit.app, omni.kit.window, or extension scaffolding.

ARGUMENTS:
- query (str): description of the desired code pattern.
- top_k (int, optional): number of examples to return (default 10).

RETURNS:
Formatted code examples with implementation code, file paths, extension IDs, descriptions, relevance scores, and tags.

USAGE EXAMPLES:
search_kit_code_examples "create window with buttons"
search_kit_code_examples "extension lifecycle management"
search_kit_code_examples "subscribe to stage events"

COVERAGE CAVEAT:
Examples are harvested from Kit SDK builds; coverage tracks what ships in
those SDKs and may lag very new or internal-only APIs. Fall back to
`search_kit_knowledge` for conceptual answers when no example matches.

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Conceptual "how does X work?" → use search_kit_knowledge.
- API class/method signature lookup → use get_kit_api_details.
- Test-authoring patterns → use search_kit_test_examples.
- OmniUI widget/layout code → use the OmniUI MCP's search_ui_code_examples.
- USD-API code (pxr.*) → use the USD Code MCP's search_usd_code_examples.
- Isaac Sim robotics code → use the Isaac Sim MCP's search_isaac_sim_code_examples.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand."""


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

            sanitized_query = sanitize_query(input.query)

            result = await search_code_examples(
                query=sanitized_query,
                rerank_k=input.top_k,
                enable_rerank=config.enable_rerank,
            )

            if verbose:
                logger.debug(f"Searched code examples for: '{_sanitize_log(input.query)}', top_k: {input.top_k}")

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
