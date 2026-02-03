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

"""Registration wrapper for search_ui_code_examples function."""

import logging
import os
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .config import DEFAULT_RERANK_CODE
from .functions.get_code_examples import get_code_examples

logger = logging.getLogger(__name__)


# Define input schema for single argument function
class SearchUICodeExamplesInput(BaseModel):
    """Input for search_ui_code_examples function."""

    query: str = Field(description="Your query describing the desired code example")


# Tool description
SEARCH_UI_CODE_EXAMPLES_DESCRIPTION = """Retrieves relevant code examples using semantic vector search and optional reranking.

WHAT IT DOES:
- Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
- Performs semantic similarity search against pre-indexed OmniUI code examples
- Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
- Returns formatted code examples with their metadata and source code

QUERY MATCHING:
Your query is compared against OmniUI widget and component implementations,
including examples like:
- SearchField, SearchWordButton, and other UI widgets
- Widget styling and theming functions
- UI component building patterns (ZStack, VStack, HStack)
- Event handling and callback patterns
- Layout and spacing utilities

ARGUMENTS:
- query (str): Your query describing the desired OmniUI code example

RETURNS:
Formatted code examples with file paths, method names, and Python code snippets

USAGE EXAMPLES:
search_ui_code_examples "How to create a search field?"
search_ui_code_examples "Button styling with themes"
search_ui_code_examples "event handling callbacks"
search_ui_code_examples "VStack and HStack layout"
search_ui_code_examples "create custom widget"

TIPS FOR BETTER RESULTS:
- Use specific OmniUI terminology (e.g., "SearchField", "ZStack", "VStack")
- Include UI operations (e.g., "build_ui", "style", "event handling")
- Reference widget types (e.g., "Button", "Label", "Rectangle", "Spacer")
- Ask about patterns (e.g., "callback", "subscription", "model binding")
"""


class SearchUICodeExamplesConfig(FunctionBaseConfig, name="search_ui_code_examples"):
    """Configuration for search_ui_code_examples function."""

    name: str = "search_ui_code_examples"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    rerank_k: int = Field(default=DEFAULT_RERANK_CODE, description="Number of documents to keep after reranking")
    enable_rerank: bool = Field(default=True, description="Enable reranking of search results")

    # Embedding configuration
    embedding_model: Optional[str] = Field(default="nvidia/nv-embedqa-e5-v5", description="Embedding model to use")
    embedding_endpoint: Optional[str] = Field(
        default=None, description="Embedding service endpoint (None for NVIDIA API)"
    )
    embedding_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for embedding service")

    # Reranking configuration
    reranking_model: Optional[str] = Field(
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2", description="Reranking model to use"
    )
    reranking_endpoint: Optional[str] = Field(
        default=None, description="Reranking service endpoint (None for NVIDIA API)"
    )
    reranking_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for reranking service")


@register_function(config_type=SearchUICodeExamplesConfig, framework_wrappers=[])
async def register_search_ui_code_examples(config: SearchUICodeExamplesConfig, builder: Builder):
    """Register search_ui_code_examples function with AIQ."""

    # Access config fields here
    if config.verbose:
        logger.info("Registering search_ui_code_examples in verbose mode")

    async def search_ui_code_examples_wrapper(input: SearchUICodeExamplesInput) -> str:
        """Single argument with schema."""
        import time

        from omni_ui_fns.utils.input_sanitization import sanitize_query
        from omni_ui_fns.utils.usage_logging import get_usage_logger

        # Extract and sanitize the query string from the input model
        query = sanitize_query(input.query)

        # Debug logging
        logger.info(f"[DEBUG] search_ui_code_examples_wrapper called with input type: {type(input)}")
        logger.info(f"[DEBUG] search_ui_code_examples_wrapper query value: {query}")

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {"query": query}
        error_msg = None
        success = True

        try:
            # Handle environment variable substitution for API keys
            embedding_api_key = config.embedding_api_key
            if embedding_api_key == "${NVIDIA_API_KEY}":
                embedding_api_key = os.getenv("NVIDIA_API_KEY")

            reranking_api_key = config.reranking_api_key
            if reranking_api_key == "${NVIDIA_API_KEY}":
                reranking_api_key = os.getenv("NVIDIA_API_KEY")

            result = await get_code_examples(
                query,
                rerank_k=config.rerank_k,
                enable_rerank=config.enable_rerank,
                embedding_config={
                    "model": config.embedding_model,
                    "endpoint": config.embedding_endpoint,
                    "api_key": embedding_api_key,
                },
                reranking_config={
                    "model": config.reranking_model,
                    "endpoint": config.reranking_endpoint,
                    "api_key": reranking_api_key,
                },
            )

            # Use config fields to modify behavior
            if config.verbose:
                logger.debug(
                    f"Retrieved OmniUI code examples for: {query}, rerank_k: {config.rerank_k}, enable_rerank: {config.enable_rerank}"
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
            return f"ERROR: Failed to retrieve OmniUI code examples - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_ui_code_examples",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_ui_code_examples: {log_error}")

    # Pass input_schema for proper MCP parameter handling
    function_info = FunctionInfo.from_fn(
        search_ui_code_examples_wrapper,
        description=SEARCH_UI_CODE_EXAMPLES_DESCRIPTION,
        input_schema=SearchUICodeExamplesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
