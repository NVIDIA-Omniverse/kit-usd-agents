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

"""Registration wrapper for search_usd_code_examples function."""

import logging
import os
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .config import DEFAULT_RERANK_CODE


class SearchUSDCodeExamplesInput(BaseModel):
    """Input for search_usd_code_examples function."""

    request: str = Field(description="Description of desired USD code functionality")


from .functions.get_usd_code_example import get_usd_code_example
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)

# Tool description
SEARCH_USD_CODE_EXAMPLES_DESCRIPTION = """Retrieves relevant USD code examples using semantic vector search and optional reranking.

WHAT IT DOES:
- Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
- Performs semantic similarity search against pre-indexed code examples
- Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
- Returns formatted code examples with their associated questions

QUERY MATCHING:
Your query is compared against the 'index_text' field of each code example,
which contains concise questions like:
- "How to create a USD stage?"
- "How to compute points at a specific time using pxr.UsdGeom.PointBased.ComputePointsAtTime?"
- "How to get the prim stack with layer offsets?"

ARGUMENTS:
- request (str): Your query describing the desired USD code example

RETURNS:
Formatted code examples with questions and Python code snippets, or error message

USAGE EXAMPLES:
search_usd_code_examples "How to create a mesh?"
search_usd_code_examples "UsdSkel animation"
search_usd_code_examples "layer composition"

TIPS FOR BETTER RESULTS:
- Use specific USD terminology (e.g., "UsdStage", "UsdPrim", "layer offsets")
- Frame queries as "How to..." questions when possible
- Include relevant USD classes/methods in your query (e.g., "UsdGeom.Mesh", "CreatePrim")
- Be specific about the USD operation you want to perform
"""


class SearchUSDCodeExamplesConfig(FunctionBaseConfig, name="search_usd_code_examples"):
    """Configuration for search_usd_code_examples function."""

    name: str = "search_usd_code_examples"
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


@register_function(config_type=SearchUSDCodeExamplesConfig, framework_wrappers=[])
async def register_search_usd_code_examples(config: SearchUSDCodeExamplesConfig, builder: Builder):
    """Register search_usd_code_examples function with AIQ."""

    # Access config fields here
    if config.verbose:
        logger.info(f"Registering search_usd_code_examples in verbose mode")

    @log_tool_usage("search_usd_code_examples")
    async def search_usd_code_examples_wrapper(request: str) -> str:
        """Single argument - no schema needed."""
        try:
            # Sanitize user input before sending to external APIs
            from omni_aiq_usd_code.utils.input_sanitization import sanitize_query

            sanitized_request = sanitize_query(request)

            # Handle environment variable substitution for API keys
            embedding_api_key = config.embedding_api_key
            if embedding_api_key == "${NVIDIA_API_KEY}":
                embedding_api_key = os.getenv("NVIDIA_API_KEY")

            reranking_api_key = config.reranking_api_key
            if reranking_api_key == "${NVIDIA_API_KEY}":
                reranking_api_key = os.getenv("NVIDIA_API_KEY")

            result = await get_usd_code_example(
                sanitized_request,
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
                    f"Retrieved code examples for: {request}, rerank_k: {config.rerank_k}, enable_rerank: {config.enable_rerank}"
                )

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {result['error']}"

        except Exception as e:
            return f"ERROR: Failed to retrieve USD code examples - {str(e)}"

    function_info = FunctionInfo.from_fn(
        search_usd_code_examples_wrapper,
        description=SEARCH_USD_CODE_EXAMPLES_DESCRIPTION,
        input_schema=SearchUSDCodeExamplesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
