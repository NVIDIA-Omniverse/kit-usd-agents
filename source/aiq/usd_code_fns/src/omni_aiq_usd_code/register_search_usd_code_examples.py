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
from omni_aiq_usd_code.utils._retrieval_compat import empty_result_sentinel as _empty_result_sentinel
from omni_aiq_usd_code.utils._retrieval_compat import sanitize_query
from pydantic import BaseModel, Field

from .config import DEFAULT_RERANK_CODE


# Audit R4/R5: strip CR/LF and cap length on user-provided strings before
# embedding them in log lines or MCP error responses, so a query with
# ``\r\n`` can't forge structured-log records or leak arbitrary text
# back to the caller. Not a sanitizer for search purposes — only for
# *rendering* into logs/errors.
def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


class SearchUSDCodeExamplesInput(BaseModel):
    """Input for search_usd_code_examples function."""

    request: str = Field(description="Description of desired USD code functionality")


from .functions.get_usd_code_example import get_usd_code_example
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)

# Tool description
SEARCH_USD_CODE_EXAMPLES_DESCRIPTION = """Find copy-pastable USD (pxr.*) Python snippets — stage creation, prim ops, composition, UsdGeom/UsdSkel/UsdLux patterns.

WHEN TO USE THIS TOOL:
- "How to create a USD stage / mesh / point instancer?"
- "Show me code for ComputePointsAtTime / SetTransformOp / composition arcs."
- You want runnable Python using pxr modules, not prose docs.

ARGUMENTS:
- request (str): natural-language query describing the desired USD code example.

RETURNS:
Formatted code examples, each with the question it answers and a Python snippet using pxr.*.

USAGE EXAMPLES:
search_usd_code_examples "How to create a mesh?"
search_usd_code_examples "UsdSkel animation"
search_usd_code_examples "layer composition"

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Conceptual USD questions → use search_usd_knowledge.
- Class signature / docstring for pxr types → use get_usd_class_detail.
- Method-level lookup → use get_usd_method_detail.
- Kit-side USD integration (stage attach, hydra viewport, etc.) → use the Kit MCP's search_kit_code_examples.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand.
"""


class SearchUSDCodeExamplesConfig(FunctionBaseConfig, name="search_usd_code_examples"):
    """Configuration for search_usd_code_examples function."""

    name: str = "search_usd_code_examples"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    rerank_k: int = Field(default=DEFAULT_RERANK_CODE, description="Number of documents to keep after reranking")
    enable_rerank: bool = Field(default=True, description="Enable reranking of search results")

    # Embedding configuration
    embedding_model: Optional[str] = Field(default="nvidia/nemotron-3-embed-1b", description="Embedding model to use")
    embedding_endpoint: Optional[str] = Field(
        default=None, description="Embedding service endpoint (None for NVIDIA API)"
    )
    embedding_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for embedding service")

    # Reranking configuration
    reranking_model: Optional[str] = Field(
        default="nvidia/llama-nemotron-rerank-vl-1b-v2", description="Reranking model to use"
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
                    f"Retrieved code examples for: {_sanitize_log(request)}, rerank_k: {config.rerank_k}, enable_rerank: {config.enable_rerank}"
                )

            if result["success"]:
                text = result["result"] or ""
                if not text.strip():
                    return _empty_result_sentinel()
                return text
            else:
                return f"ERROR: {_sanitize_log(result['error'], cap=500)}"

        except Exception as e:
            return f"ERROR: Failed to retrieve USD code examples - {_sanitize_log(str(e), cap=500)}"

    function_info = FunctionInfo.from_fn(
        search_usd_code_examples_wrapper,
        description=SEARCH_USD_CODE_EXAMPLES_DESCRIPTION,
        input_schema=SearchUSDCodeExamplesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
