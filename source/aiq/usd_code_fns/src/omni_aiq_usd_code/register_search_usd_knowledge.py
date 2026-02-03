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

"""Registration wrapper for search_usd_knowledge function."""

import logging
import os
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .config import DEFAULT_RERANK_KNOWLEDGE


class SearchUSDKnowledgeInput(BaseModel):
    """Input for search_usd_knowledge function."""

    request: str = Field(description="Query about USD concepts, workflows, or documentation")


from .functions.get_usd_knowledge import get_usd_knowledge
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)

# Tool description
SEARCH_USD_KNOWLEDGE_DESCRIPTION = """Retrieves relevant USD documentation and knowledge using semantic vector search and optional reranking.

WHAT IT DOES:
- Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
- Performs semantic similarity search against pre-indexed USD documentation
- Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
- Returns formatted documentation excerpts with titles and URLs

QUERY MATCHING:
Your query is compared against the 'index_text' field of each document,
which contains concise summaries like:
- "UsdLuxGeometryLight lacks detailed specifications for consistent behavior across renderers"
- "Different renderers have varying approaches to mesh lights"
- "The design should cater to specific workflow requirements of geometry lights"

KNOWLEDGE DOMAINS COVERED:
- USD concepts and architecture (layers, prims, stages, composition)
- Rendering and lighting (UsdLux, geometry lights, materials)
- Animation and skeletal systems (UsdSkel)
- Geometry and scenes (UsdGeom)
- Workflows and best practices

ARGUMENTS:
- request (str): Your query about USD concepts, workflows, or documentation

RETURNS:
Formatted documentation excerpts with titles, content, and source URLs, or error message

USAGE EXAMPLES:
search_usd_knowledge "What is layer composition?"
search_usd_knowledge "USD lighting workflow"
search_usd_knowledge "prim inheritance"

TIPS FOR BETTER RESULTS:
- Use specific USD concepts (e.g., "layer composition", "prim inheritance")
- Include relevant domains (e.g., "lighting", "animation", "geometry")
- Ask about workflows, specifications, or behavioral details
- Use USD schema names when relevant (e.g., "UsdLux", "UsdGeom", "UsdSkel")
"""


class SearchUSDKnowledgeConfig(FunctionBaseConfig, name="search_usd_knowledge"):
    """Configuration for search_usd_knowledge function."""

    name: str = "search_usd_knowledge"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    rerank_k: int = Field(default=DEFAULT_RERANK_KNOWLEDGE, description="Number of documents to keep after reranking")
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


@register_function(config_type=SearchUSDKnowledgeConfig, framework_wrappers=[])
async def register_search_usd_knowledge(config: SearchUSDKnowledgeConfig, builder: Builder):
    """Register search_usd_knowledge function with AIQ."""

    # Access config fields here
    if config.verbose:
        logger.info(f"Registering search_usd_knowledge in verbose mode")

    @log_tool_usage("search_usd_knowledge")
    async def search_usd_knowledge_wrapper(request: str) -> str:
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

            result = await get_usd_knowledge(
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
                    f"Retrieved knowledge for: {request}, rerank_k: {config.rerank_k}, enable_rerank: {config.enable_rerank}"
                )

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {result['error']}"

        except Exception as e:
            return f"ERROR: Failed to retrieve USD knowledge - {str(e)}"

    function_info = FunctionInfo.from_fn(
        search_usd_knowledge_wrapper,
        description=SEARCH_USD_KNOWLEDGE_DESCRIPTION,
        input_schema=SearchUSDKnowledgeInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
