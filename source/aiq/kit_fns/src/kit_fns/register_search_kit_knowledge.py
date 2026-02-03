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

"""Registration wrapper for search_kit_knowledge function."""

import logging
import os
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from .functions.search_knowledge import search_kit_knowledge
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


# Tool description
SEARCH_KIT_KNOWLEDGE_DESCRIPTION = """Retrieves relevant Kit documentation and knowledge using semantic vector search and optional reranking.

WHAT IT DOES:
- Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
- Performs semantic similarity search against pre-indexed Kit documentation
- Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
- Returns formatted documentation excerpts with titles and URLs

QUERY MATCHING:
Your query is compared against the 'index_text' field of each document,
which contains concise summaries of Kit concepts and documentation.

KNOWLEDGE DOMAINS COVERED:
- Kit framework architecture and extension system
- UI development with omni.ui
- USD integration and scene manipulation
- Testing frameworks and patterns
- Application lifecycle and settings

ARGUMENTS:
- request (str): Your query about Kit concepts, workflows, or documentation

RETURNS:
Formatted documentation excerpts with titles, content, and source URLs, or error message

USAGE EXAMPLES:
search_kit_knowledge "How does extension lifecycle work?"
search_kit_knowledge "Kit viewport rendering"
search_kit_knowledge "omni.ui styling patterns"

TIPS FOR BETTER RESULTS:
- Use specific Kit concepts (e.g., "extension lifecycle", "viewport rendering")
- Include relevant domains (e.g., "UI", "settings", "testing")
- Ask about workflows, patterns, or implementation details
- Use Kit module names when relevant (e.g., "omni.ui", "omni.kit")
"""


class SearchKitKnowledgeConfig(FunctionBaseConfig, name="search_kit_knowledge"):
    """Configuration for search_kit_knowledge function."""

    name: str = "search_kit_knowledge"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    enable_rerank: bool = Field(default=True, description="Whether to enable reranking for improved relevance")
    rerank_k: int = Field(default=10, description="Number of documents to keep after reranking")

    # Embedding configuration
    embedding_model: str = Field(default="nvidia/nv-embedqa-e5-v5", description="The embedding model to use for search")
    embedding_endpoint: str = Field(default="", description="Custom embedding endpoint URL (optional)")
    embedding_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for embedding service")

    # Reranking configuration
    reranking_model: str = Field(default="nvidia/llama-3.2-nv-rerankqa-1b-v2", description="The reranking model to use")
    reranking_endpoint: str = Field(
        default="https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        description="Custom reranking endpoint URL (optional)",
    )
    reranking_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for reranking service")


@register_function(config_type=SearchKitKnowledgeConfig, framework_wrappers=[])
async def register_search_kit_knowledge(config: SearchKitKnowledgeConfig, builder: Builder):
    """Register search_kit_knowledge function with AIQ."""

    verbose = config.verbose

    if verbose:
        logger.info("Registering search_kit_knowledge in verbose mode")

    async def search_kit_knowledge_wrapper(request: str) -> str:
        """Search for Kit knowledge documentation."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {
            "request": request,
            "rerank_k": config.rerank_k,
            "enable_rerank": config.enable_rerank,
        }
        error_msg = None
        success = True

        try:
            # Sanitize user input before sending to external APIs
            from kit_fns.utils.input_sanitization import sanitize_query

            sanitized_request = sanitize_query(request)

            # Handle environment variable substitution for API keys
            embedding_api_key = config.embedding_api_key
            if embedding_api_key == "${NVIDIA_API_KEY}":
                embedding_api_key = os.getenv("NVIDIA_API_KEY")

            reranking_api_key = config.reranking_api_key
            if reranking_api_key == "${NVIDIA_API_KEY}":
                reranking_api_key = os.getenv("NVIDIA_API_KEY")

            result = await search_kit_knowledge(
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

            if verbose:
                logger.debug(
                    f"Searched knowledge for: '{request}', rerank_k: {config.rerank_k}, "
                    f"enable_rerank: {config.enable_rerank}"
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
            return f"ERROR: Failed to search Kit knowledge - {error_msg}"
        finally:
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_kit_knowledge",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_kit_knowledge: {log_error}")

    function_info = FunctionInfo.from_fn(
        search_kit_knowledge_wrapper,
        description=SEARCH_KIT_KNOWLEDGE_DESCRIPTION,
    )

    function_info.metadata = {"mcp_exposed": True}

    yield function_info
