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

from kit_fns.utils._retrieval_compat import empty_result_sentinel as _empty_result_sentinel
from kit_fns.utils._retrieval_compat import sanitize_query
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from .functions.search_knowledge import search_kit_knowledge
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


# Tool description
SEARCH_KIT_KNOWLEDGE_DESCRIPTION = """PRIMARY tool for any question about NVIDIA Omniverse Kit or the broader
Omniverse platform. Start here for concepts, architecture, workflows,
how-to questions, API behavior, extension system, UI, USD integration,
application lifecycle, rendering, testing, and anything of the form
"how do I…?" or "what is…?".

Prefer this tool over specialized Kit/Omniverse tools whenever the
question is conceptual or explanatory. Only fall back to a specialized
tool when the answer you need is a concrete artifact (setting path,
class signature, code example, extension metadata).

KNOWLEDGE SOURCES:
Indexed from the full Kit documentation site and related Omniverse
knowledge bases (Kit docs, omniverse docs, extension guides, USD
documentation, survival guide).

SEARCH METHOD:
Semantic vector search (nemotron-3-embed-1b) with optional reranking
(llama-nemotron-rerank-vl-1b-v2).

ARGUMENTS:
- request (str): natural-language question

RETURNS:
Formatted documentation excerpts with titles, content, and source URLs.

USAGE EXAMPLES:
search_kit_knowledge "How does extension lifecycle work?"
search_kit_knowledge "How do I implement mesh light sampling?"
search_kit_knowledge "What is the difference between omni.ui and omni.kit.widget?"
search_kit_knowledge "How does RTX integrate with viewport rendering?"
search_kit_knowledge "What are Carbonite settings?"
search_kit_knowledge "omni.ui styling patterns"

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Looking up a specific setting path (e.g. /rtx/rendermode): use
  search_kit_settings.
- Finding a class signature, method, or API reference:
  use get_kit_api_details.
- Finding sample code to copy: use search_kit_code_examples.
- Discovering an extension: use search_kit_extensions or
  get_kit_extension_details."""


class SearchKitKnowledgeConfig(FunctionBaseConfig, name="search_kit_knowledge"):
    """Configuration for search_kit_knowledge function."""

    name: str = "search_kit_knowledge"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    enable_rerank: bool = Field(default=True, description="Whether to enable reranking for improved relevance")
    rerank_k: int = Field(default=10, description="Number of documents to keep after reranking")

    # Embedding configuration
    embedding_model: str = Field(
        default="nvidia/nemotron-3-embed-1b", description="The embedding model to use for search"
    )
    embedding_endpoint: str = Field(default="", description="Custom embedding endpoint URL (optional)")
    embedding_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for embedding service")

    # Reranking configuration
    reranking_model: str = Field(
        default="nvidia/llama-nemotron-rerank-vl-1b-v2", description="The reranking model to use"
    )
    reranking_endpoint: str = Field(
        default="https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking",
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
