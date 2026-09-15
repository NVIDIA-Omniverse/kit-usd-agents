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
from omni_aiq_usd_code.utils._retrieval_compat import empty_result_sentinel as _empty_result_sentinel
from omni_aiq_usd_code.utils._retrieval_compat import sanitize_query
from pydantic import BaseModel, Field

from .config import DEFAULT_RERANK_KNOWLEDGE


# Audit R4/R5: strip CR/LF and cap length on user-provided strings before
# embedding them in log lines or MCP error responses, so a query with
# ``\r\n`` can't forge structured-log records or leak arbitrary text
# back to the caller. Not a sanitizer for search purposes — only for
# *rendering* into logs/errors.
def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


class SearchUSDKnowledgeInput(BaseModel):
    """Input for search_usd_knowledge function."""

    request: str = Field(description="Query about USD concepts, workflows, or documentation")


from .functions.get_usd_knowledge import get_usd_knowledge
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)

# Tool description
SEARCH_USD_KNOWLEDGE_DESCRIPTION = """PRIMARY tool for any question about Universal Scene Description (USD). Start here for concepts, architecture, workflows, schema behavior, composition, lighting, animation, and "how do I…?" questions about USD.

WHEN TO USE THIS TOOL:
- "How does layer composition / stage / prim inheritance work?"
- "What is UsdLux / UsdGeom / UsdSkel for?"
- Behavioral or workflow questions about USD across renderers.
- Any conceptual USD question before reaching for a code example or class signature.

ARGUMENTS:
- request (str): natural-language question about USD concepts, workflows, or documentation.

RETURNS:
Formatted documentation excerpts with titles, content, and source URLs.

USAGE EXAMPLES:
search_usd_knowledge "What is layer composition?"
search_usd_knowledge "USD lighting workflow"
search_usd_knowledge "prim inheritance"

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Copy-pastable USD Python code → use search_usd_code_examples.
- Specific class signature / methods (e.g. UsdStage) → use get_usd_class_detail.
- Specific method docs (e.g. GetPrim) → use get_usd_method_detail.
- Enumerate pxr.* modules / classes → use list_usd_modules or list_usd_classes.
- Kit-specific (not pxr) USD integration → use the Kit MCP's search_kit_knowledge.
- omni.ui scene (3D UI) integration → use the OmniUI MCP's search_ui_code_examples.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand.
"""


class SearchUSDKnowledgeConfig(FunctionBaseConfig, name="search_usd_knowledge"):
    """Configuration for search_usd_knowledge function."""

    name: str = "search_usd_knowledge"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    rerank_k: int = Field(default=DEFAULT_RERANK_KNOWLEDGE, description="Number of documents to keep after reranking")
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
                    f"Retrieved knowledge for: {_sanitize_log(request)}, rerank_k: {config.rerank_k}, enable_rerank: {config.enable_rerank}"
                )

            if result["success"]:
                text = result["result"] or ""
                if not text.strip():
                    return _empty_result_sentinel()
                return text
            else:
                return f"ERROR: {_sanitize_log(result['error'], cap=500)}"

        except Exception as e:
            return f"ERROR: Failed to retrieve USD knowledge - {_sanitize_log(str(e), cap=500)}"

    function_info = FunctionInfo.from_fn(
        search_usd_knowledge_wrapper,
        description=SEARCH_USD_KNOWLEDGE_DESCRIPTION,
        input_schema=SearchUSDKnowledgeInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
