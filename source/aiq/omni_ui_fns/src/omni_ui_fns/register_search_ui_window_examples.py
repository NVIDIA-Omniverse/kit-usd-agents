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

"""Registration wrapper for search_ui_window_examples function."""

import logging
import os
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from omni_ui_fns.utils._retrieval_compat import empty_result_sentinel as _empty_result_sentinel
from omni_ui_fns.utils._retrieval_compat import sanitize_query
from pydantic import BaseModel, Field

from .functions.get_window_examples import get_window_examples


# Audit R4/R5: strip CR/LF and cap length on user-provided strings before
# embedding them in log lines or MCP error responses, so a query with
# ``\r\n`` can't forge structured-log records or leak arbitrary text
# back to the caller. Not a sanitizer for search purposes — only for
# *rendering* into logs/errors.
def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


logger = logging.getLogger(__name__)


# Define input schema for single argument function
class SearchUIWindowExamplesInput(BaseModel):
    """Input for search_ui_window_examples function."""

    query: str = Field(description="Your query describing the desired UI window example")


# Tool description
SEARCH_UI_WINDOW_EXAMPLES_DESCRIPTION = """Find complete omni.ui window / dialog / popup layouts — modal dialogs, settings windows, error boxes, full authoring panels.

WHEN TO USE THIS TOOL:
- "Show me a full modal dialog with buttons."
- "Give me a resizable settings window example."
- You need a whole window, not a single widget snippet.

ARGUMENTS:
- query (str): natural-language description of the desired window/dialog.

RETURNS:
Formatted window examples with a description, complete Python implementation, file paths, class names, and line numbers.

USAGE EXAMPLES:
search_ui_window_examples "modal dialog with buttons"
search_ui_window_examples "resizable window with sliders"
search_ui_window_examples "error message dialog"

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Individual widget / layout snippets (not whole windows) → use search_ui_code_examples.
- Styling rules (colors, shades, fonts) → use get_ui_style_docs.
- Full Window class signature / methods → use get_ui_class_detail.
- Kit-side window management (docking, viewport bindings) → use the Kit MCP's search_kit_code_examples.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand.
"""


class SearchUIWindowExamplesConfig(FunctionBaseConfig, name="search_ui_window_examples"):
    """Configuration for search_ui_window_examples function."""

    name: str = "search_ui_window_examples"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    top_k: int = Field(default=5, description="Number of window examples to return")
    format_type: str = Field(
        default="formatted",
        description="Format type: 'structured', 'formatted', or 'raw'",
    )

    # Embedding configuration
    embedding_model: Optional[str] = Field(default="nvidia/nemotron-3-embed-1b", description="Embedding model to use")
    embedding_endpoint: Optional[str] = Field(
        default=None, description="Embedding service endpoint (None for NVIDIA API)"
    )
    embedding_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for embedding service")

    # FAISS database configuration
    faiss_index_path: Optional[str] = Field(default=None, description="Path to FAISS index (uses default if None)")


@register_function(config_type=SearchUIWindowExamplesConfig, framework_wrappers=[])
async def register_search_ui_window_examples(config: SearchUIWindowExamplesConfig, builder: Builder):
    """Register search_ui_window_examples function with AIQ."""

    # Access config fields here
    if config.verbose:
        logger.info("Registering search_ui_window_examples in verbose mode")

    async def search_ui_window_examples_wrapper(
        input: SearchUIWindowExamplesInput,
    ) -> str:
        """Single argument with schema."""
        import time

        from omni_ui_fns.utils.usage_logging import get_usage_logger

        # Extract and sanitize the query string from the input model
        query = sanitize_query(input.query)

        # Debug logging
        logger.info(f"[DEBUG] search_ui_window_examples_wrapper called with input type: {type(input)}")
        logger.info(f"[DEBUG] search_ui_window_examples_wrapper query value: {query}")

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

            result = await get_window_examples(
                query,
                top_k=config.top_k,
                format_type=config.format_type,
                embedding_config={
                    "model": config.embedding_model,
                    "endpoint": config.embedding_endpoint,
                    "api_key": embedding_api_key,
                },
                faiss_index_path=config.faiss_index_path,
            )

            # Use config fields to modify behavior
            if config.verbose:
                logger.debug(
                    f"Retrieved UI window examples for: {query}, top_k: {config.top_k}, format: {config.format_type}"
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
            return f"ERROR: Failed to retrieve UI window examples - {_sanitize_log(error_msg, cap=500)}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_ui_window_examples",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_ui_window_examples: {log_error}")

    # Pass input_schema for proper MCP parameter handling
    function_info = FunctionInfo.from_fn(
        search_ui_window_examples_wrapper,
        description=SEARCH_UI_WINDOW_EXAMPLES_DESCRIPTION,
        input_schema=SearchUIWindowExamplesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
