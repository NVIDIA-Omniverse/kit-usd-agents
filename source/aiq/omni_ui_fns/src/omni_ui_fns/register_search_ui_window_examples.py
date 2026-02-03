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
from pydantic import BaseModel, Field

from .functions.get_window_examples import get_window_examples

logger = logging.getLogger(__name__)


# Define input schema for single argument function
class SearchUIWindowExamplesInput(BaseModel):
    """Input for search_ui_window_examples function."""

    query: str = Field(description="Your query describing the desired UI window example")


# Tool description
SEARCH_UI_WINDOW_EXAMPLES_DESCRIPTION = """Retrieves relevant UI window examples using semantic vector search from a curated database of OmniUI implementations.

WHAT IT DOES:
- Converts your query to embeddings using NVIDIA's embedding model
- Performs semantic similarity search against indexed UI window/dialog implementations
- Returns formatted examples with descriptions, complete code, and file paths
- Focuses specifically on window creation, dialog boxes, and UI container patterns

QUERY MATCHING:
Your query is compared against OmniUI window and dialog implementations, including:
- Window creation with various configurations (modal, non-modal, resizable, etc.)
- Dialog boxes and popup interfaces
- UI containers with buttons, controls, and layouts
- Error dialogs and confirmation windows
- Animation and curve editing interfaces
- Settings and configuration windows

ARGUMENTS:
- query (str): Your query describing the desired UI window example

RETURNS:
Formatted UI window examples with:
- Detailed descriptions of window functionality
- Complete Python code implementations
- File paths and function locations
- Class names and line numbers

USAGE EXAMPLES:
search_ui_window_examples "Create a modal dialog with buttons"
search_ui_window_examples "Window with sliders and controls"
search_ui_window_examples "Error message dialog box"
search_ui_window_examples "Animation curve simplification window"
search_ui_window_examples "Resizable window with UI components"

TIPS FOR BETTER RESULTS:
- Use window-specific terminology (e.g., "modal", "dialog", "window", "popup")
- Include UI component types (e.g., "buttons", "sliders", "checkboxes", "fields")
- Reference window behaviors (e.g., "resizable", "closable", "modal", "fixed size")
- Ask about specific UI patterns (e.g., "error dialog", "settings window", "confirmation")
"""


class SearchUIWindowExamplesConfig(FunctionBaseConfig, name="search_ui_window_examples"):
    """Configuration for search_ui_window_examples function."""

    name: str = "search_ui_window_examples"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    top_k: int = Field(default=5, description="Number of window examples to return")
    format_type: str = Field(default="formatted", description="Format type: 'structured', 'formatted', or 'raw'")

    # Embedding configuration
    embedding_model: Optional[str] = Field(default="nvidia/nv-embedqa-e5-v5", description="Embedding model to use")
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

    async def search_ui_window_examples_wrapper(input: SearchUIWindowExamplesInput) -> str:
        """Single argument with schema."""
        import time

        from omni_ui_fns.utils.input_sanitization import sanitize_query
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
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve UI window examples - {error_msg}"
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
