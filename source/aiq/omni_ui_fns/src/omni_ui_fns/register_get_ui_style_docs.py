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

"""Registration wrapper for get_ui_style_docs function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_style_docs import get_style_docs
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_sections_input(sections_str: str) -> Union[List[str], None]:
    """Parse sections string input into a list of section names.

    Args:
        sections_str: String input that can be:
            - None/empty: Return None (get combined documentation)
            - Single section: "buttons" or "widgets"
            - JSON array: '["buttons", "widgets", "containers"]'
            - Comma-separated: "buttons, widgets, containers"

    Returns:
        List of section names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not sections_str or not sections_str.strip():
        return None

    sections_str = sections_str.strip()

    # Try to parse as JSON array first
    if sections_str.startswith("[") and sections_str.endswith("]"):
        try:
            parsed = json.loads(sections_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in sections_str:
        return [name.strip() for name in sections_str.split(",") if name.strip()]

    # Single section name
    return [sections_str]


# Define input schema for get_ui_style_docs function
class GetUIStyleDocsInput(BaseModel):
    """Input parameters for get_ui_style_docs function.

    Provide section names in any convenient format - the system will handle the conversion automatically.
    """

    sections: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Sections to retrieve. Accepts multiple flexible formats:

        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single section: "buttons" or "widgets"
        - Native array: ["buttons", "widgets", "containers"] ← WORKS DIRECTLY!
        - JSON string: '["buttons", "widgets", "containers"]'
        - Comma-separated: "buttons, widgets, containers"
        - Empty/null: Gets complete combined documentation

        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )


# Tool description
GET_UI_STYLE_DOCS_DESCRIPTION = """Retrieve comprehensive OmniUI style documentation - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- sections: Section names in ANY convenient format:
  * Single section string: "buttons" or "widgets"
  * Native array: ["buttons", "widgets", "containers"] ← WORKS DIRECTLY!
  * JSON string: '["buttons", "widgets", "containers"]'
  * Comma-separated: "buttons, widgets, containers"
  * Empty/null: Gets complete combined documentation

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_style_docs(sections=["buttons", "widgets"])
✅ Single string: get_ui_style_docs(sections="buttons")
✅ JSON string: get_ui_style_docs(sections='["buttons", "widgets", "containers"]')
✅ Comma format: get_ui_style_docs(sections="buttons, widgets, containers")
✅ Complete docs: get_ui_style_docs() or get_ui_style_docs(sections=null)

💡 FOR AI MODELS: You can pass arrays directly like ["buttons", "widgets"] - no need to convert to strings!

AVAILABLE SECTIONS:
- **overview**: High-level introduction to OmniUI styling system
- **styling**: Core styling syntax and rules
- **units**: Measurement system for UI elements (px, %, em, rem)
- **fonts**: Typography system and text styling
- **shades**: Color palettes and theme management (dark/light modes)
- **window**: Window-level styling and frame customization
- **containers**: Layout components (Frame, Stack, Grid, ScrollArea)
- **widgets**: Individual UI components (Label, Input, Checkbox, ComboBox, etc.)
- **buttons**: Button variations and states (normal, hover, pressed, disabled)
- **sliders**: Slider and range components with customization options
- **shapes**: Basic geometric elements (Rectangle, Circle, Triangle, Polygon)
- **line**: Line and curve elements with styling options

RETURNS:
- For single section: Section content with metadata
- For multiple sections: Dictionary of sections with their content
- For combined: Complete documentation with all sections (37,820+ tokens)
- Includes: Property descriptions, usage examples, and best practices
- Maximum compatibility with all AI models

USE CASES:
- Learning OmniUI styling syntax and customization
- Finding specific styling properties for UI components
- Understanding theme and color management systems
- Implementing custom widget styles and layouts"""


class GetUIStyleDocsConfig(FunctionBaseConfig, name="get_ui_style_docs"):
    """Configuration for get_ui_style_docs function."""

    name: str = "get_ui_style_docs"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUIStyleDocsConfig, framework_wrappers=[])
async def register_get_ui_style_docs(config: GetUIStyleDocsConfig, builder: Builder):
    """Register get_ui_style_docs function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_style_docs in verbose mode")

    async def get_ui_style_docs_wrapper(input: GetUIStyleDocsInput) -> str:
        """Wrapper for get_style_docs with AIQ integration."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.sections is None:
                sections_to_fetch = None
            elif isinstance(input.sections, list):
                # Direct array input - validate and use as-is
                if len(input.sections) == 0:
                    sections_to_fetch = None  # Empty array = get combined docs
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.sections):
                        if not isinstance(item, str):
                            return f"ERROR: All items in sections array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in sections array"
                    sections_to_fetch = [item.strip() for item in input.sections]
            elif isinstance(input.sections, str):
                # String input - parse using existing logic
                sections_to_fetch = _parse_sections_input(input.sections)
            else:
                return f"ERROR: sections must be a string, array, or null, got {type(input.sections).__name__}"

            parameters = {"sections": input.sections}
        except ValueError as e:
            return f"ERROR: Invalid sections parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function with parsed parameters (new simplified API)
            result = await get_style_docs(sections_to_fetch)

            # Use config fields to modify behavior
            if verbose:
                if sections_to_fetch is None:
                    section_info = "combined"
                elif len(sections_to_fetch) == 1:
                    section_info = sections_to_fetch[0]
                else:
                    section_info = f"{len(sections_to_fetch)} sections"
                logger.debug(f"Retrieved style documentation for: {section_info}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve style documentation - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_ui_style_docs",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_style_docs: {log_error}")

    # Create function info with input schema
    function_info = FunctionInfo.from_fn(
        get_ui_style_docs_wrapper,
        description=GET_UI_STYLE_DOCS_DESCRIPTION,
        input_schema=GetUIStyleDocsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
