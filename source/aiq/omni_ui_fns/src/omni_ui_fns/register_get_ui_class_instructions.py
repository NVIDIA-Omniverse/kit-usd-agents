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

"""Registration wrapper for get_ui_class_instructions function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_class_instructions import get_class_instructions, list_class_categories, list_classes_in_category
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_class_names_input(class_names_str: str) -> Union[List[str], None, str]:
    """Parse class_names string input into appropriate format for get_class_instructions.

    Args:
        class_names_str: String input that can be:
            - None/empty: Return None (list all categories)
            - Special commands: "categories", "category:widgets", "category:scene"
            - Single class: "Button" or "TreeView"
            - JSON array: '["Button", "Label", "TreeView"]'
            - Comma-separated: "Button, Label, TreeView"

    Returns:
        - None: For empty input (list categories)
        - String: For special commands (categories, category:xxx)
        - List[str]: For class names

    Raises:
        ValueError: If input format is invalid
    """
    if not class_names_str or not class_names_str.strip():
        return None

    class_names_str = class_names_str.strip()

    # Handle special commands
    if class_names_str.lower() == "categories":
        return "categories"
    elif class_names_str.lower().startswith("category:"):
        return class_names_str.lower()  # Return as-is for category parsing

    # Try to parse as JSON array first
    if class_names_str.startswith("[") and class_names_str.endswith("]"):
        try:
            parsed = json.loads(class_names_str)
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
    if "," in class_names_str:
        return [name.strip() for name in class_names_str.split(",") if name.strip()]

    # Single class name
    return [class_names_str]


class GetUIClassInstructionsInput(BaseModel):
    """Input for get_ui_class_instructions function.

    Provide class names in any convenient format - the system will handle the conversion automatically.
    """

    class_names: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Class names to look up. Accepts multiple flexible formats:

        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single class: "Button" or "TreeView"
        - Native array: ["Button", "Label", "TreeView"] ← WORKS DIRECTLY!
        - JSON string: '["Button", "Label", "TreeView"]'
        - Comma-separated: "Button, Label, TreeView"
        - Scene classes: "scene.Line", "scene.Rectangle", "omni.ui.scene.Line"
        - Empty/null: Lists all categories

        🎯 SPECIAL COMMANDS (string only):
        - "categories": List all available categories
        - "category:widgets": List all classes in widgets category
        - "category:scene": List all 3D scene UI classes

        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_UI_CLASS_INSTRUCTIONS_DESCRIPTION = """Opinionated usage guidance for a named omni.ui class — how to instantiate it, style it, and common patterns. Also exposes category listings for discovery.

WHEN TO USE THIS TOOL:
- "How am I supposed to use ui.Button / ui.Window / ui.scene.Line?"
- Browsing categories (widgets, containers, layouts, shapes, windows, scene, etc.).
- After get_ui_class_detail, when you want the "here is how to actually wire this up" guidance.

ARGUMENTS:
- class_names (str | list[str] | null): class name(s) to look up; accepts single string, list, JSON-array string, or comma-separated string. Also accepts the special commands "categories" and "category:<name>" (e.g. "category:widgets", "category:scene"). Null lists all categories.

RETURNS:
For class names: formatted documentation with usage, styling, properties, and code examples. For category commands: structured category / class listings.

USAGE EXAMPLES:
get_ui_class_instructions "Button"
get_ui_class_instructions ["Button", "TreeView"]
get_ui_class_instructions "category:widgets"

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Raw class signature / method list → use get_ui_class_detail.
- Free-form "show me a widget example" → use search_ui_code_examples.
- Full window / dialog layouts → use search_ui_window_examples.
- Styling rules (colors, shades, fonts) → use get_ui_style_docs.
- Top-level OmniUI playbook → use get_ui_instructions."""


class GetUIClassInstructionsConfig(FunctionBaseConfig, name="get_ui_class_instructions"):
    """Configuration for get_ui_class_instructions function."""

    name: str = "get_ui_class_instructions"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUIClassInstructionsConfig, framework_wrappers=[])
async def register_get_ui_class_instructions(config: GetUIClassInstructionsConfig, builder: Builder):
    """Register get_ui_class_instructions function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_class_instructions in verbose mode")

    async def get_ui_class_instructions_wrapper(input: GetUIClassInstructionsInput) -> str:
        """Get OmniUI class instructions with support for categories and listings."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.class_names is None:
                # Empty input - list all categories
                result = await list_class_categories()
            elif isinstance(input.class_names, list):
                # Direct array input - validate and use as-is
                if len(input.class_names) == 0:
                    # Empty array = list all categories
                    result = await list_class_categories()
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.class_names):
                        if not isinstance(item, str):
                            return f"ERROR: All items in class_names array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in class_names array"
                    # Process as list of class names
                    classes_to_fetch = [item.strip() for item in input.class_names]
                    result = await get_class_instructions(classes_to_fetch)
            elif isinstance(input.class_names, str):
                # String input - handle special commands and regular parsing
                parsed_input = _parse_class_names_input(input.class_names)
                if isinstance(parsed_input, str):
                    # Special commands
                    if parsed_input == "categories":
                        result = await list_class_categories()
                    elif parsed_input.startswith("category:"):
                        # Category specified - extract category name
                        category_name = parsed_input[9:]  # Remove "category:" prefix
                        if not category_name or category_name.lower() == "none":
                            # If category name is empty or 'none', list all categories
                            result = await list_class_categories()
                        else:
                            result = await list_classes_in_category(category_name)
                    else:
                        # Single class name passed as string (fallback)
                        result = await get_class_instructions([parsed_input])
                elif isinstance(parsed_input, list):
                    # List of class names from string parsing
                    result = await get_class_instructions(parsed_input)
                else:
                    # None from empty string
                    result = await list_class_categories()
            else:
                return f"ERROR: class_names must be a string, array, or null, got {type(input.class_names).__name__}"

            parameters = {"class_names": input.class_names}
        except ValueError as e:
            return f"ERROR: Invalid class_names parameter: {str(e)}"

        error_msg = None
        success = True

        try:

            # Use config fields to modify behavior
            if verbose:
                if result.get("metadata"):
                    metadata = result["metadata"]
                    if isinstance(metadata, dict) and "classes" in metadata:
                        # Multiple classes
                        logger.debug(f"Retrieved {metadata.get('successful', 0)} class instructions")
                    else:
                        # Single class
                        logger.debug(
                            f"Retrieved: {metadata.get('class_name', 'N/A')} from {metadata.get('category', 'N/A')} category"
                        )
                else:
                    logger.debug(f"Listed categories or classes")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve class instructions - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_ui_class_instructions",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_class_instructions: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_ui_class_instructions_wrapper,
        description=GET_UI_CLASS_INSTRUCTIONS_DESCRIPTION,
        input_schema=GetUIClassInstructionsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
