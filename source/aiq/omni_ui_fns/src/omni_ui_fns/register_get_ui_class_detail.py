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

"""Registration wrapper for get_ui_class_detail function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_class_detail import get_class_detail
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_class_names_input(class_names_str: str) -> Union[List[str], None]:
    """Parse class_names string input into a list of class names.

    Args:
        class_names_str: String input that can be:
            - None/empty: Return None (get all available classes)
            - Single class: "TreeView"
            - JSON array: '["Button", "Label", "TreeView"]'
            - Comma-separated: "Button, Label, TreeView"

    Returns:
        List of class names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not class_names_str or not class_names_str.strip():
        raise ValueError("class_names cannot be empty or whitespace. Use null to list all available classes.")

    class_names_str = class_names_str.strip()

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


class GetUIClassDetailInput(BaseModel):
    """Input for get_ui_class_detail function.

    Provide class names in any convenient format - the system will handle the conversion automatically.
    """

    class_names: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Class names to look up. Accepts multiple flexible formats:

        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single class: "TreeView"
        - Native array: ["Button", "Label", "TreeView"]
        - JSON string: '["Button", "Label", "TreeView"]'
        - Comma-separated: "Button, Label, TreeView"
        - Empty/null: Lists all available classes

        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_UI_CLASS_DETAIL_DESCRIPTION = """Get detailed information about OmniUI classes - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- class_names: Class names in ANY convenient format:
  * Single class string: "TreeView"
  * Native array: ["Button", "Label", "TreeView"] ← WORKS DIRECTLY!
  * JSON string: '["Button", "Label", "TreeView"]'
  * Comma-separated: "Button, Label, TreeView"
  * Empty/null: Lists all available classes

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_class_detail(class_names=["TreeView", "Window"])
✅ Single string: get_ui_class_detail(class_names="TreeView")
✅ JSON string: get_ui_class_detail(class_names='["Button", "Label", "TreeView"]')
✅ Comma format: get_ui_class_detail(class_names="Button, Label, TreeView")
✅ List all: get_ui_class_detail() or get_ui_class_detail(class_names=null)

💡 FOR AI MODELS: You can pass arrays directly like ["TreeView", "Window"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- 80% faster when fetching multiple classes
- Single API call instead of multiple round-trips
- Efficient context window usage
- Maximum compatibility with all AI models

RETURNS:
- For single class: Standard JSON with class details
- For multiple classes: Array with all class details plus metadata
- Includes: full_name, methods, parent_classes, docstring, etc.
- Error handling for invalid/missing classes"""


class GetUIClassDetailConfig(FunctionBaseConfig, name="get_ui_class_detail"):
    """Configuration for get_ui_class_detail function."""

    name: str = "get_ui_class_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUIClassDetailConfig, framework_wrappers=[])
async def register_get_ui_class_detail(config: GetUIClassDetailConfig, builder: Builder):
    """Register get_ui_class_detail function with AIQ."""

    # Access config fields here
    if config.verbose:
        logger.info("Registering get_ui_class_detail in verbose mode")

    async def get_ui_class_detail_wrapper(input: GetUIClassDetailInput) -> str:
        """Wrapper for get_class_detail function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.class_names is None:
                classes_to_fetch = None
            elif isinstance(input.class_names, list):
                # Direct array input - validate and use as-is
                if len(input.class_names) == 0:
                    return "ERROR: class_names array cannot be empty. Use null to list all available classes."
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.class_names):
                        if not isinstance(item, str):
                            return f"ERROR: All items in class_names array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in class_names array"
                    classes_to_fetch = [item.strip() for item in input.class_names]
            elif isinstance(input.class_names, str):
                # String input - parse using existing logic
                classes_to_fetch = _parse_class_names_input(input.class_names)
            else:
                return f"ERROR: class_names must be a string, array, or null, got {type(input.class_names).__name__}"

            parameters = {"class_names": input.class_names}
        except ValueError as e:
            return f"ERROR: Invalid class_names parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_class_detail(classes_to_fetch)

            # Use config fields to modify behavior
            if config.verbose:
                if isinstance(classes_to_fetch, list):
                    logger.debug(f"Retrieved details for {len(classes_to_fetch)} OmniUI classes")
                elif classes_to_fetch is None:
                    logger.debug("Retrieved available classes information")
                else:
                    logger.debug(f"Retrieved details for OmniUI class: {classes_to_fetch}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve class detail - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_ui_class_detail",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_class_detail: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_ui_class_detail_wrapper,
        description=GET_UI_CLASS_DETAIL_DESCRIPTION,
        input_schema=GetUIClassDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
