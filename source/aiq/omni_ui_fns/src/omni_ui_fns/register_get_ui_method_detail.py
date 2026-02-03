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

"""Registration wrapper for get_ui_method_detail function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_method_detail import get_method_detail
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_method_names_input(method_names_str: str) -> Union[List[str], None]:
    """Parse method_names string input into a list of method names.

    Args:
        method_names_str: String input that can be:
            - None/empty: Return None (get all available methods)
            - Single method: "__init__" or "clicked_fn"
            - JSON array: '["__init__", "clicked_fn", "set_value", "get_value"]'
            - Comma-separated: "__init__, clicked_fn, set_value, get_value"

    Returns:
        List of method names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not method_names_str or not method_names_str.strip():
        return None

    method_names_str = method_names_str.strip()

    # Try to parse as JSON array first
    if method_names_str.startswith("[") and method_names_str.endswith("]"):
        try:
            parsed = json.loads(method_names_str)
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
    if "," in method_names_str:
        return [name.strip() for name in method_names_str.split(",") if name.strip()]

    # Single method name
    return [method_names_str]


class GetUIMethodDetailInput(BaseModel):
    """Input for get_ui_method_detail function.

    Provide method names in any convenient format - the system will handle the conversion automatically.
    """

    method_names: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Method names to look up. Accepts multiple flexible formats:

        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single method: "__init__"
        - Native array: ["__init__", "clicked_fn", "set_value", "get_value"] ← WORKS DIRECTLY!
        - JSON string: '["__init__", "clicked_fn", "set_value", "get_value"]'
        - Comma-separated: "__init__, clicked_fn, set_value, get_value"
        - Empty/null: Lists all available methods

        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_UI_METHOD_DETAIL_DESCRIPTION = """Get detailed information about OmniUI methods - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- method_names: Method names in ANY convenient format:
  * Single method string: "__init__"
  * Native array: ["__init__", "clicked_fn", "set_value", "get_value"] ← WORKS DIRECTLY!
  * JSON string: '["__init__", "clicked_fn", "set_value", "get_value"]'
  * Comma-separated: "__init__, clicked_fn, set_value, get_value"
  * Empty/null: Lists all available methods

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_method_detail(method_names=["__init__", "clicked_fn"])
✅ Single string: get_ui_method_detail(method_names="__init__")
✅ JSON string: get_ui_method_detail(method_names='["__init__", "clicked_fn", "set_value", "get_value"]')
✅ Comma format: get_ui_method_detail(method_names="__init__, clicked_fn, set_value, get_value")
✅ List all: get_ui_method_detail() or get_ui_method_detail(method_names=null)

💡 FOR AI MODELS: You can pass arrays directly like ["__init__", "clicked_fn"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- 60-80% faster when fetching multiple methods
- Single API call instead of multiple round-trips
- Efficient context window usage
- Maximum compatibility with all AI models

RETURNS:
- For single method: Standard JSON with method details
- For multiple methods: Array with all method details plus metadata
- Includes: signatures, parameters, return types, docstrings for each method
- Error handling for invalid/missing methods"""


class GetUIMethodDetailConfig(FunctionBaseConfig, name="get_ui_method_detail"):
    """Configuration for get_ui_method_detail function."""

    name: str = "get_ui_method_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUIMethodDetailConfig, framework_wrappers=[])
async def register_get_ui_method_detail(config: GetUIMethodDetailConfig, builder: Builder):
    """Register get_ui_method_detail function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_method_detail in verbose mode")

    async def get_ui_method_detail_wrapper(input: GetUIMethodDetailInput) -> str:
        """Wrapper for get_method_detail function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.method_names is None:
                methods_to_fetch = None
            elif isinstance(input.method_names, list):
                # Direct array input - validate and use as-is
                if len(input.method_names) == 0:
                    methods_to_fetch = None  # Empty array = list all
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.method_names):
                        if not isinstance(item, str):
                            return f"ERROR: All items in method_names array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in method_names array"
                    methods_to_fetch = [item.strip() for item in input.method_names]
            elif isinstance(input.method_names, str):
                # String input - parse using existing logic
                methods_to_fetch = _parse_method_names_input(input.method_names)
            else:
                return f"ERROR: method_names must be a string, array, or null, got {type(input.method_names).__name__}"

            parameters = {"method_names": input.method_names}
        except ValueError as e:
            return f"ERROR: Invalid method_names parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_method_detail(methods_to_fetch)

            # Use config fields to modify behavior
            if verbose:
                if isinstance(methods_to_fetch, list):
                    logger.debug(f"Retrieved details for {len(methods_to_fetch)} OmniUI methods")
                elif methods_to_fetch is None:
                    logger.debug("Retrieved available methods information")
                else:
                    logger.debug(f"Retrieved details for OmniUI method: {methods_to_fetch}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve method detail - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_ui_method_detail",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_method_detail: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_ui_method_detail_wrapper,
        description=GET_UI_METHOD_DETAIL_DESCRIPTION,
        input_schema=GetUIMethodDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
