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

"""Registration wrapper for get_ui_module_detail function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_module_detail import get_module_detail
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_module_names_input(module_names_str: str) -> Union[List[str], None]:
    """Parse module_names string input into a list of module names.

    Args:
        module_names_str: String input that can be:
            - None/empty: Return None (get all available modules)
            - Single module: "omni.ui"
            - JSON array: '["omni.ui", "omni.ui.scene", "omni.ui.workspace"]'
            - Comma-separated: "omni.ui, omni.ui.scene, omni.ui.workspace"

    Returns:
        List of module names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not module_names_str or not module_names_str.strip():
        raise ValueError("module_names cannot be empty or whitespace. Use null to list all available modules.")

    module_names_str = module_names_str.strip()

    # Try to parse as JSON array first
    if module_names_str.startswith("[") and module_names_str.endswith("]"):
        try:
            parsed = json.loads(module_names_str)
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
    if "," in module_names_str:
        return [name.strip() for name in module_names_str.split(",") if name.strip()]

    # Single module name
    return [module_names_str]


class GetUIModuleDetailInput(BaseModel):
    """Input for get_ui_module_detail function.

    Provide module names in any convenient format - the system will handle the conversion automatically.
    """

    module_names: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Module names to look up. Accepts multiple flexible formats:
        
        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single module: "omni.ui"
        - Native array: ["omni.ui", "omni.ui.scene", "omni.ui.workspace"] ← WORKS DIRECTLY!
        - JSON string: '["omni.ui", "omni.ui.scene", "omni.ui.workspace"]'
        - Comma-separated: "omni.ui, omni.ui.scene, omni.ui.workspace"
        - Empty/null: Lists all available modules
        
        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_UI_MODULE_DETAIL_DESCRIPTION = """Get detailed information about OmniUI modules - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- module_names: Module names in ANY convenient format:
  * Single module string: "omni.ui"
  * Native array: ["omni.ui", "omni.ui.scene", "omni.ui.workspace"] ← WORKS DIRECTLY!
  * JSON string: '["omni.ui", "omni.ui.scene", "omni.ui.workspace"]'
  * Comma-separated: "omni.ui, omni.ui.scene, omni.ui.workspace"
  * Empty/null: Lists all available modules

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_module_detail(module_names=["omni.ui", "omni.ui.scene"])
✅ Single string: get_ui_module_detail(module_names="omni.ui")
✅ JSON string: get_ui_module_detail(module_names='["omni.ui", "omni.ui.scene", "omni.ui.workspace"]')
✅ Comma format: get_ui_module_detail(module_names="omni.ui, omni.ui.scene, omni.ui.workspace") 
✅ List all: get_ui_module_detail() or get_ui_module_detail(module_names=null)

💡 FOR AI MODELS: You can pass arrays directly like ["omni.ui", "omni.ui.scene"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- 70% faster when fetching multiple modules
- Single API call instead of multiple round-trips
- Efficient context window usage
- Maximum compatibility with all AI models

RETURNS:
- For single module: Standard JSON with module details
- For multiple modules: Array with all module details plus metadata
- Includes: classes, functions, file paths, extensions for each module
- Error handling for invalid/missing modules"""


class GetUIModuleDetailConfig(FunctionBaseConfig, name="get_ui_module_detail"):
    """Configuration for get_ui_module_detail function."""

    name: str = "get_ui_module_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUIModuleDetailConfig, framework_wrappers=[])
async def register_get_ui_module_detail(config: GetUIModuleDetailConfig, builder: Builder):
    """Register get_ui_module_detail function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info("Registering get_ui_module_detail in verbose mode")

    async def get_ui_module_detail_wrapper(input: GetUIModuleDetailInput) -> str:
        """Wrapper for get_module_detail function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.module_names is None:
                modules_to_fetch = None
            elif isinstance(input.module_names, list):
                # Direct array input - validate and use as-is
                if len(input.module_names) == 0:
                    return "ERROR: module_names array cannot be empty. Use null to list all available modules."
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.module_names):
                        if not isinstance(item, str):
                            return f"ERROR: All items in module_names array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in module_names array"
                    modules_to_fetch = [item.strip() for item in input.module_names]
            elif isinstance(input.module_names, str):
                # String input - parse using existing logic
                modules_to_fetch = _parse_module_names_input(input.module_names)
            else:
                return f"ERROR: module_names must be a string, array, or null, got {type(input.module_names).__name__}"

            parameters = {"module_names": input.module_names}
        except ValueError as e:
            return f"ERROR: Invalid module_names parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_module_detail(modules_to_fetch)

            # Use config fields to modify behavior
            if verbose:
                if isinstance(modules_to_fetch, list):
                    logger.debug(f"Retrieved details for {len(modules_to_fetch)} OmniUI modules")
                elif modules_to_fetch is None:
                    logger.debug("Retrieved available modules information")
                else:
                    logger.debug(f"Retrieved details for OmniUI module: {modules_to_fetch}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve module detail - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_ui_module_detail",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_module_detail: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_ui_module_detail_wrapper,
        description=GET_UI_MODULE_DETAIL_DESCRIPTION,
        input_schema=GetUIModuleDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
