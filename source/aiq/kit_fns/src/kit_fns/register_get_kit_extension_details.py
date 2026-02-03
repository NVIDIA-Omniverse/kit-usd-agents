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

"""Registration wrapper for get_extension_details function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_extension_details import get_extension_details
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_extension_ids_input(extension_ids_str: str) -> Union[List[str], None]:
    """Parse extension_ids string input into a list of extension IDs."""
    if not extension_ids_str or not extension_ids_str.strip():
        return None

    extension_ids_str = extension_ids_str.strip()

    # Try to parse as JSON array first
    if extension_ids_str.startswith("[") and extension_ids_str.endswith("]"):
        try:
            parsed = json.loads(extension_ids_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in extension_ids_str:
        return [name.strip() for name in extension_ids_str.split(",") if name.strip()]

    # Single extension ID
    return [extension_ids_str]


class GetKitExtensionDetailsInput(BaseModel):
    """Input for get_extension_details function."""

    extension_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Extension IDs to retrieve. Accepts multiple flexible formats:

        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single extension: "omni.ui"
        - Native array: ["omni.ui", "omni.kit.window.console"] ← WORKS DIRECTLY!
        - JSON string: '["omni.ui", "omni.kit.window.console"]'
        - Comma-separated: "omni.ui, omni.kit.window.console"
        - Empty/null: Lists all available extensions

        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_KIT_EXTENSION_DETAILS_DESCRIPTION = """Get detailed information about Kit extensions - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- extension_ids: Extension IDs in ANY convenient format:
  * Single ID: "omni.ui"
  * Native array: ["omni.ui", "omni.kit.window.console"] ← WORKS DIRECTLY!
  * JSON string: '["omni.ui", "omni.kit.window.console"]'
  * Comma-separated: "omni.ui, omni.kit.window.console"
  * Empty/null: Lists all available extensions

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_extension_details(extension_ids=["omni.ui", "omni.ui.scene"])
✅ Single string: get_extension_details(extension_ids="omni.ui")
✅ JSON string: get_extension_details(extension_ids='["omni.ui", "omni.kit.window.console"]')
✅ Comma format: get_extension_details(extension_ids="omni.ui, omni.ui.scene")
✅ List all: get_extension_details() or get_extension_details(extension_ids=null)

💡 FOR AI MODELS: You can pass arrays directly like ["omni.ui", "omni.ui.scene"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- 70% faster when fetching multiple extensions
- Single API call instead of multiple round-trips
- Efficient context window usage
- Maximum compatibility with all AI models

RETURNS:
- For single extension: Complete extension information (2-4k tokens max)
- For multiple extensions: Array with all extension details plus summary
- Includes: features, dependencies, configuration options, usage patterns
- Error handling for invalid/missing extensions


NOTES:
- long_description is derived from Overview.md. If it exceeds 500 characters, the response includes a concise summary (< 500 chars)."""


class GetKitExtensionDetailsConfig(FunctionBaseConfig, name="get_kit_extension_details"):
    """Configuration for get_extension_details function."""

    name: str = "get_kit_extension_details"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetKitExtensionDetailsConfig, framework_wrappers=[])
async def register_get_kit_extension_details(config: GetKitExtensionDetailsConfig, builder: Builder):
    """Register get_extension_details function with AIQ."""

    verbose = config.verbose

    if verbose:
        logger.info(f"Registering get_kit_extension_details in verbose mode")

    async def get_kit_extension_details_wrapper(input: GetKitExtensionDetailsInput) -> str:
        """Wrapper for get_extension_details function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.extension_ids is None:
                extension_ids_to_fetch = None
            elif isinstance(input.extension_ids, list):
                if len(input.extension_ids) == 0:
                    extension_ids_to_fetch = None  # Empty array = list all
                else:
                    for i, item in enumerate(input.extension_ids):
                        if not isinstance(item, str):
                            return f"ERROR: All items in extension_ids array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in extension_ids array"
                    extension_ids_to_fetch = [item.strip() for item in input.extension_ids]
            elif isinstance(input.extension_ids, str):
                extension_ids_to_fetch = _parse_extension_ids_input(input.extension_ids)
            else:
                return (
                    f"ERROR: extension_ids must be a string, array, or null, got {type(input.extension_ids).__name__}"
                )

            parameters = {"extension_ids": input.extension_ids}
        except ValueError as e:
            return f"ERROR: Invalid extension_ids parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            result = await get_extension_details(extension_ids_to_fetch)

            if verbose:
                if isinstance(extension_ids_to_fetch, list):
                    logger.debug(f"Retrieved details for {len(extension_ids_to_fetch)} Kit extensions")
                elif extension_ids_to_fetch is None:
                    logger.debug("Retrieved available extensions information")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve extension details - {error_msg}"
        finally:
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_kit_extension_details",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_extension_details: {log_error}")

    function_info = FunctionInfo.from_fn(
        get_kit_extension_details_wrapper,
        description=GET_KIT_EXTENSION_DETAILS_DESCRIPTION,
        input_schema=GetKitExtensionDetailsInput,
    )

    function_info.metadata = {"mcp_exposed": True}

    yield function_info
