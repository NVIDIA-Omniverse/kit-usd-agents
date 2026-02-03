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

"""Registration wrapper for get_api_details function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_api_details import get_api_details
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_api_references_input(api_references_str: str) -> Union[List[str], None]:
    """Parse api_references string input into a list of API references."""
    if not api_references_str or not api_references_str.strip():
        return None

    api_references_str = api_references_str.strip()

    if api_references_str.startswith("[") and api_references_str.endswith("]"):
        try:
            parsed = json.loads(api_references_str)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    if "," in api_references_str:
        return [name.strip() for name in api_references_str.split(",") if name.strip()]

    return [api_references_str]


class GetAPIDetailsInput(BaseModel):
    """Input for get_api_details function."""

    api_references: Optional[Union[str, List[str]]] = Field(
        None,
        description="""API references in format 'extension_id@symbol'. Accepts:
        - Single: 'omni.ui@Window'
        - Array: ['omni.ui@Window', 'omni.ui@Button']
        - JSON: '["omni.ui@Window", "omni.ui@Button"]'
        - Comma: 'omni.ui@Window, omni.ui@Button'
        - Empty/null: Lists available API references""",
    )

    model_config = {"extra": "forbid"}


GET_KIT_API_DETAILS_DESCRIPTION = """Get detailed Kit API documentation - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON!

PARAMETER:
- api_references: API references in format 'extension_id@symbol':
  * Single API: "omni.ui@Window"
  * Native array: ["omni.ui@Window", "omni.ui@Button"] ← WORKS DIRECTLY!
  * JSON string: '["omni.ui@Window", "omni.ui@Button"]'
  * Comma-separated: "omni.ui@Window, omni.ui@Button"
  * Empty/null: Lists all available API references

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_api_details(api_references=["omni.ui@Window", "omni.ui@Button"])
✅ Single string: get_api_details(api_references="omni.ui@Window")
✅ JSON string: get_api_details(api_references='["omni.ui@Window", "omni.ui@Button"]')
✅ Comma format: get_api_details(api_references="omni.ui@Window, omni.ui@Button") 
✅ List all: get_api_details() or get_api_details(api_references=null)

RETURNS:
Complete API documentation with:
- Full docstrings and descriptions
- Method signatures and parameters
- Return types and exceptions
- Property information
- Usage examples
- Error handling for invalid/missing APIs"""


class GetAPIDetailsConfig(FunctionBaseConfig, name="get_kit_api_details"):
    """Configuration for get_api_details function."""

    name: str = "get_kit_api_details"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetAPIDetailsConfig, framework_wrappers=[])
async def register_get_kit_api_details(config: GetAPIDetailsConfig, builder: Builder):
    """Register get_api_details function with AIQ."""

    verbose = config.verbose

    if verbose:
        logger.info(f"Registering get_kit_api_details in verbose mode")

    async def get_kit_api_details_wrapper(input: GetAPIDetailsInput) -> str:
        """Wrapper for get_api_details function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        try:
            if input.api_references is None:
                api_references_to_fetch = None
            elif isinstance(input.api_references, list):
                if len(input.api_references) == 0:
                    api_references_to_fetch = None
                else:
                    for i, item in enumerate(input.api_references):
                        if not isinstance(item, str):
                            return f"ERROR: All items in api_references array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in api_references array"
                    api_references_to_fetch = [item.strip() for item in input.api_references]
            elif isinstance(input.api_references, str):
                api_references_to_fetch = _parse_api_references_input(input.api_references)
            else:
                return (
                    f"ERROR: api_references must be a string, array, or null, got {type(input.api_references).__name__}"
                )

            parameters = {"api_references": input.api_references}
        except ValueError as e:
            return f"ERROR: Invalid api_references parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            result = await get_api_details(api_references_to_fetch)

            if verbose and api_references_to_fetch:
                logger.debug(f"Retrieved details for {len(api_references_to_fetch)} Kit APIs")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve API details - {error_msg}"
        finally:
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_kit_api_details",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_api_details: {log_error}")

    function_info = FunctionInfo.from_fn(
        get_kit_api_details_wrapper,
        description=GET_KIT_API_DETAILS_DESCRIPTION,
        input_schema=GetAPIDetailsInput,
    )

    function_info.metadata = {"mcp_exposed": True}

    yield function_info
