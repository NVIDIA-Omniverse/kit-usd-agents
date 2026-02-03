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

"""Registration wrapper for get_usd_method_detail function."""

import logging
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

# Removed shared config imports
from .functions.get_usd_method_detail import get_usd_method_detail
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)


# Define input schema for multiple arguments
class GetUSDMethodDetailInput(BaseModel):
    """Input parameters for USD method detail retrieval."""

    method_names: str = Field(
        description="Comma-separated list of USD method names to look up (e.g., 'GetPrim,CreatePrim')"
    )
    class_name: Optional[str] = Field(
        default="", description="Optional class name to narrow down the search for all methods"
    )


# Tool description
GET_USD_METHOD_DETAIL_DESCRIPTION = """Return detailed information about specific USD methods, including docstring, arguments, and return types.

WHAT IT DOES:
- Retrieves comprehensive details about one or more USD methods
- Shows method signatures, arguments, and return types
- Provides method docstrings
- Can search within a specific class or across all classes
- Supports searching in ancestor classes if method not found in specified class

ARGUMENTS:
- method_names (str): Comma-separated list of USD method names to look up
- class_name (Optional[str]): Optional class name to narrow down the search

RETURNS:
JSON string with detailed information about the USD methods including:
- Method full name and signature
- Docstring documentation
- Arguments with their types
- Return type information
- Class the method belongs to
- Whether the method is inherited from a parent class

USAGE EXAMPLES:
get_usd_method_detail {"method_names": "GetPrim"}
get_usd_method_detail {"method_names": "GetPrim,CreatePrim", "class_name": "UsdStage"}
get_usd_method_detail {"method_names": "Clear,IsValid", "class_name": "Attribute"}

TIPS:
- Method names support fuzzy matching
- If a class_name is provided, it will search that class and its ancestors
- Multiple methods can be queried at once using comma separation
- The tool shows up to 5 best matches for each method query
"""


class GetUSDMethodDetailConfig(FunctionBaseConfig, name="get_usd_method_detail"):
    """Configuration for get_usd_method_detail function."""

    name: str = "get_usd_method_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUSDMethodDetailConfig, framework_wrappers=[])
async def register_get_usd_method_detail(config: GetUSDMethodDetailConfig, builder: Builder):
    """Register get_usd_method_detail function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_usd_method_detail in verbose mode")

    @log_tool_usage("get_usd_method_detail")
    async def get_usd_method_detail_wrapper(input: GetUSDMethodDetailInput) -> str:
        """Multiple arguments - schema required."""
        try:
            result = await get_usd_method_detail(method_names=input.method_names, class_name=input.class_name)

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved method details for: {input.method_names}")

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {result['error']}"

        except Exception as e:
            return f"ERROR: Failed to retrieve method details - {str(e)}"

    # Pass input_schema for multiple argument function
    function_info = FunctionInfo.from_fn(
        get_usd_method_detail_wrapper,
        description=GET_USD_METHOD_DETAIL_DESCRIPTION,
        input_schema=GetUSDMethodDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
