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

"""Registration wrapper for get_usd_module_detail function."""

import logging

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field


class GetUSDModuleDetailInput(BaseModel):
    """Input for get_usd_module_detail function."""

    module_names: str = Field(description="Comma-separated list of USD module names to get details for")


# Removed shared config imports
from .functions.get_usd_module_detail import get_usd_module_detail
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)

# Tool description
GET_USD_MODULE_DETAIL_DESCRIPTION = """Return detailed information about specific USD modules, including their classes and functions.

WHAT IT DOES:
- Retrieves comprehensive details about one or more USD modules
- Shows all classes and functions within each module
- Provides module metadata (name, full name, file path)
- Supports fuzzy matching for module names

ARGUMENTS:
- module_names (str): Comma-separated list of module names

RETURNS:
JSON string with detailed information about the modules including:
- Module information (name, full name, file path)
- List of all classes in the module
- List of all functions in the module
- Summary statistics

USAGE EXAMPLES:
get_usd_module_detail "pxr.Usd"
get_usd_module_detail "Usd,UsdGeom"
get_usd_module_detail "pxr.Usd,pxr.UsdGeom,pxr.UsdShade"

TIPS:
- Module names support fuzzy matching
- You can use short names (e.g., "Usd") or full names (e.g., "pxr.Usd")
- Multiple modules can be queried at once using comma separation
"""


class GetUSDModuleDetailConfig(FunctionBaseConfig, name="get_usd_module_detail"):
    """Configuration for get_usd_module_detail function."""

    name: str = "get_usd_module_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUSDModuleDetailConfig, framework_wrappers=[])
async def register_get_usd_module_detail(config: GetUSDModuleDetailConfig, builder: Builder):
    """Register get_usd_module_detail function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_usd_module_detail in verbose mode")

    @log_tool_usage("get_usd_module_detail")
    async def get_usd_module_detail_wrapper(module_names: str) -> str:
        """Single argument - no schema needed."""
        try:
            result = await get_usd_module_detail(module_names)

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved module details for: {module_names}")

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {result['error']}"

        except Exception as e:
            return f"ERROR: Failed to retrieve module details - {str(e)}"

    function_info = FunctionInfo.from_fn(
        get_usd_module_detail_wrapper,
        description=GET_USD_MODULE_DETAIL_DESCRIPTION,
        input_schema=GetUSDModuleDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
