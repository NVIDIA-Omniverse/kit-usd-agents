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

"""Registration wrapper for get_usd_class_detail function."""

import logging

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field


class GetUSDClassDetailInput(BaseModel):
    """Input for get_usd_class_detail function."""

    class_names: str = Field(description="Comma-separated list of USD class names to get details for")


# Removed shared config imports
from .functions.get_usd_class_detail import get_usd_class_detail
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)

# Tool description
GET_USD_CLASS_DETAIL_DESCRIPTION = """Return detailed information about specific USD classes, including their methods.

WHAT IT DOES:
- Retrieves comprehensive details about one or more USD classes
- Shows all methods (both own and inherited from parent classes)
- Displays class variables and their type annotations
- Provides class metadata (docstring, parent classes, module)
- Supports fuzzy matching for class names

ARGUMENTS:
- class_names (str): Comma-separated list of class names

RETURNS:
JSON string with detailed information about the classes including:
- Class information (name, full name, module, docstring)
- Methods (own, inherited, and all combined)
- Class variables (own, inherited, and all combined)
- Parent classes hierarchy
- Summary statistics

USAGE EXAMPLES:
get_usd_class_detail "UsdStage"
get_usd_class_detail "UsdStage,UsdPrim"
get_usd_class_detail "pxr.Usd.Stage,pxr.Usd.Prim,pxr.UsdGeom.Mesh"

TIPS:
- Class names support fuzzy matching
- You can use short names (e.g., "Stage") or full names (e.g., "pxr.Usd.Stage")
- Multiple classes can be queried at once using comma separation
- The tool shows inherited methods and variables from parent classes
"""


class GetUSDClassDetailConfig(FunctionBaseConfig, name="get_usd_class_detail"):
    """Configuration for get_usd_class_detail function."""

    name: str = "get_usd_class_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUSDClassDetailConfig, framework_wrappers=[])
async def register_get_usd_class_detail(config: GetUSDClassDetailConfig, builder: Builder):
    """Register get_usd_class_detail function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_usd_class_detail in verbose mode")

    @log_tool_usage("get_usd_class_detail")
    async def get_usd_class_detail_wrapper(class_names: str) -> str:
        """Single argument - no schema needed."""
        try:
            result = await get_usd_class_detail(class_names)

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved class details for: {class_names}")

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {result['error']}"

        except Exception as e:
            return f"ERROR: Failed to retrieve class details - {str(e)}"

    function_info = FunctionInfo.from_fn(
        get_usd_class_detail_wrapper,
        description=GET_USD_CLASS_DETAIL_DESCRIPTION,
        input_schema=GetUSDClassDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
