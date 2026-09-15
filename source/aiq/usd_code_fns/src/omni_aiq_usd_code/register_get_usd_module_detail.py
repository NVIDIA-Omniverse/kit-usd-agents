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

import json
import logging
from typing import List, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

# Removed shared config imports
from .functions.get_usd_module_detail import get_usd_module_detail
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)


def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


def _parse_module_names_input(module_names: Union[str, List[str]]) -> str:
    """Normalize native arrays, JSON-array strings, and comma strings."""
    if isinstance(module_names, list):
        if not module_names:
            raise ValueError("module_names array cannot be empty")
        for i, item in enumerate(module_names):
            if not isinstance(item, str):
                raise ValueError(
                    f"All items in module_names array must be strings, got {type(item).__name__} at index {i}"
                )
            if not item.strip():
                raise ValueError(f"Empty string at index {i} in module_names array")
        return ",".join(item.strip() for item in module_names)

    if not isinstance(module_names, str):
        raise ValueError(f"module_names must be a string or array, got {type(module_names).__name__}")

    value = module_names.strip()
    if not value:
        raise ValueError("module_names cannot be empty")
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}") from e
        return _parse_module_names_input(parsed)
    return value


class GetUSDModuleDetailInput(BaseModel):
    """Input for get_usd_module_detail function."""

    module_names: Union[str, List[str]] = Field(
        description="USD module names as a single string, native array, JSON-array string, or comma-separated string"
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_USD_MODULE_DETAIL_DESCRIPTION = """Inspect one or more pxr.* modules — lists every class and function the module exposes, plus module metadata.

WHEN TO USE THIS TOOL:
- "What's in pxr.Sdf / pxr.UsdGeom / pxr.UsdShade?"
- Picking the right class from a module before calling get_usd_class_detail.
- Comparing two modules' contents side-by-side.

ARGUMENTS:
- module_names (str | list[str]): module names (short "Usd" or full "pxr.Usd" both work; accepts a native list, JSON-array string, or comma-separated string).

RETURNS:
JSON with per-module metadata (name, full name, file path), the classes and functions in each module, and summary statistics.

USAGE EXAMPLES:
get_usd_module_detail "pxr.Usd"
get_usd_module_detail ["Usd","UsdGeom"]
get_usd_module_detail "Usd,UsdGeom"
get_usd_module_detail "pxr.Usd,pxr.UsdGeom,pxr.UsdShade"

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- You don't know which module to ask about → use list_usd_modules.
- You want a specific class → use get_usd_class_detail.
- You want a specific method → use get_usd_method_detail.
- Conceptual questions → use search_usd_knowledge.
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
    async def get_usd_module_detail_wrapper(input: GetUSDModuleDetailInput) -> str:
        """Schema wrapper accepting string or native list input."""
        try:
            module_names = _parse_module_names_input(input.module_names)
            result = await get_usd_module_detail(module_names)

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved module details for: {_sanitize_log(module_names)}")

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {_sanitize_log(result['error'], cap=500)}"

        except Exception as e:
            return f"ERROR: Failed to retrieve module details - {_sanitize_log(str(e), cap=500)}"

    function_info = FunctionInfo.from_fn(
        get_usd_module_detail_wrapper,
        description=GET_USD_MODULE_DETAIL_DESCRIPTION,
        input_schema=GetUSDModuleDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
