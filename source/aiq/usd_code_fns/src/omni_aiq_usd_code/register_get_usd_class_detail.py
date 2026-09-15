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

import json
import logging
from typing import List, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

# Removed shared config imports
from .functions.get_usd_class_detail import get_usd_class_detail
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)


def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


def _parse_class_names_input(class_names: Union[str, List[str]]) -> str:
    """Normalize native arrays, JSON-array strings, and comma strings."""
    if isinstance(class_names, list):
        if not class_names:
            raise ValueError("class_names array cannot be empty")
        for i, item in enumerate(class_names):
            if not isinstance(item, str):
                raise ValueError(
                    f"All items in class_names array must be strings, got {type(item).__name__} at index {i}"
                )
            if not item.strip():
                raise ValueError(f"Empty string at index {i} in class_names array")
        return ",".join(item.strip() for item in class_names)

    if not isinstance(class_names, str):
        raise ValueError(f"class_names must be a string or array, got {type(class_names).__name__}")

    value = class_names.strip()
    if not value:
        raise ValueError("class_names cannot be empty")
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}") from e
        return _parse_class_names_input(parsed)
    return value


class GetUSDClassDetailInput(BaseModel):
    """Input for get_usd_class_detail function."""

    class_names: Union[str, List[str]] = Field(
        description="USD class names as a single string, native array, JSON-array string, or comma-separated string"
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_USD_CLASS_DETAIL_DESCRIPTION = """Inspect one or more pxr.* classes — full docstring, own & inherited methods, variables, and parent-class hierarchy.

WHEN TO USE THIS TOOL:
- "What methods does UsdStage / UsdPrim / UsdGeomMesh have?"
- Understanding a class's inheritance and full API surface before using it.
- Comparing several USD classes at once.

ARGUMENTS:
- class_names (str | list[str]): class names (short "Stage" or full "pxr.Usd.Stage"; accepts a native list, JSON-array string, or comma-separated string).

RETURNS:
JSON with class metadata (name, full name, module, docstring), methods (own + inherited), class variables (own + inherited), the parent hierarchy, and summary statistics.

USAGE EXAMPLES:
get_usd_class_detail "UsdStage"
get_usd_class_detail ["UsdStage","UsdPrim"]
get_usd_class_detail "UsdStage,UsdPrim"
get_usd_class_detail "pxr.Usd.Stage,pxr.UsdGeom.Mesh"

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- You don't know the class name → use list_usd_classes or search_usd_knowledge.
- You want a specific method's detail → use get_usd_method_detail.
- You want the enclosing module → use get_usd_module_detail.
- You want working code using the class → use search_usd_code_examples.
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
    async def get_usd_class_detail_wrapper(input: GetUSDClassDetailInput) -> str:
        """Schema wrapper accepting string or native list input."""
        try:
            class_names = _parse_class_names_input(input.class_names)
            result = await get_usd_class_detail(class_names)

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved class details for: {_sanitize_log(class_names)}")

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {_sanitize_log(result['error'], cap=500)}"

        except Exception as e:
            return f"ERROR: Failed to retrieve class details - {_sanitize_log(str(e), cap=500)}"

    function_info = FunctionInfo.from_fn(
        get_usd_class_detail_wrapper,
        description=GET_USD_CLASS_DETAIL_DESCRIPTION,
        input_schema=GetUSDClassDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
