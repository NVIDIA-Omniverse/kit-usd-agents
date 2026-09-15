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

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

# Removed shared config imports
from .functions.get_usd_method_detail import get_usd_method_detail
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)


def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


def _parse_method_names_input(method_names: Union[str, List[str]]) -> str:
    """Normalize native arrays, JSON-array strings, and comma strings."""
    if isinstance(method_names, list):
        if not method_names:
            raise ValueError("method_names array cannot be empty")
        for i, item in enumerate(method_names):
            if not isinstance(item, str):
                raise ValueError(
                    f"All items in method_names array must be strings, got {type(item).__name__} at index {i}"
                )
            if not item.strip():
                raise ValueError(f"Empty string at index {i} in method_names array")
        return ",".join(item.strip() for item in method_names)

    if not isinstance(method_names, str):
        raise ValueError(f"method_names must be a string or array, got {type(method_names).__name__}")

    value = method_names.strip()
    if not value:
        raise ValueError("method_names cannot be empty")
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}") from e
        return _parse_method_names_input(parsed)
    return value


# Define input schema for multiple arguments
class GetUSDMethodDetailInput(BaseModel):
    """Input parameters for USD method detail retrieval."""

    method_names: Union[str, List[str]] = Field(
        description="USD method names as a single string, native array, JSON-array string, or comma-separated string"
    )
    class_name: Optional[str] = Field(
        default="",
        description="Optional class name to narrow down the search for all methods",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_USD_METHOD_DETAIL_DESCRIPTION = """Inspect one or more pxr.* methods — signature, arguments, return type, and docstring; optionally scoped to a class.

WHEN TO USE THIS TOOL:
- "What does UsdStage.GetPrimAtPath take and return?"
- Disambiguating overloaded or inherited methods.
- Batch-looking-up several methods on the same class.

ARGUMENTS:
- method_names (str | list[str]): method names (fuzzy-matched; accepts a native list, JSON-array string, or comma-separated string).
- class_name (str, optional): class to scope the search to (searches class + ancestors).

RETURNS:
JSON per method with full name, signature, docstring, arguments (with types), return type, the owning class, and whether the method is inherited. Up to 5 best matches per query.

USAGE EXAMPLES:
get_usd_method_detail {"method_names": "GetPrim"}
get_usd_method_detail {"method_names": ["GetPrim","CreatePrim"], "class_name": "UsdStage"}
get_usd_method_detail {"method_names": "GetPrim,CreatePrim", "class_name": "UsdStage"}
get_usd_method_detail {"method_names": "Clear,IsValid", "class_name": "Attribute"}

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- You want the whole class API → use get_usd_class_detail.
- You don't know the class / method name → use search_usd_knowledge or list_usd_classes.
- You want working code → use search_usd_code_examples.
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
            method_names = _parse_method_names_input(input.method_names)
            result = await get_usd_method_detail(method_names=method_names, class_name=input.class_name or "")

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved method details for: {_sanitize_log(method_names)}")

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {_sanitize_log(result['error'], cap=500)}"

        except Exception as e:
            return f"ERROR: Failed to retrieve method details - {_sanitize_log(str(e), cap=500)}"

    # Pass input_schema for multiple argument function
    function_info = FunctionInfo.from_fn(
        get_usd_method_detail_wrapper,
        description=GET_USD_METHOD_DETAIL_DESCRIPTION,
        input_schema=GetUSDMethodDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
