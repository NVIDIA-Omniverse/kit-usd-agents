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

"""Registration wrapper for list_usd_classes function."""

import logging

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

# Removed shared config imports
from .functions.get_usd_classes import get_usd_classes
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)


def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


# Define input schema for zero-argument function
class ListUSDClassesInput(BaseModel):
    """Empty input for zero-argument function."""

    pass


# Tool description
LIST_USD_CLASSES_DESCRIPTION = """Enumerate every pxr.* class known to the USD Atlas — a flat alphabetical listing for discovery.

WHEN TO USE THIS TOOL:
- "What classes does USD expose?"
- You need to confirm a class name (e.g. UsdStage, UsdPrim, UsdGeomMesh) before calling get_usd_class_detail.
- Browsing / auditing the USD class surface.

ARGUMENTS:
- (no arguments)

RETURNS:
JSON with `class_full_names` (sorted list of full class names) and `total_count`.

USAGE EXAMPLES:
list_usd_classes

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Full docstring / methods of a specific class → use get_usd_class_detail.
- Classes grouped by module → use list_usd_modules or get_usd_module_detail.
- Conceptual questions → use search_usd_knowledge.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand.
"""


class ListUSDClassesConfig(FunctionBaseConfig, name="list_usd_classes"):
    """Configuration for list_usd_classes function."""

    name: str = "list_usd_classes"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=ListUSDClassesConfig, framework_wrappers=[])
async def register_list_usd_classes(config: ListUSDClassesConfig, builder: Builder):
    """Register list_usd_classes function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering list_usd_classes in verbose mode")

    @log_tool_usage("list_usd_classes")
    async def list_usd_classes_wrapper(input: ListUSDClassesInput) -> str:
        """Zero arguments - schema required."""
        try:
            result = await get_usd_classes()

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved USD classes")

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {_sanitize_log(result['error'], cap=500)}"

        except Exception as e:
            return f"ERROR: Failed to retrieve USD classes - {_sanitize_log(str(e), cap=500)}"

    # Pass input_schema for zero argument function
    function_info = FunctionInfo.from_fn(
        list_usd_classes_wrapper,
        description=LIST_USD_CLASSES_DESCRIPTION,
        input_schema=ListUSDClassesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
