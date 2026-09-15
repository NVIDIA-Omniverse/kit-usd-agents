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

"""Registration wrapper for list_usd_modules function."""

import logging

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

# Removed shared config imports
from .functions.get_usd_modules import get_usd_modules
from .utils.usage_logging_decorator import log_tool_usage

logger = logging.getLogger(__name__)


def _sanitize_log(s: object, *, cap: int = 200) -> str:
    text = str(s if s is not None else "")
    return text.replace("\r", " ").replace("\n", " ")[:cap]


# Input schema for zero argument function
class ListUSDModulesInput(BaseModel):
    """Input schema for list_usd_modules (no arguments)."""

    pass


# Tool description
LIST_USD_MODULES_DESCRIPTION = """Enumerate every pxr.* module known to the USD Atlas — useful for discovering what USD modules exist.

WHEN TO USE THIS TOOL:
- "Which pxr modules are available?"
- You need to confirm a module name (Sdf, Usd, UsdGeom, UsdShade, …) before calling get_usd_module_detail.
- Producing a sitemap of USD APIs.

ARGUMENTS:
- (no arguments)

RETURNS:
JSON containing module names, full names, file paths, the classes and functions belonging to each module, and summary statistics (total modules / classes / functions).

USAGE EXAMPLES:
list_usd_modules

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Full details for a named module → use get_usd_module_detail.
- Enumerate classes, not modules → use list_usd_classes.
- Conceptual / how-to questions → use search_usd_knowledge.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand.
"""


class ListUSDModulesConfig(FunctionBaseConfig, name="list_usd_modules"):
    """Configuration for list_usd_modules function."""

    name: str = "list_usd_modules"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=ListUSDModulesConfig, framework_wrappers=[])
async def register_list_usd_modules(config: ListUSDModulesConfig, builder: Builder):
    """Register list_usd_modules function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering list_usd_modules in verbose mode")

    @log_tool_usage("list_usd_modules")
    async def list_usd_modules_wrapper(input: ListUSDModulesInput) -> str:
        """Zero arguments - schema required."""
        try:
            result = await get_usd_modules()

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved USD modules")

            if result["success"]:
                return result["result"]
            else:
                return f"ERROR: {_sanitize_log(result['error'], cap=500)}"

        except Exception as e:
            return f"ERROR: Failed to retrieve USD modules - {_sanitize_log(str(e), cap=500)}"

    # Pass input_schema for zero argument function
    function_info = FunctionInfo.from_fn(
        list_usd_modules_wrapper,
        description=LIST_USD_MODULES_DESCRIPTION,
        input_schema=ListUSDModulesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
