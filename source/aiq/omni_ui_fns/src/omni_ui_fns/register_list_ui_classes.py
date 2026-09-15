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

"""Registration wrapper for list_ui_classes function."""

import logging

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_classes import get_classes
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


# Define input schema for zero-argument function
class ListUIClassesInput(BaseModel):
    """Empty input for zero-argument function."""

    pass


# Tool description
LIST_UI_CLASSES_DESCRIPTION = """Enumerate every omni.ui class known to the UI Atlas — flat alphabetical listing for discovery.

WHEN TO USE THIS TOOL:
- "Which OmniUI classes exist?"
- You need to confirm a class name (Button, Label, TreeView, Window, …) before calling get_ui_class_detail.
- Auditing the OmniUI widget / layout / style surface.

ARGUMENTS:
- (no arguments)

RETURNS:
JSON with `class_full_names` (sorted), `total_count`, and a brief description. Covers widgets, layout components, style classes, and testing utilities.

USAGE EXAMPLES:
list_ui_classes

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Full docstring / methods of a named class → use get_ui_class_detail.
- Usage patterns for a class → use get_ui_class_instructions.
- Classes grouped by module → use list_ui_modules or get_ui_module_detail.
- Working code → use search_ui_code_examples.

Abbreviation tip: the retriever auto-expands common Omniverse abbreviations (SSS, PBR, DLSS, LIVRPS, Gf/Sdf/UsdGeom, etc.). Write the natural term — you don't have to pre-expand.
"""


class ListUIClassesConfig(FunctionBaseConfig, name="list_ui_classes"):
    """Configuration for list_ui_classes function."""

    name: str = "list_ui_classes"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=ListUIClassesConfig, framework_wrappers=[])
async def register_list_ui_classes(config: ListUIClassesConfig, builder: Builder):
    """Register list_ui_classes function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering list_ui_classes in verbose mode")

    async def list_ui_classes_wrapper(input: ListUIClassesInput) -> str:
        """Zero arguments - schema required."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {}
        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_classes()

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved OmniUI classes")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve OmniUI classes - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="list_ui_classes",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for list_ui_classes: {log_error}")

    # Pass input_schema for zero argument function
    function_info = FunctionInfo.from_fn(
        list_ui_classes_wrapper,
        description=LIST_UI_CLASSES_DESCRIPTION,
        input_schema=ListUIClassesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
