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

"""Registration wrapper for get_ui_instructions function."""

import logging
from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_instructions import get_instructions, list_instructions
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


# Define input schema
class GetUIInstructionsInput(BaseModel):
    """Input schema for get_ui_instructions function."""

    name: Optional[str] = Field(
        default=None,
        description="""The name of the instruction set to retrieve. Valid values are:
- 'agent_system': Core Omniverse UI Assistant system prompt with omni.ui framework basics. Use for understanding omni.ui fundamentals, omni.ui.scene for 3D UI, widget filters, options menus, searchable comboboxes, and general code writing guidelines.
- 'classes': Comprehensive omni.ui class API reference and model patterns. Use for working with AbstractValueModel, data models, custom model implementations, callbacks, and model-view patterns.
- 'omni_ui_scene_system': Complete omni.ui.scene 3D UI system documentation. Use for creating 3D shapes, SceneView, camera controls, transforms, gestures, manipulators, and USD stage synchronization.
- 'omni_ui_system': Core omni.ui widgets, containers, layouts and styling. Use for basic UI shapes, widgets (Labels, Buttons, Fields, Sliders), layouts (HStack, VStack, ZStack, Grid), Window management, styling, drag & drop, and MDV pattern.

If not provided or None, lists all available instructions with their descriptions.""",
    )


# Tool description
GET_UI_INSTRUCTIONS_DESCRIPTION = """Retrieve OmniUI system instructions and documentation for code generation and UI development.

WHAT IT DOES:
- Retrieves specific OmniUI system instruction documents
- Provides comprehensive documentation for different aspects of omni.ui
- Lists all available instructions when no name is specified
- Returns formatted content with metadata and use cases

INSTRUCTION SETS:
1. **agent_system**: Core system prompt and omni.ui framework basics
   - Understanding omni.ui and omni.ui.scene fundamentals
   - Widget filters, options menus, searchable comboboxes
   - General code writing guidelines and placeholder patterns

2. **classes**: Comprehensive class API reference and model patterns
   - AbstractValueModel and data model implementations
   - SimpleStringModel, SimpleBoolModel, SimpleFloatModel, SimpleIntModel
   - Custom models, callbacks, and model-view patterns

3. **omni_ui_scene_system**: Complete 3D UI system documentation
   - 3D shapes (Line, Curve, Rectangle, Arc, etc.)
   - SceneView, camera controls, Transform containers
   - Gestures, mouse interactions, manipulators
   - USD camera and stage synchronization

4. **omni_ui_system**: Core widgets, containers, layouts and styling
   - Basic UI shapes and widgets (Labels, Buttons, Fields, Sliders)
   - Layout systems (HStack, VStack, ZStack, Grid)
   - Window management, styling with selectors
   - Drag & drop, MDV pattern, callbacks

RETURNS:
When name is provided:
- Formatted instruction content with metadata header
- Description and use cases for the instruction set
- Full documentation content

When name is not provided:
- List of all available instructions
- Descriptions and use cases for each

USAGE EXAMPLES:
get_ui_instructions(name="agent_system")  # Get core system prompt
get_ui_instructions(name="omni_ui_system")  # Get widgets and layouts documentation
get_ui_instructions()  # List all available instructions

WHEN TO USE:
- Load agent_system when starting omni.ui development for fundamental concepts
- Load classes when working with data models and custom implementations
- Load omni_ui_scene_system for 3D UI and manipulator development
- Load omni_ui_system for standard UI widgets and layouts
- Call without parameters to see all available instructions"""


class GetUIInstructionsConfig(FunctionBaseConfig, name="get_ui_instructions"):
    """Configuration for get_ui_instructions function."""

    name: str = "get_ui_instructions"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetUIInstructionsConfig, framework_wrappers=[])
async def register_get_ui_instructions(config: GetUIInstructionsConfig, builder: Builder):
    """Register get_ui_instructions function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_instructions in verbose mode")

    async def get_ui_instructions_wrapper(input: GetUIInstructionsInput) -> str:
        """Get OmniUI system instructions."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {"name": input.name} if input.name else {}
        error_msg = None
        success = True

        try:
            # If no name provided, list all instructions
            if input.name is None:
                result = await list_instructions()
            else:
                # Get specific instruction
                result = await get_instructions(input.name)

            # Use config fields to modify behavior
            if verbose:
                if input.name:
                    logger.debug(f"Retrieved instruction: {input.name}")
                else:
                    logger.debug("Listed all available instructions")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve instructions - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_ui_instructions",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_instructions: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_ui_instructions_wrapper,
        description=GET_UI_INSTRUCTIONS_DESCRIPTION,
        input_schema=GetUIInstructionsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
