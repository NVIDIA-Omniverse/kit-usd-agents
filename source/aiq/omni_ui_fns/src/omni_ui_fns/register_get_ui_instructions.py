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
GET_UI_INSTRUCTIONS_DESCRIPTION = """Top-level OmniUI router / preamble. Loads the canonical omni.ui playbook (framework basics, data models, 3D scene UI, widgets / layouts / styling) before a coding task.

WHEN TO USE THIS TOOL:
- You are about to start an omni.ui coding task and want the playbook in context.
- You need the core "how omni.ui is structured" primer.
- Call with no argument to see all available instruction sets.

AVAILABLE INSTRUCTION SETS:
- "agent_system": core system prompt + omni.ui / omni.ui.scene fundamentals, writing conventions.
- "classes": class API reference & model patterns (AbstractValueModel, SimpleStringModel, custom models, MVC).
- "omni_ui_scene_system": complete omni.ui.scene 3D UI docs (shapes, SceneView, gestures, manipulators, USD camera sync).
- "omni_ui_system": widgets, containers, layouts, window management, styling, drag & drop.

ARGUMENTS:
- name (str, optional): one of the set names above; null lists all sets with descriptions.

RETURNS:
Formatted instruction content with metadata and use cases, or a directory listing when called with no argument.

USAGE EXAMPLES:
get_ui_instructions name="agent_system"
get_ui_instructions name="omni_ui_scene_system"
get_ui_instructions

CROSS-SERVER ROUTING:
- For Universal Scene Description (USD) concepts and pxr API details → use the USD Code MCP's `search_usd_knowledge`.
- For Isaac Sim robotics, sensor extensions, and synthetic data generation → use the Isaac Sim MCP's `search_isaac_sim_code_examples`.
- For general Kit runtime, lifecycle, and architecture → use the Kit MCP's `search_kit_knowledge`.

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Per-class usage guidance → use get_ui_class_instructions.
- Widget / window code examples → use search_ui_code_examples or search_ui_window_examples.
- Styling specifics → use get_ui_style_docs.
- Raw class / method signatures → use get_ui_class_detail / get_ui_method_detail."""


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
