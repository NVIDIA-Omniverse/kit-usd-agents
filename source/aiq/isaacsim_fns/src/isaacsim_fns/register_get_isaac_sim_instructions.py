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

"""Registration wrapper for get_instructions function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_instructions import get_instructions, list_instructions
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_instruction_sets_input(instruction_sets_str: str) -> Union[List[str], None]:
    """Parse instruction_sets string input into appropriate format for get_instructions.

    Args:
        instruction_sets_str: String input that can be:
            - None/empty: Return None (list all instruction sets)
            - Single instruction: "isaacsim_system"
            - JSON array: '["isaacsim_system", "robot_setup", "sensors"]'
            - Comma-separated: "isaacsim_system, robot_setup, sensors"

    Returns:
        - None: For empty input (list all instruction sets)
        - List[str]: For instruction set names

    Raises:
        ValueError: If input format is invalid
    """
    if not instruction_sets_str or not instruction_sets_str.strip():
        return None

    instruction_sets_str = instruction_sets_str.strip()

    # Try to parse as JSON array first
    if instruction_sets_str.startswith("[") and instruction_sets_str.endswith("]"):
        try:
            parsed = json.loads(instruction_sets_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in instruction_sets_str:
        return [name.strip() for name in instruction_sets_str.split(",") if name.strip()]

    # Single instruction set name
    return [instruction_sets_str]


class GetIsaacSimInstructionsInput(BaseModel):
    """Input for get_instructions function.

    Provide instruction sets in any convenient format - the system will handle the conversion automatically.
    """

    instruction_sets: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Instruction sets to retrieve. Accepts multiple flexible formats:

        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single instruction: "isaacsim_system"
        - Native array: ["isaacsim_system", "robot_setup", "sensors"] ← WORKS DIRECTLY!
        - JSON string: '["isaacsim_system", "physics", "ros_2"]'
        - Comma-separated: "isaacsim_system, robot_setup, sensors"
        - Empty/null: Lists all available instruction sets

        📚 AVAILABLE INSTRUCTION SETS:
        - "isaacsim_system": Introduction to Isaac Sim, design, simulation, sensors, digital twins
        - "what_is_isaac_sim": Introduction to Isaac Sim as an Omniverse robotics simulation app
        - "quick_install": Short install steps for Linux and Windows
        - "installation": Full installation options (workstation, container, cloud, Python)
        - "workflows": GUI, Extensions, and Standalone Python workflows
        - "quick_tutorials": Introductory and Robot Setup tutorials
        - "examples": Interactive and standalone examples
        - "python_scripting_and_tutorials": Python scripting (standalone/interactive), core API, tutorials
        - "robot_setup": Robot Wizard, editors, assembler, and robot building tutorials
        - "robot_simulation": Wheeled robots, manipulators, policy control, motion generation
        - "sensors": Cameras, depth, RTX sensors, physics-based sensors, calibration
        - "synthetic_data_generation": Replicator, perception/action data, grasping, MobilityGen
        - "physics": PhysX, Newton, USD schemas, simulation flow
        - "ros_2": ROS 2 bridge, installation, workspaces, tutorials
        - "omnigraph": Visual programming for Replicators, ROS 2, sensors, controllers
        - "omniverse_and_usd": USD basics, robot schema, USD tools, Omniverse commands
        - "isaac_lab": Robot learning framework with RL/imitation learning
        - "digital_twin": Warehouse logistics, Cortex robotics, mapping
        - "importers_and_exporters": URDF, MJCF, CAD, ShapeNet import/export
        - "asset_structure": Asset organization (base, parts, materials)
        - "isaac_sim_assets": Available robots, sensors, props, environments
        - "browsers": Content, Asset, Material, NVIDIA Asset, SimReady browsers
        - "gui_reference": GUI overview, shortcuts, Create/Replicator menus, preferences
        - "user_interface_reference": Menu bar, viewport, toolbar, stage, property panel
        - "keyboard_shortcuts_reference": Viewport, gizmo, and common action shortcuts
        - "development_tools": VS Code, Jupyter, Script Editor, Carb settings
        - "debugging_profiling": Debug Drawing, VS Code debugging, Tracy profiling
        - "application_template": Build custom apps from extension registry
        - "templates": Extension Template Generator, VS Code templates
        - "adding_and_updating_extensions_guide": Add/update extensions via registry
        - "api_documentation": Isaac Sim and Omniverse API reference links
        - "isaac_sim_conventions": Units, rotations, coordinate conventions
        - "isaac_sim_performance_optimization_handbook": Physics/rendering/sensor tuning
        - "isaac_sim_benchmarks": Performance KPIs and measurement
        - "reference_architecture_and_task_groupings": Architecture and use case groupings
        - "release_notes": Isaac Sim 6.0.0 release notes
        - "renaming_extensions_in_isaac_sim_4_5": Deprecated-to-new extension name mapping
        - "glossary": Omniverse and Isaac Sim term definitions
        - "help_faq": FAQ, troubleshooting, developer resources
        - "licenses": Licensing terms
        - "data_collection_usage": Telemetry and data collection settings
        - "community_project_highlights": Community projects and tools
        - "omniverse_feedback_and_forums": Forums, Discord, feedback

        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_ISAAC_SIM_INSTRUCTIONS_DESCRIPTION = """Retrieve Isaac Sim system instructions and documentation for development.

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- instruction_sets: Instruction sets in ANY convenient format:
  * Single instruction: "isaacsim_system"
  * Native array: ["isaacsim_system", "robot_setup", "sensors"] ← WORKS DIRECTLY!
  * JSON string: '["isaacsim_system", "physics", "ros_2"]'
  * Comma-separated: "isaacsim_system, robot_setup, sensors"
  * Empty/null: Lists all available instruction sets

AVAILABLE INSTRUCTION SETS:
- **isaacsim_system**: Introduction to Isaac Sim, design, simulation, sensors, and digital twins
- **what_is_isaac_sim**: Introduction to Isaac Sim as an Omniverse robotics simulation app
- **quick_install**: Short install steps for Linux and Windows
- **installation**: Full installation options (workstation, container, cloud, Python)
- **workflows**: GUI, Extensions, and Standalone Python workflows
- **quick_tutorials**: Introductory and Robot Setup tutorials
- **examples**: Interactive and standalone examples
- **python_scripting_and_tutorials**: Python scripting (standalone/interactive), core API, tutorials
- **robot_setup**: Robot Wizard, editors, assembler, and robot building tutorials
- **robot_simulation**: Wheeled robots, manipulators, policy control, motion generation
- **sensors**: Cameras, depth, RTX sensors, physics-based sensors, calibration
- **synthetic_data_generation**: Replicator, perception/action data, grasping, MobilityGen
- **physics**: PhysX, Newton, USD schemas, simulation flow
- **ros_2**: ROS 2 bridge, installation, workspaces, tutorials
- **omnigraph**: Visual programming for Replicators, ROS 2, sensors, controllers
- **omniverse_and_usd**: USD basics, robot schema, USD tools, Omniverse commands
- **isaac_lab**: Robot learning framework with RL/imitation learning
- **digital_twin**: Warehouse logistics, Cortex robotics, mapping
- **importers_and_exporters**: URDF, MJCF, CAD, ShapeNet import/export
- **asset_structure**: Asset organization (base, parts, materials)
- **isaac_sim_assets**: Available robots, sensors, props, environments
- **browsers**: Content, Asset, Material, NVIDIA Asset, SimReady browsers
- **gui_reference**: GUI overview, shortcuts, Create/Replicator menus, preferences
- **user_interface_reference**: Menu bar, viewport, toolbar, stage, property panel
- **keyboard_shortcuts_reference**: Viewport, gizmo, and common action shortcuts
- **development_tools**: VS Code, Jupyter, Script Editor, Carb settings
- **debugging_profiling**: Debug Drawing, VS Code debugging, Tracy profiling
- **application_template**: Build custom apps from extension registry
- **templates**: Extension Template Generator, VS Code templates
- **adding_and_updating_extensions_guide**: Add/update extensions via registry
- **api_documentation**: Isaac Sim and Omniverse API reference links
- **isaac_sim_conventions**: Units, rotations, coordinate conventions
- **isaac_sim_performance_optimization_handbook**: Physics/rendering/sensor tuning
- **isaac_sim_benchmarks**: Performance KPIs and measurement
- **reference_architecture_and_task_groupings**: Architecture and use case groupings
- **release_notes**: Isaac Sim 6.0.0 release notes
- **renaming_extensions_in_isaac_sim_4_5**: Deprecated-to-new extension name mapping
- **glossary**: Omniverse and Isaac Sim term definitions
- **help_faq**: FAQ, troubleshooting, developer resources
- **licenses**: Licensing terms
- **data_collection_usage**: Telemetry and data collection settings
- **community_project_highlights**: Community projects and tools
- **omniverse_feedback_and_forums**: Forums, Discord, feedback

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_instructions(instruction_sets=["isaacsim_system", "robot_setup"])
✅ Single string: get_instructions(instruction_sets="isaacsim_system")
✅ JSON string: get_instructions(instruction_sets='["isaacsim_system", "physics", "sensors"]')
✅ Comma format: get_instructions(instruction_sets="isaacsim_system, robot_setup, sensors")
✅ List all: get_instructions() or get_instructions(instruction_sets=null)

💡 FOR AI MODELS: You can pass arrays directly like ["isaacsim_system", "robot_setup"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- Single API call for multiple instruction sets
- Combined documentation with clear sections
- Efficient context window usage
- Maximum compatibility with all AI models

RETURNS:
- For single instruction set: Formatted documentation with use cases
- For multiple sets: Combined documentation with section headers
- For listing: All available instruction sets with descriptions

WHEN TO USE:
- Load "isaacsim_system" when starting Isaac Sim development for framework fundamentals
- Load "robot_setup" for robot building with Wizard, editors, and assembler
- Load "robot_simulation" for robot control, motion, and policy-based examples
- Load "sensors" for camera and sensor configuration
- Load "synthetic_data_generation" for Replicator and data generation
- Load "physics" for PhysX simulation setup
- Load "ros_2" for ROS 2 integration and bridging
- Load "omniverse_and_usd" for USD and Omniverse fundamentals
- Call without parameters to see all available instructions"""


class GetIsaacSimInstructionsConfig(FunctionBaseConfig, name="get_isaac_sim_instructions"):
    """Configuration for get_instructions function."""

    name: str = "get_isaac_sim_instructions"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetIsaacSimInstructionsConfig, framework_wrappers=[])
async def register_get_isaac_sim_instructions(config: GetIsaacSimInstructionsConfig, builder: Builder):
    """Register get_instructions function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_isaac_sim_instructions in verbose mode")

    async def get_isaac_sim_instructions_wrapper(input: GetIsaacSimInstructionsInput) -> str:
        """Get Isaac Sim system instructions."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.instruction_sets is None:
                instruction_sets_to_fetch = None
            elif isinstance(input.instruction_sets, list):
                # Direct array input - validate and use as-is
                if len(input.instruction_sets) == 0:
                    instruction_sets_to_fetch = None  # Empty array = list all
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.instruction_sets):
                        if not isinstance(item, str):
                            return f"ERROR: All items in instruction_sets array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in instruction_sets array"
                    instruction_sets_to_fetch = [item.strip() for item in input.instruction_sets]
            elif isinstance(input.instruction_sets, str):
                # String input - parse using existing logic
                instruction_sets_to_fetch = _parse_instruction_sets_input(input.instruction_sets)
            else:
                return f"ERROR: instruction_sets must be a string, array, or null, got {type(input.instruction_sets).__name__}"

            parameters = {"instruction_sets": input.instruction_sets}
        except ValueError as e:
            return f"ERROR: Invalid instruction_sets parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_instructions(instruction_sets_to_fetch)

            # Use config fields to modify behavior
            if verbose:
                if instruction_sets_to_fetch is None:
                    logger.debug("Listed all available instructions")
                elif isinstance(instruction_sets_to_fetch, list):
                    logger.debug(f"Retrieved {len(instruction_sets_to_fetch)} instruction sets")
                else:
                    logger.debug(f"Retrieved instruction: {instruction_sets_to_fetch}")

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
                        tool_name="get_isaac_sim_instructions",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_instructions: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_isaac_sim_instructions_wrapper,
        description=GET_ISAAC_SIM_INSTRUCTIONS_DESCRIPTION,
        input_schema=GetIsaacSimInstructionsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
