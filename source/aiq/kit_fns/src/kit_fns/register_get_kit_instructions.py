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
            - Single instruction: "kit_system"
            - JSON array: '["kit_system", "extensions", "testing"]'
            - Comma-separated: "kit_system, extensions, testing"

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


class GetKitInstructionsInput(BaseModel):
    """Input for get_instructions function.

    Provide instruction sets in any convenient format - the system will handle the conversion automatically.
    """

    instruction_sets: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Instruction sets to retrieve. Accepts multiple flexible formats:
        
        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single instruction: "kit_system"
        - Native array: ["kit_system", "extensions", "testing"] ← WORKS DIRECTLY!
        - JSON string: '["kit_system", "extensions", "testing"]'
        - Comma-separated: "kit_system, extensions, testing"
        - Empty/null: Lists all available instruction sets
        
        📚 AVAILABLE INSTRUCTION SETS:
        - "kit_system": Core Kit framework fundamentals and architecture
        - "extensions": Extension development guidelines and patterns  
        - "testing": Test writing best practices and framework usage
        - "usd": USD integration and scene description patterns
        - "ui": UI development with Kit widgets and layouts
        
        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_KIT_INSTRUCTIONS_DESCRIPTION = """Router / preamble tool that loads Kit development guidance (framework fundamentals, extension patterns, testing, USD, UI) before a coding task.

WHEN TO USE THIS TOOL:
- You are about to start a Kit coding task and want the relevant playbook loaded into context.
- You need the canonical best-practices for extensions / tests / UI / USD within Kit.
- Call with no argument to see all available instruction sets.

AVAILABLE INSTRUCTION SETS:
- "kit_system": core Kit framework fundamentals and architecture.
- "extensions": extension development guidelines (config, lifecycle, services, testing).
- "testing": test-writing best practices and framework usage.
- "usd": USD integration patterns within Kit.
- "ui": omni.ui widget / layout / styling basics within Kit.

ARGUMENTS:
- instruction_sets (str | list[str] | null): one or more set names; accepts single string, list, JSON-array string, or comma-separated string. Null lists all sets.

RETURNS:
Formatted documentation for the requested set(s) with use cases, or a directory listing when called with no argument.

USAGE EXAMPLES:
get_kit_instructions "kit_system"
get_kit_instructions ["extensions", "testing"]
get_kit_instructions null

CROSS-SERVER ROUTING:
- For Universal Scene Description (USD) concepts and pxr API details → use the USD Code MCP's `search_usd_knowledge`.
- For Omniverse UI (omni.ui) widget styling and window layouts → use the OmniUI MCP's `search_ui_code_examples`.
- For Isaac Sim robotics, sensor extensions, and synthetic data generation → use the Isaac Sim MCP's `search_isaac_sim_code_examples`.

WHEN TO USE A DIFFERENT TOOL INSTEAD:
- Free-form Kit questions → use search_kit_knowledge.
- Concrete code examples → use search_kit_code_examples.
- Settings / API / extension lookup → use the corresponding specialized tool."""


class GetKitInstructionsConfig(FunctionBaseConfig, name="get_kit_instructions"):
    """Configuration for get_instructions function."""

    name: str = "get_kit_instructions"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetKitInstructionsConfig, framework_wrappers=[])
async def register_get_kit_instructions(config: GetKitInstructionsConfig, builder: Builder):
    """Register get_instructions function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_kit_instructions in verbose mode")

    async def get_kit_instructions_wrapper(input: GetKitInstructionsInput) -> str:
        """Get Kit system instructions."""
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
                        tool_name="get_kit_instructions",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_instructions: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_kit_instructions_wrapper,
        description=GET_KIT_INSTRUCTIONS_DESCRIPTION,
        input_schema=GetKitInstructionsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
