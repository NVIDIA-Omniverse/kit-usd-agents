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

"""Function to retrieve Kit system instructions."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from ..config import KIT_VERSION
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Define the instruction files and their metadata
INSTRUCTION_FILES = {
    "kit_system": {
        "filename": "kit_system.md",
        "description": "Core Kit framework fundamentals and architecture",
        "use_cases": [
            "Understanding Kit extension system",
            "Learning about USD integration patterns",
            "Extension lifecycle management",
            "Kit application architecture",
            "Carbonite services integration",
            "General Kit development guidelines",
        ],
    },
    "extensions": {
        "filename": "extensions.md",
        "description": "Extension development guidelines and patterns",
        "use_cases": [
            "Creating new Kit extensions",
            "Extension configuration and dependencies",
            "Extension lifecycle best practices",
            "Service registration patterns",
            "UI extension development",
            "Extension testing strategies",
        ],
    },
    "testing": {
        "filename": "testing.md",
        "description": "Test writing best practices and framework usage",
        "use_cases": [
            "Writing unit tests for Kit code",
            "UI testing with omni.kit.test",
            "USD stage testing patterns",
            "Extension lifecycle testing",
            "Performance testing strategies",
            "Mock and test double usage",
        ],
    },
    "usd": {
        "filename": "usd.md",
        "description": "USD integration and scene description patterns",
        "use_cases": [
            "Stage management and context handling",
            "Prim creation and manipulation",
            "Layer composition and editing",
            "Transform and animation workflows",
            "Material and shading setup",
            "USD performance optimization",
        ],
    },
    "ui": {
        "filename": "ui.md",
        "description": "UI development with Kit widgets and layouts",
        "use_cases": [
            "Window and layout creation",
            "Widget usage and styling",
            "Model-view data binding",
            "3D UI with omni.ui.scene",
            "Event handling and callbacks",
            "Custom widget development",
        ],
    },
}


async def get_instructions(instruction_sets) -> Dict[str, Any]:
    """
    Retrieve specific Kit system instructions by name.

    Args:
        instruction_sets: The instruction sets to retrieve. Valid values are:
            - 'kit_system': Core framework and architecture
            - 'extensions': Extension development patterns
            - 'testing': Test writing best practices
            - 'usd': USD integration patterns
            - 'ui': UI development with widgets and layouts
            - None: Lists all available instruction sets

    Returns:
        Dictionary with:
        - success: Whether retrieval was successful
        - result: The instruction content if successful
        - error: Error message if failed
        - metadata: Additional information about the instruction
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {"instruction_sets": instruction_sets}

    success = True
    error_msg = None

    try:
        # Handle None input (list all available instructions)
        if instruction_sets is None:
            return await list_instructions()

        # Handle list input
        elif isinstance(instruction_sets, list):
            if len(instruction_sets) == 0:
                return await list_instructions()

            # Process multiple instruction sets
            results = []
            for instruction_set in instruction_sets:
                if instruction_set not in INSTRUCTION_FILES:
                    available = list(INSTRUCTION_FILES.keys())
                    return {
                        "success": False,
                        "error": f"Unknown instruction set '{instruction_set}'. Available: {', '.join(available)}",
                        "result": None,
                    }

                # Read instruction file
                instruction_info = INSTRUCTION_FILES[instruction_set]
                instructions_dir = Path(__file__).parent.parent / "data" / KIT_VERSION / "instructions"
                file_path = instructions_dir / instruction_info["filename"]

                if not file_path.exists():
                    return {
                        "success": False,
                        "error": f"Instruction file not found: {instruction_info['filename']}",
                        "result": None,
                    }

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                results.append(
                    {
                        "name": instruction_set,
                        "content": content,
                        "metadata": {
                            "description": instruction_info["description"],
                            "use_cases": instruction_info["use_cases"],
                            "filename": instruction_info["filename"],
                            "content_length": len(content),
                            "line_count": content.count("\n") + 1,
                        },
                    }
                )

            # Combine all instruction sets
            combined_content = []
            for result in results:
                combined_content.append(f"# Kit Instruction: {result['name']}")
                combined_content.append(f"\n## Description")
                combined_content.append(f"{result['metadata']['description']}")
                combined_content.append(f"\n---\n")
                combined_content.append(result["content"])
                combined_content.append(f"\n{'='*80}\n")

            return {
                "success": True,
                "result": "\n".join(combined_content),
                "metadata": {
                    "instruction_sets": [r["name"] for r in results],
                    "total_sets": len(results),
                    "combined_length": sum(r["metadata"]["content_length"] for r in results),
                },
            }

        # Handle string input (single instruction set)
        elif isinstance(instruction_sets, str):
            instruction_set = instruction_sets

            # Validate instruction name
            if instruction_set not in INSTRUCTION_FILES:
                available = list(INSTRUCTION_FILES.keys())
                return {
                    "success": False,
                    "error": f"Unknown instruction set '{instruction_set}'. Available: {', '.join(available)}",
                    "result": None,
                }

            instruction_info = INSTRUCTION_FILES[instruction_set]

            # Get the instructions directory relative to this file
            instructions_dir = Path(__file__).parent.parent / "data" / "instructions"
            file_path = instructions_dir / instruction_info["filename"]

            # Check if file exists
            if not file_path.exists():
                logger.error(f"Instruction file not found: {file_path}")
                return {
                    "success": False,
                    "error": f"Instruction file not found: {instruction_info['filename']}",
                    "result": None,
                }

            # Read the instruction content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read instruction file {file_path}: {e}")
                return {"success": False, "error": f"Failed to read instruction file: {str(e)}", "result": None}

            # Prepare metadata
            metadata = {
                "name": instruction_set,
                "description": instruction_info["description"],
                "use_cases": instruction_info["use_cases"],
                "filename": instruction_info["filename"],
                "content_length": len(content),
                "line_count": content.count("\n") + 1,
            }

            # Format the result with metadata header
            result = f"""# Kit Instruction: {instruction_set}

## Description
{instruction_info['description']}

## Use Cases
This instruction set is useful for:
{chr(10).join(f"- {use_case}" for use_case in instruction_info['use_cases'])}

---

{content}"""

            logger.info(f"Successfully retrieved instruction '{instruction_set}' ({metadata['line_count']} lines)")

            return {"success": True, "result": result, "metadata": metadata}

        else:
            return {
                "success": False,
                "error": f"instruction_sets must be string, array, or null, got {type(instruction_sets).__name__}",
                "result": None,
            }

    except Exception as e:
        logger.error(f"Unexpected error retrieving instruction '{instruction_sets}': {e}")
        error_msg = f"Unexpected error: {str(e)}"
        success = False
        return {"success": False, "error": error_msg, "result": None}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_instructions",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )


async def list_instructions() -> Dict[str, Any]:
    """
    List all available Kit system instructions.

    Returns:
        Dictionary with:
        - success: Whether listing was successful
        - result: Formatted list of available instructions
        - instructions: Detailed information about each instruction
    """
    try:
        instructions_list = []

        for name, info in INSTRUCTION_FILES.items():
            instructions_list.append({"name": name, "description": info["description"], "use_cases": info["use_cases"]})

        # Format result as readable text
        result_lines = ["# Available Kit Instructions\n"]

        for inst in instructions_list:
            result_lines.append(f"\n## {inst['name']}")
            result_lines.append(f"{inst['description']}")
            result_lines.append("\n**Use cases:**")
            for use_case in inst["use_cases"]:
                result_lines.append(f"  - {use_case}")

        result_lines.append(f"\n\nTotal instructions available: {len(instructions_list)}")

        return {"success": True, "result": "\n".join(result_lines), "instructions": instructions_list}

    except Exception as e:
        logger.error(f"Failed to list instructions: {e}")
        return {"success": False, "error": f"Failed to list instructions: {str(e)}", "result": None}
