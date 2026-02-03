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

"""Function to retrieve OmniUI system instructions."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Define the instruction files and their metadata
INSTRUCTION_FILES = {
    "agent_system": {
        "filename": "agent_system.md",
        "description": "Core Omniverse UI Assistant system prompt with omni.ui framework basics",
        "use_cases": [
            "Understanding omni.ui framework fundamentals",
            "Learning about omni.ui.scene for 3D UI",
            "Understanding widget filters and options menus",
            "Working with searchable comboboxes and search fields",
            "General omni.ui code writing guidelines",
            "Understanding UI placeholder patterns for scene operations",
        ],
    },
    "classes": {
        "filename": "classes.md",
        "description": "Comprehensive omni.ui class API reference and model patterns",
        "use_cases": [
            "Working with AbstractValueModel and data models",
            "Understanding SimpleStringModel, SimpleBoolModel, SimpleFloatModel, SimpleIntModel",
            "Creating custom model implementations",
            "Model callbacks and value change handling",
            "Understanding model-view patterns in omni.ui",
        ],
    },
    "omni_ui_scene_system": {
        "filename": "omni_ui_scene_system.md",
        "description": "Complete omni.ui.scene 3D UI system documentation",
        "use_cases": [
            "Creating 3D shapes (Line, Curve, Rectangle, Arc, etc.)",
            "Working with SceneView and camera controls",
            "Understanding Transform containers and matrices",
            "Implementing gestures and mouse interactions in 3D",
            "Building manipulators and custom 3D controls",
            "Syncing with USD camera and stage",
            "Using standard transform manipulators",
        ],
    },
    "omni_ui_system": {
        "filename": "omni_ui_system.md",
        "description": "Core omni.ui widgets, containers, layouts and styling system",
        "use_cases": [
            "Understanding basic UI shapes and widgets",
            "Working with Labels, Buttons, Fields, Sliders",
            "Creating layouts with HStack, VStack, ZStack, Grid",
            "Understanding Window creation and management",
            "Styling with selectors and style sheets",
            "Working with shades and color palettes",
            "Implementing drag & drop functionality",
            "Understanding Model-Delegate-View (MDV) pattern",
            "Managing callbacks and subscriptions",
        ],
    },
}


async def get_instructions(name: str) -> Dict[str, Any]:
    """
    Retrieve specific OmniUI system instructions by name.

    Args:
        name: The name of the instruction set to retrieve. Valid values are:
            - 'agent_system': Core system prompt and framework basics
            - 'classes': Class API reference and model patterns
            - 'omni_ui_scene_system': 3D UI system documentation
            - 'omni_ui_system': Core widgets, layouts and styling

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
    telemetry_data = {"name": name}

    success = True
    error_msg = None

    try:
        # Validate instruction name
        if name not in INSTRUCTION_FILES:
            available = list(INSTRUCTION_FILES.keys())
            return {
                "success": False,
                "error": f"Unknown instruction name '{name}'. Available instructions: {', '.join(available)}",
                "result": None,
            }

        instruction_info = INSTRUCTION_FILES[name]

        # Get the instructions directory relative to this file
        # This file is at: src/omni_aiq_omni_ui/functions/get_instructions.py
        # Instructions are at: src/omni_aiq_omni_ui/data/instructions/
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
            "name": name,
            "description": instruction_info["description"],
            "use_cases": instruction_info["use_cases"],
            "filename": instruction_info["filename"],
            "content_length": len(content),
            "line_count": content.count("\n") + 1,
        }

        # Format the result with metadata header
        result = f"""# OmniUI Instruction: {name}

## Description
{instruction_info['description']}

## Use Cases
This instruction set is useful for:
{chr(10).join(f"- {use_case}" for use_case in instruction_info['use_cases'])}

---

{content}"""

        logger.info(f"Successfully retrieved instruction '{name}' ({metadata['line_count']} lines)")

        return {"success": True, "result": result, "metadata": metadata}

    except Exception as e:
        logger.error(f"Unexpected error retrieving instruction '{name}': {e}")
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
    List all available OmniUI system instructions.

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
        result_lines = ["# Available OmniUI Instructions\n"]

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
