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

"""Function to retrieve Isaac Sim system instructions."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from ..config import ISAACSIM_VERSION
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Define the instruction files and their metadata
INSTRUCTION_FILES = {
    "isaacsim_system": {
        "filename": "What_Is_Isaac_Sim.md",
        "description": "Introduction to Isaac Sim: design, simulation, sensors, digital twins, and core capabilities",
        "use_cases": [
            "Understand Isaac Sim architecture and capabilities",
            "Learn about simulation, sensors, and digital twins",
            "Explore available workflows and tools",
            "Get started with Isaac Sim development",
        ],
    },
    "adding_and_updating_extensions_guide": {
        "filename": "Adding_and_Updating_Extensions_Guide.md",
        "description": "How to add or update Omniverse extensions via the Extensions menu, search paths, and registries",
        "use_cases": [
            "Add extensions from the registry",
            "Update existing extensions",
            "Configure extension search paths",
            "Manage third-party extensions",
        ],
    },
    "api_documentation": {
        "filename": "API_Documentation.md",
        "description": "Links to Isaac Sim and Omniverse API reference documentation",
        "use_cases": [
            "Look up Isaac Sim Python API",
            "Find Omniverse API references",
            "Navigate SDK documentation",
        ],
    },
    "application_template": {
        "filename": "Application_Template.md",
        "description": "Using the Isaac Sim application template repo to build and customize lightweight apps",
        "use_cases": [
            "Create custom Isaac Sim apps",
            "Build lightweight apps from extension registry",
            "Customize application templates",
            "Package and distribute apps",
        ],
    },
    "asset_structure": {
        "filename": "Asset_Structure.md",
        "description": "How Isaac Sim assets are organized (base, parts, materials) and transformed for simulation",
        "use_cases": [
            "Understand asset hierarchy and organization",
            "Structure assets for simulation",
            "Manage materials and parts",
            "Normalize scale and units",
        ],
    },
    "browsers": {
        "filename": "Browsers.md",
        "description": "Isaac Sim browsers: Content, Asset, Material, NVIDIA Asset, and SimReady Explorer",
        "use_cases": [
            "Browse and organize assets",
            "Use the Content Browser",
            "Find materials and SimReady assets",
            "Navigate NVIDIA Asset browser",
        ],
    },
    "community_project_highlights": {
        "filename": "Community_Project_Highlights.md",
        "description": "Community projects shared with the Isaac Sim community (MCP extension, typings, tutorials)",
        "use_cases": [
            "Discover community-built tools",
            "Find third-party tutorials and examples",
            "Explore MCP extension projects",
        ],
    },
    "data_collection_usage": {
        "filename": "Data_Collection_Usage.md",
        "description": "What Omniverse collects (installation, hardware, usage, errors) and data collection settings",
        "use_cases": [
            "Understand data collection policies",
            "Change data collection settings",
            "Review collected telemetry types",
        ],
    },
    "debugging_profiling": {
        "filename": "Debugging_Profiling.md",
        "description": "Debugging tools (Debug Drawing, Omniverse Commands, VS Code) and profiling with Tracy",
        "use_cases": [
            "Debug extensions with VS Code",
            "Use debug drawing utilities",
            "Profile performance with Tracy",
            "Inspect Omniverse commands",
        ],
    },
    "development_tools": {
        "filename": "Development_Tools.md",
        "description": "Tools for creating extensions: VS Code, Jupyter, Omniverse Script Editor, and Carb settings",
        "use_cases": [
            "Set up VS Code for development",
            "Use Jupyter notebooks with Isaac Sim",
            "Run scripts in Omniverse Script Editor",
            "Configure Carbonite settings",
        ],
    },
    "digital_twin": {
        "filename": "Digital_Twin.md",
        "description": "Digital twin features: warehouse logistics, Cortex robotics, mapping, and troubleshooting",
        "use_cases": [
            "Build warehouse digital twins",
            "Use Cortex for robotics scenarios",
            "Create digital twin mappings",
            "Troubleshoot digital twin setups",
        ],
    },
    "examples": {
        "filename": "Examples.md",
        "description": "Interactive and standalone examples in Isaac Sim, including how to run them",
        "use_cases": [
            "Run built-in interactive examples",
            "Execute standalone example scripts",
            "Learn from reference implementations",
            "Explore feature demonstrations",
        ],
    },
    "glossary": {
        "filename": "Glossary.md",
        "description": "Definitions of Omniverse and Isaac Sim terms (Apps, Connectors, Nucleus, etc.)",
        "use_cases": [
            "Look up Omniverse terminology",
            "Understand Isaac Sim concepts",
            "Clarify technical terms",
        ],
    },
    "gui_reference": {
        "filename": "GUI_Reference.md",
        "description": "Isaac Sim GUI overview: UI reference, shortcuts, Create/Replicator menus, and preferences",
        "use_cases": [
            "Navigate the Isaac Sim GUI",
            "Configure UI preferences",
            "Use Create and Replicator menus",
            "Customize panel layouts",
        ],
    },
    "help_faq": {
        "filename": "Help_FAQ.md",
        "description": "Developer resources, Discord, forums, feedback, FAQ, and troubleshooting",
        "use_cases": [
            "Find answers to common questions",
            "Access developer support channels",
            "Troubleshoot common issues",
            "Submit feedback and bug reports",
        ],
    },
    "importers_and_exporters": {
        "filename": "Importers_and_Exporters.md",
        "description": "Import/export for URDF, MJCF, CAD, ShapeNet, and related tutorials",
        "use_cases": [
            "Import robots via URDF or MJCF",
            "Export scenes and assets",
            "Convert CAD models for simulation",
            "Import ShapeNet objects",
        ],
    },
    "installation": {
        "filename": "Installation.md",
        "description": "Isaac Sim installation options: workstation, container, cloud, livestream, and Python environment",
        "use_cases": [
            "Install Isaac Sim on workstation",
            "Deploy via container or cloud",
            "Set up livestream mode",
            "Configure Python environment",
        ],
    },
    "isaac_lab": {
        "filename": "Isaac_Lab.md",
        "description": "Isaac Lab robot learning framework for Isaac Sim with RL/imitation learning APIs and examples",
        "use_cases": [
            "Set up Isaac Lab for robot learning",
            "Train with reinforcement learning",
            "Run imitation learning workflows",
            "Use Isaac Lab example tasks",
        ],
    },
    "isaac_sim_assets": {
        "filename": "Isaac_Sim_Assets.md",
        "description": "Available assets (robots, sensors, props, environments) and Content Browser usage",
        "use_cases": [
            "Browse available robot models",
            "Find sensor and prop assets",
            "Use environment presets",
            "Navigate the Content Browser",
        ],
    },
    "isaac_sim_benchmarks": {
        "filename": "Isaac_Sim_Benchmarks.md",
        "description": "Performance KPIs for Isaac Sim and how to measure them on your hardware",
        "use_cases": [
            "Benchmark Isaac Sim performance",
            "Measure simulation KPIs",
            "Compare hardware configurations",
            "Identify performance bottlenecks",
        ],
    },
    "isaac_sim_conventions": {
        "filename": "Isaac_Sim_Conventions.md",
        "description": "Units, rotation representations, and coordinate conventions used in Isaac Sim",
        "use_cases": [
            "Understand unit conventions",
            "Work with rotation representations",
            "Convert between coordinate systems",
            "Ensure consistent asset scales",
        ],
    },
    "isaac_sim_performance_optimization_handbook": {
        "filename": "Isaac_Sim_Performance_Optimization_Handbook.md",
        "description": "Performance tuning for physics, rendering, and sensors in Isaac Sim",
        "use_cases": [
            "Optimize physics simulation speed",
            "Tune rendering performance",
            "Improve sensor throughput",
            "Reduce memory and GPU usage",
        ],
    },
    "keyboard_shortcuts_reference": {
        "filename": "Keyboard_Shortcuts_Reference.md",
        "description": "Keyboard shortcuts for viewport controls, gizmos (move/rotate/scale), and common actions",
        "use_cases": [
            "Learn viewport navigation shortcuts",
            "Use gizmo keyboard shortcuts",
            "Speed up common workflows",
        ],
    },
    "licenses": {
        "filename": "Licenses.md",
        "description": "Isaac Sim licensing (Apache 2.0, additional software, WebRTC, Omniverse) and related terms",
        "use_cases": [
            "Review Isaac Sim license terms",
            "Check third-party license requirements",
            "Understand usage restrictions",
        ],
    },
    "omnigraph": {
        "filename": "Omnigraph.md",
        "description": "OmniGraph visual programming framework for Replicators, ROS 2 bridge, sensors, and controllers",
        "use_cases": [
            "Build visual programming graphs",
            "Configure ROS 2 bridge via OmniGraph",
            "Create sensor pipelines",
            "Set up controller nodes",
        ],
    },
    "omniverse_and_usd": {
        "filename": "Omniverse_and_USD.md",
        "description": "USD basics, robot schema, USD tools, and Omniverse commands for Isaac Sim",
        "use_cases": [
            "Work with USD stages and prims",
            "Use robot USD schema",
            "Run Omniverse commands programmatically",
            "Manage USD layers and references",
        ],
    },
    "omniverse_feedback_and_forums": {
        "filename": "Omniverse_Feedback_and_Forums.md",
        "description": "How to contact Omniverse support: forums, Discord, and feedback forms",
        "use_cases": [
            "Find community support channels",
            "Submit feedback to NVIDIA",
            "Join the Omniverse Discord",
        ],
    },
    "physics": {
        "filename": "Physics.md",
        "description": "Omniverse Physics (PhysX, Newton), USD schemas, and simulation flow",
        "use_cases": [
            "Configure PhysX simulation",
            "Set up rigid body and articulation physics",
            "Understand physics USD schemas",
            "Tune simulation step parameters",
        ],
    },
    "python_scripting_and_tutorials": {
        "filename": "Python_Scripting_and_Tutorials.md",
        "description": "Python scripting concepts (standalone vs interactive), core API, snippets, and tutorials",
        "use_cases": [
            "Write standalone Python scripts",
            "Use interactive scripting mode",
            "Access core Python API",
            "Follow scripting tutorials",
        ],
    },
    "quick_install": {
        "filename": "Quick_Install.md",
        "description": "Short install steps for Linux and Windows to get Isaac Sim running quickly",
        "use_cases": [
            "Install Isaac Sim on Linux",
            "Install Isaac Sim on Windows",
            "Verify installation",
        ],
    },
    "quick_tutorials": {
        "filename": "Quick_Tutorials.md",
        "description": "Introductory tutorials for beginners and Robot Setup tutorials",
        "use_cases": [
            "Get started with Isaac Sim basics",
            "Follow beginner tutorials",
            "Complete robot setup walkthroughs",
        ],
    },
    "reference_architecture_and_task_groupings": {
        "filename": "Reference_Architecture_and_Task_Groupings.md",
        "description": "Reference architecture for Isaac Sim use cases: geometry authoring, assets, scene setup, digital twin",
        "use_cases": [
            "Understand Isaac Sim architecture",
            "Map tasks to reference workflows",
            "Plan scene setup pipelines",
            "Design end-to-end simulation workflows",
        ],
    },
    "release_notes": {
        "filename": "Release_Notes.md",
        "description": "Isaac Sim 6.0.0 early developer release notes: Kit SDK, Python, RT 2.0, dependencies",
        "use_cases": [
            "Review new features and changes",
            "Check breaking changes and deprecations",
            "Verify dependency updates",
        ],
    },
    "renaming_extensions_in_isaac_sim_4_5": {
        "filename": "Renaming_Extensions_in_Isaac_Sim_4_5.md",
        "description": "Mapping of deprecated extensions to new names for Isaac Sim 4.5 migration (alternate format)",
        "use_cases": [
            "Migrate extensions to 4.5 naming",
            "Look up renamed extension identifiers",
            "Update .kit files for compatibility",
        ],
    },
    "robot_setup": {
        "filename": "Robot_Setup.md",
        "description": "Robot setup tools (Robot Wizard, editors, assembler) and tutorials for building custom robots",
        "use_cases": [
            "Use Robot Wizard for setup",
            "Assemble custom robots",
            "Configure joints and drives",
            "Follow robot building tutorials",
        ],
    },
    "robot_simulation": {
        "filename": "Robot_Simulation.md",
        "description": "Robot control: wheeled, manipulators, policy-controlled robots, joint control, and motion generation",
        "use_cases": [
            "Control wheeled robots",
            "Operate manipulator arms",
            "Run policy-based control",
            "Generate motion trajectories",
        ],
    },
    "ros_2": {
        "filename": "ROS_2.md",
        "description": "ROS 2 integration with Isaac Sim: bridge, installation, workspaces, and tutorials",
        "use_cases": [
            "Set up the ROS 2 bridge",
            "Install ROS 2 for Isaac Sim",
            "Bridge sensor and control topics",
            "Follow ROS 2 tutorials",
        ],
    },
    "sensors": {
        "filename": "Sensors.md",
        "description": "Sensor simulation: cameras, depth, RTX sensors, physics-based sensors, and calibration",
        "use_cases": [
            "Configure camera and depth sensors",
            "Use RTX-based sensors",
            "Set up physics-based sensors",
            "Calibrate and validate sensors",
        ],
    },
    "synthetic_data_generation": {
        "filename": "Synthetic_Data_Generation.md",
        "description": "SDG tools and workflows: Replicator, perception data, action/event data, grasping, MobilityGen",
        "use_cases": [
            "Generate synthetic training data",
            "Configure Replicator pipelines",
            "Capture perception annotations",
            "Run grasping and MobilityGen workflows",
        ],
    },
    "templates": {
        "filename": "Templates.md",
        "description": "Project templates: custom examples, Extension Template Generator, and VS Code extension templates",
        "use_cases": [
            "Create projects from templates",
            "Generate extension boilerplate",
            "Use VS Code extension templates",
            "Build custom examples",
        ],
    },
    "user_interface_reference": {
        "filename": "User_Interface_Reference.md",
        "description": "Isaac Sim UI overview: menu bar, viewport, toolbar, browsers, stage, and property panel",
        "use_cases": [
            "Understand the Isaac Sim UI layout",
            "Navigate menus and toolbars",
            "Use the Stage and Property panels",
            "Customize viewport settings",
        ],
    },
    "what_is_isaac_sim": {
        "filename": "What_Is_Isaac_Sim.md",
        "description": "Introduction to Isaac Sim as an Omniverse app for AI-driven robotics simulation and design",
        "use_cases": [
            "Learn what Isaac Sim offers",
            "Understand core capabilities",
            "Explore robotics simulation features",
        ],
    },
    "workflows": {
        "filename": "Workflows.md",
        "description": "Three main workflows: GUI, Extensions, and Standalone Python, and when to use each",
        "use_cases": [
            "Choose between GUI and scripting workflows",
            "Develop with Extensions workflow",
            "Run standalone Python pipelines",
            "Understand workflow trade-offs",
        ],
    },
}


async def get_instructions(instruction_sets) -> Dict[str, Any]:
    """
    Retrieve specific Isaac Sim system instructions by name.

    Args:
        instruction_sets: The instruction sets to retrieve. Pass None or an empty
            list to list all available instruction sets. Valid keys correspond to
            entries in INSTRUCTION_FILES (e.g. 'robot_setup', 'sensors', 'ros_2',
            'physics', 'synthetic_data_generation', etc.).

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
                instructions_dir = Path(__file__).parent.parent / "data" / ISAACSIM_VERSION / "instructions"
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
                combined_content.append(f"# Isaac Sim Instruction: {result['name']}")
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
            result = f"""# Isaac Sim Instruction: {instruction_set}

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
    List all available Isaac Sim system instructions.

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
        result_lines = ["# Available Isaac Sim Instructions\n"]

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
