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

"""Function to retrieve specific OmniUI class instructions from categorized files."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)

# Define the available categories and their descriptions
CATEGORIES = {
    "models": {
        "description": "Data models and delegates for UI components",
        "classes": ["AbstractValueModel", "AbstractItemModel", "AbstractItemDelegate"],
    },
    "shapes": {
        "description": "Basic shapes and geometric primitives",
        "classes": [
            "Rectangle",
            "FreeRectangle",
            "Circle",
            "FreeCircle",
            "Ellipse",
            "FreeEllipse",
            "Triangle",
            "FreeTriangle",
            "Line",
            "FreeLine",
            "BezierCurve",
            "FreeBezierCurve",
        ],
    },
    "widgets": {
        "description": "Interactive UI widgets and controls",
        "classes": [
            "Button",
            "RadioButton",
            "ToolButton",
            "CheckBox",
            "ComboBox",
            "Label",
            "Image",
            "ImageWithProvider",
            "Plot",
            "ColorWidget",
            "ProgressBar",
            "TreeView",
        ],
    },
    "containers": {
        "description": "Layout containers and frames",
        "classes": ["Frame", "CanvasFrame", "ScrollingFrame", "CollapsableFrame", "HStack", "VStack", "ZStack"],
    },
    "layouts": {"description": "Layout management and positioning", "classes": ["VGrid", "HGrid", "Placer"]},
    "inputs": {
        "description": "Input controls and field widgets",
        "classes": [
            "FloatSlider",
            "IntSlider",
            "FloatDrag",
            "IntDrag",
            "MultiFloatField",
            "MultiIntField",
            "AbstractMultiField",
        ],
    },
    "windows": {
        "description": "Windows, dialogs, and menus",
        "classes": ["Window", "MainWindow", "Menu", "MenuBar", "Tooltip", "Separator"],
    },
    "scene": {
        "description": "3D scene UI components from omni.ui.scene",
        "classes": ["Line", "Curve", "Rectangle", "Arc", "Image", "Points", "PolygonMesh", "TexturedMesh", "Label"],
    },
    "units": {"description": "Unit and measurement types", "classes": ["Pixel", "Percent", "Fraction"]},
    "system": {"description": "System and styling components", "classes": ["Style"]},
}


def get_classes_directory() -> Path:
    """Get the path to the classes directory."""
    # This file is at: src/omni_aiq_omni_ui/functions/get_class_instructions.py
    # Classes are at: src/omni_aiq_omni_ui/data/instructions/classes/
    return Path(__file__).parent.parent / "data" / "instructions" / "classes"


def normalize_class_name(class_name: str) -> tuple[str, str]:
    """
    Normalize class name and determine if it's a scene class.

    Args:
        class_name: Class name like "Button", "scene.Line", or "omni.ui.scene.Line"

    Returns:
        Tuple of (normalized_name, category_hint)
    """
    # Remove omni.ui prefix if present
    if class_name.startswith("omni.ui."):
        class_name = class_name[8:]  # Remove "omni.ui."

    # Check if it's a scene class
    if class_name.startswith("scene."):
        return class_name.replace("scene.", ""), "scene"

    return class_name, None


def find_class_file(class_name: str, classes_dir: Path) -> Optional[tuple[Path, str]]:
    """
    Find the file for a given class name with smart resolution.

    Args:
        class_name: The class name to find
        classes_dir: The classes directory path

    Returns:
        Tuple of (file_path, category) if found, None otherwise
    """
    # Try multiple variations of the class name
    search_variations = [
        class_name,  # Original name
    ]

    # Add variations with prefixes removed/added
    if class_name.startswith("omni.ui."):
        search_variations.append(class_name[8:])  # Remove "omni.ui."
    else:
        search_variations.append(f"omni.ui.{class_name}")  # Add "omni.ui."

    if class_name.startswith("omni.ui.scene."):
        search_variations.append(class_name[8:])  # "scene.ClassName"
        search_variations.append(class_name[15:])  # "ClassName"
    elif class_name.startswith("scene."):
        search_variations.append(class_name[6:])  # "ClassName"
        search_variations.append(f"omni.ui.{class_name}")  # "omni.ui.scene.ClassName"
    elif not class_name.startswith("omni.ui.") and not class_name.startswith("scene."):
        search_variations.append(f"scene.{class_name}")  # Try as scene class
        search_variations.append(f"omni.ui.scene.{class_name}")

    # Remove duplicates while preserving order
    unique_variations = []
    for var in search_variations:
        if var not in unique_variations:
            unique_variations.append(var)

    # Try each variation
    for variation in unique_variations:
        result = _find_class_file_single(variation, classes_dir)
        if result:
            return result

    return None


def _find_class_file_single(class_name: str, classes_dir: Path) -> Optional[tuple[Path, str]]:
    """Helper function to find a single class file variation."""
    normalized_name, category_hint = normalize_class_name(class_name)

    # If we have a category hint (like scene), search there first
    if category_hint and category_hint in CATEGORIES:
        category_dir = classes_dir / category_hint
        # Try scene_ClassName format for scene classes
        scene_file = category_dir / f"scene_{normalized_name}.md"
        if scene_file.exists():
            return scene_file, category_hint
        # Try regular format
        regular_file = category_dir / f"{normalized_name}.md"
        if regular_file.exists():
            return regular_file, category_hint

    # Search all categories with case-insensitive matching
    for category, info in CATEGORIES.items():
        # Try exact match first
        if normalized_name in info["classes"]:
            category_dir = classes_dir / category
            file_path = category_dir / f"{normalized_name}.md"
            if file_path.exists():
                return file_path, category

            # For scene classes, try the scene_ prefix format
            if category == "scene":
                scene_file_path = category_dir / f"scene_{normalized_name}.md"
                if scene_file_path.exists():
                    return scene_file_path, category

        # Try case-insensitive match
        for class_in_category in info["classes"]:
            if normalized_name.lower() == class_in_category.lower():
                category_dir = classes_dir / category
                file_path = category_dir / f"{class_in_category}.md"
                if file_path.exists():
                    return file_path, category

                # For scene classes, try the scene_ prefix format
                if category == "scene":
                    scene_file_path = category_dir / f"scene_{class_in_category}.md"
                    if scene_file_path.exists():
                        return scene_file_path, category

    return None


async def get_class_instructions(class_names) -> Dict[str, Any]:
    """
    Retrieve specific OmniUI class instructions for one or more classes.

    Args:
        class_names: List of class names to look up, or None to list categories. Can be:
            - Simple name: "Button", "Label", "TreeView"
            - Scene class: "scene.Line", "scene.Rectangle"
            - Full name: "omni.ui.Button", "omni.ui.scene.Line"

    Returns:
        Dictionary with:
        - success: Whether retrieval was successful
        - result: The class instruction content if successful (single or multiple)
        - error: Error message if failed
        - metadata: Additional information about the class(es)
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "class_names": class_names,
        "is_none": class_names is None,
        "is_string": isinstance(class_names, str),
        "is_list": isinstance(class_names, list),
        "count": len(class_names) if isinstance(class_names, list) else (1 if isinstance(class_names, str) else 0),
    }

    success = True
    error_msg = None

    try:
        # Handle None input (list categories)
        if class_names is None:
            return await list_class_categories()

        # Handle list input
        elif isinstance(class_names, list):
            if len(class_names) == 0:
                return {"success": False, "error": "class_names array cannot be empty", "result": None}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(class_names) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"class_names contains empty or non-string values at indices: {empty_names}",
                    "result": None,
                }
        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(class_names, str):
            if not class_names.strip():
                return {"success": False, "error": "class_name cannot be empty or whitespace", "result": None}
            class_names = [class_names]
        else:
            actual_type = type(class_names).__name__
            return {
                "success": False,
                "error": f"class_names must be None or a list of strings, but got {actual_type}: {class_names}",
                "result": None,
            }

        logger.info(f"Retrieving instructions for {len(class_names)} OmniUI class(es): {class_names}")

        classes_dir = get_classes_directory()
        results = []
        errors = []
        all_metadata = []

        for class_name in class_names:
            try:
                # Find the class file
                file_info = find_class_file(class_name, classes_dir)
                if not file_info:
                    error_msg = f"Class '{class_name}' not found"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    results.append({"class_name": class_name, "error": error_msg, "content": None})
                    continue

                file_path, category = file_info

                # Read the class instruction content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    error_msg = f"Failed to read class file for '{class_name}': {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    results.append({"class_name": class_name, "error": error_msg, "content": None})
                    continue

                # Prepare metadata - use the actual class name from the file
                file_stem = file_path.stem  # e.g., "TreeView" from "TreeView.md"
                if file_stem.startswith("scene_"):
                    actual_class_name = file_stem[6:]  # Remove "scene_" prefix
                else:
                    actual_class_name = file_stem

                metadata = {
                    "class_name": class_name,
                    "normalized_name": actual_class_name,  # Use actual class name from file
                    "category": category,
                    "category_description": CATEGORIES[category]["description"],
                    "file_path": str(file_path.relative_to(classes_dir)),
                    "content_length": len(content),
                    "line_count": content.count("\n") + 1,
                }

                all_metadata.append(metadata)
                results.append({"class_name": class_name, "content": content, "metadata": metadata})

                logger.info(
                    f"Successfully retrieved class instructions for '{class_name}' from {category} category ({metadata['line_count']} lines)"
                )

            except Exception as e:
                error_msg = f"Unexpected error for '{class_name}': {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                results.append({"class_name": class_name, "error": error_msg, "content": None})

        # Return single result if only one class was requested
        if len(class_names) == 1:
            if results and results[0].get("content"):
                return {"success": True, "result": results[0]["content"], "metadata": results[0].get("metadata")}
            else:
                # Create helpful error message for single class
                error_lines = [f"Class '{class_names[0]}' not found.\n\nAvailable classes by category:"]
                for category, info in CATEGORIES.items():
                    error_lines.append(f"\n**{category}** ({len(info['classes'])} classes):")
                    for cls in sorted(info["classes"]):
                        if category == "scene":
                            error_lines.append(f"  - omni.ui.scene.{cls}")
                        else:
                            error_lines.append(f"  - omni.ui.{cls}")
                return {
                    "success": False,
                    "error": "\n".join(error_lines),
                    "result": None,
                }
        else:
            # Format multiple results
            successful_results = [r for r in results if r.get("content")]
            failed_results = [r for r in results if not r.get("content")]

            # Combine all content with headers
            combined_content = []
            for result in successful_results:
                combined_content.append(f"# Class: {result['class_name']}")
                combined_content.append(f"## Category: {result['metadata']['category']}")
                combined_content.append(result["content"])
                combined_content.append("\n---\n")

            if not successful_results:
                return {"success": False, "error": "; ".join(errors), "result": None}

            return {
                "success": True,
                "result": "\n".join(combined_content),
                "metadata": {
                    "total_requested": len(class_names),
                    "successful": len(successful_results),
                    "failed": len(failed_results),
                    "classes": all_metadata,
                    "errors": errors if errors else None,
                },
            }

    except Exception as e:
        logger.error(f"Unexpected error retrieving class instructions: {e}")
        error_msg = f"Unexpected error: {str(e)}"
        success = False
        return {"success": False, "error": error_msg, "result": None}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_class_instructions",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )


async def list_class_categories() -> Dict[str, Any]:
    """
    List all available class categories with their descriptions and class counts.

    Returns:
        Dictionary with:
        - success: Whether listing was successful
        - result: Formatted list of categories
        - categories: Detailed information about each category
    """
    try:
        result_lines = ["# OmniUI Class Categories\n"]
        categories_info = []

        for category, info in CATEGORIES.items():
            class_count = len(info["classes"])
            categories_info.append(
                {
                    "name": category,
                    "description": info["description"],
                    "class_count": class_count,
                    "classes": info["classes"],
                }
            )

            result_lines.append(f"\n## {category} ({class_count} classes)")
            result_lines.append(f"{info['description']}")
            result_lines.append(f"\nClasses: {', '.join(info['classes'])}")

        result_lines.append(f"\n\nTotal categories: {len(CATEGORIES)}")
        result_lines.append(f"Total classes: {sum(len(info['classes']) for info in CATEGORIES.values())}")

        return {"success": True, "result": "\n".join(result_lines), "categories": categories_info}

    except Exception as e:
        logger.error(f"Failed to list class categories: {e}")
        return {"success": False, "error": f"Failed to list categories: {str(e)}", "result": None}


async def list_classes_in_category(category: str) -> Dict[str, Any]:
    """
    List all classes in a specific category.

    Args:
        category: The category name (e.g., "widgets", "shapes", "scene")

    Returns:
        Dictionary with:
        - success: Whether listing was successful
        - result: Formatted list of classes in the category
        - classes: List of class names in the category
    """
    try:
        # Handle None or empty category
        if not category or category.lower() == "none":
            available_categories = list(CATEGORIES.keys())
            return {
                "success": False,
                "error": f"No category specified. Available categories: {', '.join(available_categories)}",
                "result": None,
            }

        if category not in CATEGORIES:
            available_categories = list(CATEGORIES.keys())
            return {
                "success": False,
                "error": f"Unknown category '{category}'. Available categories: {', '.join(available_categories)}",
                "result": None,
            }

        category_info = CATEGORIES[category]
        classes = category_info["classes"]

        result_lines = [f"# {category.title()} Classes"]
        result_lines.append(f"\n{category_info['description']}")
        result_lines.append(f"\nThis category contains {len(classes)} classes:\n")

        for class_name in sorted(classes):
            if category == "scene":
                full_name = f"omni.ui.scene.{class_name}"
            else:
                full_name = f"omni.ui.{class_name}"
            result_lines.append(f"- **{full_name}**")

        return {
            "success": True,
            "result": "\n".join(result_lines),
            "classes": classes,
            "category": category,
            "description": category_info["description"],
        }

    except Exception as e:
        logger.error(f"Failed to list classes in category '{category}': {e}")
        return {"success": False, "error": f"Failed to list classes: {str(e)}", "result": None}
