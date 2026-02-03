#!/usr/bin/env python3
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

"""
Script to split the large classes.md file into categorized individual class files.

This script reads the classes.md file and splits it at each '## omni.ui.' header,
organizing classes into logical categories for better context window management.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Define the categories and their class mappings
CLASS_CATEGORIES = {
    "models": ["AbstractValueModel", "AbstractItemModel", "AbstractItemDelegate"],
    "shapes": [
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
    "widgets": [
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
    ],
    "containers": ["Frame", "CanvasFrame", "ScrollingFrame", "CollapsableFrame", "HStack", "VStack", "ZStack"],
    "layouts": ["VGrid", "HGrid", "Placer"],
    "inputs": [
        "FloatSlider",
        "IntSlider",
        "FloatDrag",
        "IntDrag",
        "MultiFloatField",
        "MultiIntField",
        "AbstractMultiField",
    ],
    "windows": ["Window", "MainWindow", "Menu", "MenuBar", "Tooltip", "Separator"],
    "scene": [
        # All scene classes start with omni.ui.scene
        "Line",
        "Curve",
        "Rectangle",
        "Arc",
        "Image",
        "Points",
        "PolygonMesh",
        "TexturedMesh",
        "Label",
    ],
    "units": ["Pixel", "Percent", "Fraction"],
    "system": ["Style"],
}


def create_class_to_category_mapping() -> Dict[str, str]:
    """Create a mapping from class name to category."""
    class_to_category = {}

    for category, classes in CLASS_CATEGORIES.items():
        for class_name in classes:
            class_to_category[class_name] = category

    return class_to_category


def get_category_for_class(class_name: str, class_to_category: Dict[str, str]) -> str:
    """Get the category for a given class name."""
    # Handle scene classes specially
    if class_name.startswith("scene."):
        base_class = class_name.replace("scene.", "")
        if base_class in class_to_category.get("scene", []):
            return "scene"

    # Handle regular classes
    return class_to_category.get(class_name, "uncategorized")


def split_classes_file(input_file: Path, output_dir: Path) -> None:
    """Split the classes.md file into individual categorized files."""

    # Create class to category mapping
    class_to_category = create_class_to_category_mapping()

    # Read the input file
    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by class headers
    class_sections = re.split(r"^## (omni\.ui\.(?:scene\.)?[\w]+)", content, flags=re.MULTILINE)

    # First element is content before any class header (should be empty or minimal)
    if class_sections[0].strip():
        print(f"Warning: Found content before first class header: {class_sections[0][:100]}...")

    # Process each class section (pairs: class_name, content)
    processed_classes = []
    uncategorized_classes = []

    for i in range(1, len(class_sections), 2):
        if i + 1 >= len(class_sections):
            break

        full_class_name = class_sections[i]  # e.g., "omni.ui.Button" or "omni.ui.scene.Line"
        class_content = class_sections[i + 1]

        # Extract the actual class name
        if full_class_name.startswith("omni.ui.scene."):
            class_name = full_class_name.replace("omni.ui.", "")  # "scene.Line"
        else:
            class_name = full_class_name.replace("omni.ui.", "")  # "Button"

        # Get category
        category = get_category_for_class(class_name, class_to_category)

        if category == "uncategorized":
            uncategorized_classes.append(class_name)
            category = "misc"  # Put uncategorized classes in misc folder

        # Create category directory
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create filename (replace dots with underscores for scene classes)
        filename = class_name.replace(".", "_") + ".md"
        output_file = category_dir / filename

        # Format content with header
        formatted_content = f"# {full_class_name}\n\n{class_content.lstrip()}"

        # Write the file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_content)

        processed_classes.append((class_name, category, str(output_file)))
        print(f"Created: {category}/{filename}")

    # Report results
    print(f"\nProcessing complete!")
    print(f"Processed {len(processed_classes)} classes")
    print(f"Categories created: {len(set(cat for _, cat, _ in processed_classes))}")

    if uncategorized_classes:
        print(f"\nUncategorized classes (moved to 'misc'): {uncategorized_classes}")

    # Print category summary
    print("\nCategory summary:")
    category_counts = {}
    for _, category, _ in processed_classes:
        category_counts[category] = category_counts.get(category, 0) + 1

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} classes")


def create_category_index_files(output_dir: Path) -> None:
    """Create index files for each category listing available classes."""

    for category_name in CLASS_CATEGORIES.keys():
        category_dir = output_dir / category_name
        if not category_dir.exists():
            continue

        index_file = category_dir / "_index.md"
        classes_in_category = CLASS_CATEGORIES[category_name]

        content = f"# {category_name.title()} Classes\n\n"
        content += f"This category contains {len(classes_in_category)} omni.ui classes:\n\n"

        for class_name in sorted(classes_in_category):
            if category_name == "scene":
                full_name = f"omni.ui.scene.{class_name}"
                filename = f"scene_{class_name}.md"
            else:
                full_name = f"omni.ui.{class_name}"
                filename = f"{class_name}.md"

            content += f"- **{full_name}** (`{filename}`)\n"

        with open(index_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Created index: {category_name}/_index.md")


def main():
    """Main function to split the classes file."""

    # Paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "src" / "omni_ui_mcp" / "data" / "instructions" / "classes.md"
    output_dir = script_dir / "src" / "omni_ui_mcp" / "data" / "instructions" / "classes"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Input file size: {input_file.stat().st_size:,} bytes")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split the file
    split_classes_file(input_file, output_dir)

    # Create category index files
    create_category_index_files(output_dir)

    print(f"\nAll done! Classes split into: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
