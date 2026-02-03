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
Combine all styling documentation files into a single markdown file.
Overview.md will be placed at the top, followed by all other files in a logical order.
"""

import os
from datetime import datetime
from pathlib import Path


def read_file_content(file_path: str) -> str:
    """Read and return the content of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def combine_documentation_files():
    """Combine all specified documentation files into a single markdown file."""

    # Get the repository root (assumes this script is in source/mcp/omni_ui_mcp/utils/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent.parent
    base_path = repo_root / "source" / "aiq" / "omni_ui_fns" / "src" / "omni_ui_fns" / "data" / "styles"

    # Files to combine in specific order (overview first, then logical grouping)
    files_to_combine = [
        "overview.md",  # Overview at the top
        "styling.md",  # General styling concepts
        "ui_style_best_practice.md",  # Best practices for UI styling
        "units.md",  # Units system
        "fonts.md",  # Typography
        "shades.md",  # Colors and shading
        "window.md",  # Window styling
        "containers.md",  # Container components
        "widgets.md",  # Widget components
        "buttons.md",  # Button components
        "sliders.md",  # Slider components
        "shapes.md",  # Shape components
        "line.md",  # Line components
    ]

    # Start building the combined content
    combined_content = []

    # Add header
    combined_content.append("# Omni Kit UI Style Documentation - Complete Reference")
    combined_content.append("")
    combined_content.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    combined_content.append("")
    combined_content.append("This document combines all UI styling documentation for Omni Kit into a single reference.")
    combined_content.append("")
    combined_content.append("## Table of Contents")
    combined_content.append("")

    # Generate table of contents
    for i, filename in enumerate(files_to_combine, 1):
        title = filename.replace(".md", "").replace("_", " ").title()
        anchor = filename.replace(".md", "").lower()
        combined_content.append(f"{i}. [{title}](#{anchor})")

    combined_content.append("")
    combined_content.append("---")
    combined_content.append("")

    # Process each file
    for filename in files_to_combine:
        file_path = base_path / filename

        print(f"Processing: {filename}")

        # Read file content
        content = read_file_content(file_path)

        if content:
            # Add section separator
            section_title = filename.replace(".md", "").replace("_", " ").title()

            # Add anchor for table of contents
            anchor = filename.replace(".md", "").lower()
            combined_content.append(f'<a name="{anchor}"></a>')
            combined_content.append("")

            # Add visual separator for each major section
            combined_content.append("=" * 80)
            combined_content.append("")
            combined_content.append(f"# {section_title} Section")
            combined_content.append("")
            combined_content.append(f"*Source: {filename}*")
            combined_content.append("")
            combined_content.append("=" * 80)
            combined_content.append("")

            # Add the actual content
            combined_content.append(content)

            # Add spacing between sections
            combined_content.append("")
            combined_content.append("")
        else:
            print(f"Warning: Could not read content from {filename}")

    # Add footer
    combined_content.append("---")
    combined_content.append("")
    combined_content.append("## End of Documentation")
    combined_content.append("")
    combined_content.append(
        "This combined documentation was automatically generated from the individual documentation files."
    )
    combined_content.append(f"Total sections included: {len(files_to_combine)}")

    # Join all content
    final_content = "\n".join(combined_content)

    # Save to file
    output_path = base_path / "all_styling_combined.md"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"\nSuccessfully created combined documentation at: {output_path}")

        # Calculate some statistics
        line_count = final_content.count("\n")
        char_count = len(final_content)
        word_count = len(final_content.split())

        print(f"\nStatistics:")
        print(f"  - Total lines: {line_count:,}")
        print(f"  - Total characters: {char_count:,}")
        print(f"  - Total words: {word_count:,}")
        print(f"  - File size: {len(final_content.encode('utf-8')) / 1024:.2f} KB")

    except Exception as e:
        print(f"Error saving combined file: {e}")


def main():
    """Main function to run the documentation combination."""
    print("Starting documentation combination process...")
    print("=" * 60)

    combine_documentation_files()

    print("=" * 60)
    print("Process complete!")


if __name__ == "__main__":
    main()
