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
Script to combine all MCP documentation files into a single comprehensive document.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def read_file(filepath: Path) -> str:
    """Read a file and return its contents."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def get_documentation_files() -> List[Tuple[str, Path]]:
    """Get list of documentation files to combine in order."""
    docs_dir = Path(__file__).parent / "docs"

    # Define the files in the order they should appear
    files = [
        ("Overview", docs_dir / "README.md"),
        ("Quick Start Guide", docs_dir / "QUICK_START.md"),
        ("Detailed Tool Documentation", docs_dir / "MCP_TOOLS_DETAILED.md"),
        ("Architecture Documentation", docs_dir / "MCP_ARCHITECTURE.md"),
        ("API Reference", docs_dir / "API_REFERENCE.md"),
        ("Feature Documentation", docs_dir / "FEATURE_DOCUMENTATION.md"),
    ]

    # Only include files that exist
    existing_files = [(name, path) for name, path in files if path.exists()]

    return existing_files


def create_table_of_contents(sections: List[Tuple[str, str]]) -> str:
    """Create a table of contents with links to sections."""
    toc = ["# Table of Contents\n"]

    for i, (name, _) in enumerate(sections, 1):
        # Create anchor-friendly ID
        anchor = name.lower().replace(" ", "-").replace("(", "").replace(")", "")
        toc.append(f"{i}. [{name}](#{anchor})")

    return "\n".join(toc)


def process_section_content(content: str, level_offset: int = 0) -> str:
    """Process section content to adjust heading levels if needed."""
    if level_offset == 0:
        return content

    lines = content.split("\n")
    processed_lines = []

    for line in lines:
        if line.startswith("#"):
            # Count the number of # characters
            hash_count = len(line) - len(line.lstrip("#"))
            # Add offset to heading level
            new_hash = "#" * (hash_count + level_offset)
            # Replace the heading marker
            processed_line = new_hash + line[hash_count:]
            processed_lines.append(processed_line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


def combine_documentation():
    """Combine all documentation files into a single document."""

    print("Starting documentation combination process...")

    # Get list of files to combine
    files = get_documentation_files()

    if not files:
        print("No documentation files found!")
        return

    print(f"Found {len(files)} documentation files to combine")

    # Start building the combined document
    combined = []

    # Add header
    combined.append("# OmniUI MCP Server - Complete Documentation")
    combined.append("")
    combined.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    combined.append("")
    combined.append("---")
    combined.append("")

    # Prepare sections for TOC
    sections = []
    section_contents = []

    # Read all files first
    for name, filepath in files:
        print(f"Reading: {filepath.name}")
        content = read_file(filepath)
        if content:
            sections.append((name, content))
            section_contents.append(content)

    # Add table of contents
    if sections:
        toc = create_table_of_contents(sections)
        combined.append(toc)
        combined.append("")
        combined.append("---")
        combined.append("")

    # Add each section
    for name, content in sections:
        # Add section separator
        combined.append("---")
        combined.append("")
        combined.append(f"# {name}")
        combined.append("")

        # Process and add content
        # Skip the first heading if it duplicates the section name
        lines = content.split("\n")
        if lines and lines[0].startswith("# "):
            # Skip the first line if it's a top-level heading
            content = "\n".join(lines[1:]).strip()

        combined.append(content)
        combined.append("")

    # Add footer
    combined.append("---")
    combined.append("")
    combined.append("## Document Information")
    combined.append("")
    combined.append(f"- **Total Sections**: {len(sections)}")
    combined.append(f"- **Generated Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    combined.append(f"- **MCP Server Version**: 0.4.5")
    combined.append(f"- **Documentation Format**: Markdown")
    combined.append("")
    combined.append("---")
    combined.append("")
    combined.append("*This document was automatically generated by combining multiple documentation files.*")

    # Write the combined document
    output_path = Path(__file__).parent / "docs" / "COMBINED_DOCUMENTATION.md"

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined))

        print(f"\n✅ Successfully created combined documentation at:")
        print(f"   {output_path}")

        # Calculate statistics
        total_lines = len("\n".join(combined).split("\n"))
        total_chars = len("\n".join(combined))

        print(f"\n📊 Statistics:")
        print(f"   - Total lines: {total_lines:,}")
        print(f"   - Total characters: {total_chars:,}")
        print(f"   - Sections combined: {len(sections)}")

    except Exception as e:
        print(f"\n❌ Error writing combined documentation: {e}")
        return

    print("\n✨ Documentation combination complete!")


def main():
    """Main entry point."""
    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Run the combination
    combine_documentation()


if __name__ == "__main__":
    main()
