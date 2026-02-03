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
Script to concatenate all Python files from omni_ui_mcp/src folder
and analyze their token counts using tiktoken.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import tiktoken


def count_tokens(text, encoding="cl100k_base"):
    """Count tokens in text using tiktoken."""
    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))


def get_file_size(file_path):
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def format_size(size_bytes):
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_python_files(directory: Path) -> List[Path]:
    """Recursively get all Python files from a directory."""
    python_files = []
    for file_path in directory.rglob("*.py"):
        python_files.append(file_path)
    return sorted(python_files)


def create_tree_structure(base_path: Path, files: List[Path]) -> str:
    """Create a visual tree structure of the files."""
    tree_lines = []

    # Build a tree structure
    tree = {}
    for file_path in files:
        parts = file_path.relative_to(base_path).parts
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Add file
        if parts:
            current[parts[-1]] = None

    def build_tree_string(node, prefix="", is_last=True):
        items = list(node.items()) if node else []
        for i, (name, subtree) in enumerate(items):
            is_last_item = i == len(items) - 1

            # Connector
            if prefix:
                connector = "└── " if is_last_item else "├── "
            else:
                connector = ""

            # Add to tree
            if subtree is None:  # It's a file
                tree_lines.append(f"{prefix}{connector}📄 {name}")
            else:  # It's a directory
                tree_lines.append(f"{prefix}{connector}📁 {name}/")
                # Recurse
                extension = "    " if is_last_item else "│   "
                new_prefix = prefix + extension if prefix else "  "
                build_tree_string(subtree, new_prefix, is_last_item)

    build_tree_string({"src/": tree})
    return "\n".join(tree_lines)


def concatenate_src_files(source_dir, output_file):
    """Concatenate all Python files from the src directory."""

    source_path = Path(source_dir)
    src_path = source_path / "src"

    if not src_path.exists():
        print(f"❌ Source directory not found: {src_path}")
        return None, 0

    # Get all Python files
    python_files = get_python_files(src_path)

    total_content = []
    file_stats = []
    folder_stats = {}
    total_tokens = 0
    total_bytes = 0

    # Add header
    header = "=" * 80 + "\n"
    header += "OMNI_UI_MCP SRC FILES CONCATENATION\n"
    header += "=" * 80 + "\n\n"
    total_content.append(header)

    # Add file tree structure
    tree_header = "FILE STRUCTURE:\n" + "-" * 40 + "\n\n"
    total_content.append(tree_header)
    tree_structure = create_tree_structure(src_path, python_files)
    total_content.append(tree_structure)
    total_content.append("\n\n" + "=" * 80 + "\n\n")

    # Process each Python file
    for file_path in python_files:
        relative_path = file_path.relative_to(source_path)

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get file stats
            file_size = get_file_size(file_path)
            token_count = count_tokens(content)

            total_bytes += file_size
            total_tokens += token_count

            # Store stats for summary
            file_stats.append(
                {"path": str(relative_path), "name": file_path.name, "size": file_size, "tokens": token_count}
            )

            # Track folder stats
            folder = str(relative_path.parent)
            if folder not in folder_stats:
                folder_stats[folder] = {"files": 0, "tokens": 0, "size": 0}
            folder_stats[folder]["files"] += 1
            folder_stats[folder]["tokens"] += token_count
            folder_stats[folder]["size"] += file_size

            # Add to concatenated content
            file_header = f"\n{'=' * 80}\n"
            file_header += f"FILE: {relative_path}\n"
            file_header += f"Size: {format_size(file_size)} | Tokens: {token_count:,}\n"
            file_header += f"{'=' * 80}\n\n"

            total_content.append(file_header)
            total_content.append(content)
            total_content.append("\n")

            print(f"✅ Processed: {relative_path} ({format_size(file_size)}, {token_count:,} tokens)")

        except Exception as e:
            print(f"❌ Error processing {relative_path}: {e}")

    # Combine all content
    final_content = "".join(total_content)

    # Create detailed summary
    summary = f"\nSUMMARY STATISTICS:\n" + "-" * 40 + "\n"
    summary += f"Total Python files processed: {len(file_stats)}\n"
    summary += f"Total size: {format_size(total_bytes)}\n"
    summary += f"Total tokens: {total_tokens:,}\n\n"

    # Folder-by-folder breakdown
    summary += "FOLDER BREAKDOWN:\n" + "-" * 40 + "\n"
    for folder, stats in sorted(folder_stats.items(), key=lambda x: x[1]["tokens"], reverse=True):
        summary += f"📁 {folder}/\n"
        summary += f"   Files: {stats['files']:>3} | "
        summary += f"Size: {format_size(stats['size']):>10} | "
        summary += f"Tokens: {stats['tokens']:>10,}\n"

    summary += "\n"

    # Top 10 largest files by tokens
    summary += "TOP 10 FILES BY TOKEN COUNT:\n" + "-" * 40 + "\n"
    sorted_files = sorted(file_stats, key=lambda x: x["tokens"], reverse=True)[:10]
    for i, stat in enumerate(sorted_files, 1):
        summary += f"{i:2}. {stat['path']:<50} {stat['tokens']:>10,} tokens\n"

    summary += "\n"

    # File type breakdown
    summary += "FILE TYPE ANALYSIS:\n" + "-" * 40 + "\n"
    file_types = {}
    for stat in file_stats:
        name = stat["name"]
        if name.startswith("register_"):
            file_type = "Registration Files"
        elif name.startswith("test_"):
            file_type = "Test Files"
        elif name == "__init__.py":
            file_type = "Init Files"
        elif "service" in stat["path"]:
            file_type = "Service Files"
        elif "function" in stat["path"]:
            file_type = "Function Files"
        elif "model" in stat["path"]:
            file_type = "Model Files"
        elif "util" in stat["path"]:
            file_type = "Utility Files"
        else:
            file_type = "Other Files"

        if file_type not in file_types:
            file_types[file_type] = {"count": 0, "tokens": 0}
        file_types[file_type]["count"] += 1
        file_types[file_type]["tokens"] += stat["tokens"]

    for file_type, stats in sorted(file_types.items(), key=lambda x: x[1]["tokens"], reverse=True):
        summary += f"  {file_type:<20} Files: {stats['count']:>3} | Tokens: {stats['tokens']:>10,}\n"

    summary += "\n" + "=" * 80 + "\n"

    # Insert summary after header and file structure
    final_parts = final_content.split("=" * 80, 3)
    if len(final_parts) >= 4:
        final_content = (
            final_parts[0]
            + "=" * 80
            + final_parts[1]
            + "=" * 80
            + final_parts[2]
            + "=" * 80
            + summary
            + "=" * 80
            + final_parts[3]
        )

    # Write to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_content)

    # Count tokens in final file
    final_tokens = count_tokens(final_content)

    print("\n" + "=" * 80)
    print(f"✨ Concatenation complete!")
    print(f"📄 Output file: {output_file}")
    print(f"📊 Final file size: {format_size(len(final_content.encode('utf-8')))}")
    print(f"🔢 Final token count: {final_tokens:,}")
    print("=" * 80)

    # Print folder summary
    print("\n📁 FOLDER SUMMARY:")
    for folder, stats in sorted(folder_stats.items(), key=lambda x: x[1]["tokens"], reverse=True):
        print(f"  {folder:<40} {stats['files']:>3} files, {stats['tokens']:>10,} tokens")

    return file_stats, total_tokens


if __name__ == "__main__":
    # Configure paths
    source_dir = "/home/horde/repos/kit-lc-agent/source/mcp/omni_ui_mcp"
    output_file = "/home/horde/repos/kit-lc-agent/source/mcp/kit_mcp/concatenated_src_files.md"

    # Run concatenation
    concatenate_src_files(source_dir, output_file)
