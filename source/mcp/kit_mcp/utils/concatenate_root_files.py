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
Script to concatenate all root-level files from omni_ui_mcp project
and analyze their token counts using tiktoken.
"""

import os
from pathlib import Path

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


def concatenate_root_files(source_dir, output_file):
    """Concatenate all root-level files from the source directory."""

    source_path = Path(source_dir)

    # Define root-level files to include (non-directory items)
    root_files = [
        "Dockerfile",
        "README.md",
        "VERSION.md",
        "build-docker.bat",
        "build-docker.sh",
        "catalog-info.yaml",
        "check_mcp_health.py",
        # "poetry.lock",  # Excluded due to large size
        "pyproject.toml",
        "run-local.bat",
        "run-local.sh",
        "setup-dev.bat",
        "setup-dev.sh",
        "start-server.bat",
    ]

    total_content = []
    file_stats = []
    total_tokens = 0
    total_bytes = 0

    # Add header
    header = "=" * 80 + "\n"
    header += "OMNI_UI_MCP ROOT FILES CONCATENATION\n"
    header += "=" * 80 + "\n\n"
    total_content.append(header)

    # Add directory structure first
    structure_header = "DIRECTORY STRUCTURE:\n" + "-" * 40 + "\n\n"
    total_content.append(structure_header)

    # List all items in root directory
    for item in sorted(source_path.iterdir()):
        if item.is_dir():
            total_content.append(f"📁 {item.name}/\n")
        else:
            total_content.append(f"📄 {item.name}\n")

    total_content.append("\n" + "=" * 80 + "\n\n")

    # Process each root file
    for filename in root_files:
        file_path = source_path / filename

        if not file_path.exists():
            print(f"⚠️  File not found: {filename}")
            continue

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
            file_stats.append({"name": filename, "size": file_size, "tokens": token_count})

            # Add to concatenated content
            file_header = f"\n{'=' * 80}\n"
            file_header += f"FILE: {filename}\n"
            file_header += f"Size: {format_size(file_size)} | Tokens: {token_count:,}\n"
            file_header += f"{'=' * 80}\n\n"

            total_content.append(file_header)
            total_content.append(content)
            total_content.append("\n")

            print(f"✅ Processed: {filename} ({format_size(file_size)}, {token_count:,} tokens)")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Combine all content
    final_content = "".join(total_content)

    # Add summary at the beginning
    summary = f"\nSUMMARY STATISTICS:\n" + "-" * 40 + "\n"
    summary += f"Total files processed: {len(file_stats)}\n"
    summary += f"Total size: {format_size(total_bytes)}\n"
    summary += f"Total tokens: {total_tokens:,}\n\n"

    summary += "File-by-file breakdown:\n"
    for stat in sorted(file_stats, key=lambda x: x["tokens"], reverse=True):
        summary += f"  - {stat['name']:<30} {format_size(stat['size']):>10} {stat['tokens']:>10,} tokens\n"

    summary += "\n" + "=" * 80 + "\n"

    # Insert summary after header
    final_parts = final_content.split("=" * 80, 2)
    if len(final_parts) >= 3:
        final_content = final_parts[0] + "=" * 80 + final_parts[1] + "=" * 80 + summary + "=" * 80 + final_parts[2]

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

    return file_stats, total_tokens


if __name__ == "__main__":
    # Configure paths
    source_dir = "/home/horde/repos/kit-lc-agent/source/mcp/omni_ui_mcp"
    output_file = "/home/horde/repos/kit-lc-agent/source/mcp/kit_mcp/concatenated_root_files.md"

    # Run concatenation
    concatenate_root_files(source_dir, output_file)
