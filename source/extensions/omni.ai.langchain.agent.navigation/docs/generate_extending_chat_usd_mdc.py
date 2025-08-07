## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import argparse
import os
from pathlib import Path


def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def generate_mdc(docs_dir: Path = None, output_path: Path = None, verbose: bool = False):
    """Generate the extending-chat-usd.mdc file."""
    # Get the base directory
    base_dir = Path(__file__).parent.parent

    # Set default docs directory if not provided
    if docs_dir is None:
        docs_dir = Path(__file__).parent

    # Set default output path if not provided
    if output_path is None:
        output_dir = base_dir.parent.parent.parent / ".cursor" / "rules"
        output_path = output_dir / "extending-chat-usd.mdc"
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_path.parent, exist_ok=True)

    # Source file path
    source_file = docs_dir / "extending_chat_usd_with_custom_agents.md"

    if not source_file.exists():
        print(f"Error: {source_file} does not exist")
        return

    print(f"Starting documentation generation from {source_file}")
    print(f"Output will be written to {output_path}")

    # Read source file content
    content = read_file(str(source_file))

    # Header content
    header = """---
description: A high-level overview of how to extend Chat USD with custom agents, using the Navigation Agent as a reference implementation.
globs:
---

"""

    # Use newline='\n' to force Unix line endings in the output file
    try:
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            # Write header
            f.write(header)

            # Write content
            f.write(content)

        print(f"\nGenerated {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")

        # Verify the file was created
        if os.path.exists(output_path):
            print(f"File successfully created at {output_path}")
        else:
            print(f"ERROR: File was not created at {output_path}")

    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MDC file from extending_chat_usd_with_custom_agents.md")
    parser.add_argument("--docs-dir", type=str, help="Path to the documentation directory")
    parser.add_argument("--output", type=str, help="Path to the output MDC file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir) if args.docs_dir else None
    output_path = Path(args.output) if args.output else None

    generate_mdc(docs_dir, output_path, args.verbose)
