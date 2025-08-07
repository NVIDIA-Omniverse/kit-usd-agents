## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
from pathlib import Path
from typing import List


def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def get_example_files(examples_dir: str) -> List[str]:
    """Get all Python files from the examples directory."""
    files = []
    if os.path.exists(examples_dir):
        for file in sorted(os.listdir(examples_dir)):
            if file.endswith(".py"):
                files.append(os.path.join(examples_dir, file))
    return files


def generate_mdc():
    """Generate the lc-agent-usd.mdc file."""
    # Get the base directory
    base_dir = Path(__file__).parent.parent
    doc_dir = base_dir / "doc"
    examples_dir = base_dir / "examples" if (base_dir / "examples").exists() else None
    src_dir = base_dir / "src" / "lc_agent_usd"

    # Core documentation files in order
    core_files = [
        doc_dir / "01_overview.md",
        doc_dir / "02_usd_knowledge_network_node.md",
        doc_dir / "03_usd_code_gen_network_node.md",
        doc_dir / "04_usd_code_interactive_network_node.md",
        doc_dir / "05_modifiers.md",
    ]

    # Header content
    header = """---
description: A high-level overview of the core components in LC Agent USD module, explaining how they work together to create a specialized system for USD development assistance.
globs: 
---

"""

    # Generate the MDC file
    output_path = base_dir / ".." / ".." / ".cursor" / "rules" / "lc-agent-usd.mdc"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use newline='\n' to force Unix line endings in the output file
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        # Write header
        f.write(header)

        # Write core documentation
        for file_path in core_files:
            if file_path.exists():
                content = read_file(str(file_path))
                f.write(content + "\n\n")

        # Write examples section if examples directory exists
        if examples_dir and os.path.exists(examples_dir):
            f.write("# Examples\n\n")
            example_files = get_example_files(str(examples_dir))
            for example_file in example_files:
                # Get the filename without path
                filename = os.path.basename(example_file)
                # Write the filename as a header
                f.write(f"## {filename}\n\n")
                # Write the file contents in a code block
                f.write("```python\n")
                f.write(read_file(example_file))
                f.write("\n```\n\n")
        
        # Add implementation examples from the source code
        f.write("# Implementation\n\n")
        
        # Add key implementation files
        implementation_files = [
            src_dir / "usd_knowledge_network_node.py",
            src_dir / "usd_code_gen_network_node.py",
            src_dir / "usd_code_network_node.py",
        ]
        
        for impl_file in implementation_files:
            if impl_file.exists():
                filename = impl_file.name
                f.write(f"## {filename}\n\n")
                f.write("```python\n")
                f.write(read_file(str(impl_file)))
                f.write("\n```\n\n")

    print(f"Generated {output_path}")


if __name__ == "__main__":
    generate_mdc() 