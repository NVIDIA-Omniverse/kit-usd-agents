## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Utility functions for handling configuration files."""

import re
import sys
from typing import Optional
from pathlib import Path


def replace_md_file_references(text: Optional[str], relative_to_path: Optional[str] = None) -> Optional[str]:
    """
    Scans a string and replaces any markdown file references with the actual content of the files.

    A markdown file reference has the format {path/to/file.md} where:
    1. It is enclosed in curly braces
    2. The file extension is .md
    3. The path can be relative or absolute

    The function checks for files in the following order:
    1. Relative to the provided relative_to_path
    2. Relative to the current directory
    3. Relative to the executable Python file

    Args:
        text: The string to scan for file references
        relative_to_path: The base path for resolving relative file paths

    Returns:
        The updated string with file references replaced by their contents (if found)
    """
    # Handle None input
    if text is None:
        return text
    
    # Pattern to match {path/to/file.md}
    pattern = r"\{([^{}]+\.md)\}"
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    if not matches:
        return text
    
    result = text
    
    for match in matches:
        # The match is the path inside the curly braces
        file_path_str = match
        
        # List of paths to check
        paths_to_check = []
        
        # 1. Relative to relative_to_path
        if relative_to_path:
            paths_to_check.append(Path(relative_to_path) / file_path_str)
        
        # 2. Relative to current directory
        paths_to_check.append(Path.cwd() / file_path_str)
        
        # 3. Relative to executable Python file
        if sys.argv[0]:
            executable_dir = Path(sys.argv[0]).parent
            paths_to_check.append(executable_dir / file_path_str)
        
        # Try to find and read the file
        file_content = None
        for path in paths_to_check:
            try:
                if path.exists() and path.is_file():
                    with open(path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    break
            except (OSError, IOError):
                # Continue to next path if there's an error
                continue
        
        # Replace the reference if file was found
        if file_content is not None:
            result = result.replace(f"{{{match}}}", file_content)
        # If not found, leave the reference as is
    
    return result
