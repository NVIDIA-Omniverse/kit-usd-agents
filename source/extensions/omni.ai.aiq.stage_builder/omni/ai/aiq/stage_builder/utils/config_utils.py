## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Utility functions for handling configuration files."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import carb


def replace_md_file_references(config: Dict[str, Any], extension_path: Path) -> Dict[str, Any]:
    """
    Recursively scans a configuration dictionary and replaces any markdown file references
    with the actual content of the files.

    A markdown file reference has the format {path/to/file.md} where:
    1. It is enclosed in curly braces
    2. The file extension is .md
    3. The path is relative to extension_path
    4. The file exists

    Args:
        config: The configuration dictionary to scan
        extension_path: The base path for resolving relative file paths

    Returns:
        The updated configuration dictionary with file references replaced by their contents
    """
    if config is None:
        return config

    # Pattern to match {path/to/file.md}
    pattern = r"\{([^{}]+\.md)\}"

    def process_value(value: Any) -> Any:
        """Process a value in the configuration dictionary."""
        if isinstance(value, str):
            # Check if the string contains a markdown file reference
            matches = re.findall(pattern, value)

            if not matches:
                return value

            result = value
            for match in matches:
                # The match is the path inside the curly braces
                file_path = extension_path / match

                # Check if the file exists
                if file_path.exists() and file_path.is_file():
                    # Security check - ensure the logical path stays within extension directory
                    try:
                        # Normalize paths (remove redundant separators, resolve .. and .)
                        # but DON'T follow symlinks
                        normalized_file = os.path.normpath(file_path)
                        normalized_extension = os.path.normpath(extension_path)

                        # Convert to Path objects for comparison
                        normalized_file_path = Path(normalized_file)
                        normalized_extension_path = Path(normalized_extension)

                        # Check if the file is within the extension directory
                        # This works by checking if the extension path is a parent of the file path
                        normalized_file_path.relative_to(normalized_extension_path)
                    except ValueError:
                        # Path is outside the extension directory
                        carb.log_warn(f"[MD Replace] Skipping {match} - path escapes extension directory")
                        continue

                    # Read the file content
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_content = f.read()

                        # Replace the file reference with its content
                        result = result.replace(f"{{{match}}}", file_content)
                    except Exception as e:
                        carb.log_error(f"[MD Replace] Could not read file {file_path}: {e}")
                else:
                    carb.log_warn(f"[MD Replace] File not found: {file_path}")

            return result

        elif isinstance(value, dict):
            # Recursively process dictionaries
            return {k: process_value(v) for k, v in value.items()}

        elif isinstance(value, list):
            # Recursively process lists
            return [process_value(item) for item in value]

        else:
            # Return other types unchanged
            return value

    # Process the entire config dictionary
    return process_value(config)
