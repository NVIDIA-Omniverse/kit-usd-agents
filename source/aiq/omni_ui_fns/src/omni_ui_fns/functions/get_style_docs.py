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
Function to retrieve OmniUI style documentation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)


async def get_style_docs(sections) -> Dict[str, Any]:
    """Return OmniUI style documentation from the stored style files.

    Args:
        sections: List of section names to look up, or None to get combined documentation.
                 Can be a single section or multiple sections.

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing the requested style documentation
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "sections": sections,
        "is_none": sections is None,
        "is_string": isinstance(sections, str),
        "is_list": isinstance(sections, list),
        "count": len(sections) if isinstance(sections, list) else (1 if isinstance(sections, str) else 0),
    }

    success = True
    error_msg = None

    try:
        # Get the data directory path
        data_dir = Path(__file__).parent.parent / "data" / "styles"

        if not data_dir.exists():
            error_msg = f"Style documentation directory not found: {data_dir}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Available sections mapping (without .md extension)
        available_sections = {
            "buttons": "buttons.md",
            "containers": "containers.md",
            "fonts": "fonts.md",
            "line": "line.md",
            "overview": "overview.md",
            "shades": "shades.md",
            "shapes": "shapes.md",
            "sliders": "sliders.md",
            "styling": "styling.md",
            "units": "units.md",
            "widgets": "widgets.md",
            "window": "window.md",
        }

        result_data = {}

        # Handle None input (get combined documentation)
        if sections is None:
            # No specific section requested, return combined documentation
            logger.info("Retrieving complete combined style documentation")

            combined_file = data_dir / "all_styling_combined.md"
            if not combined_file.exists():
                error_msg = f"Combined style documentation not found: {combined_file}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "result": ""}

            with open(combined_file, "r", encoding="utf-8") as f:
                content = f.read()

            result_data = {
                "type": "combined",
                "content": content,
                "sections_included": list(available_sections.keys()),
                "total_sections": len(available_sections),
                "size": len(content),
                "description": "Complete OmniUI style documentation covering all styling aspects",
            }

        # Handle list input
        elif isinstance(sections, list):
            if len(sections) == 0:
                return {"success": False, "error": "sections array cannot be empty", "result": ""}

            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(sections) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"sections contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }

            if len(sections) == 1:
                # Single section requested
                section = sections[0]
                logger.info(f"Retrieving style documentation for section: {section}")

                if section not in available_sections:
                    error_msg = f"Unknown section: {section}. Available sections: {list(available_sections.keys())}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "result": ""}

                file_path = data_dir / available_sections[section]
                if not file_path.exists():
                    error_msg = f"Section file not found: {file_path}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "result": ""}

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                result_data = {
                    "section": section,
                    "content": content,
                    "file": available_sections[section],
                    "size": len(content),
                }
            else:
                # Multiple sections requested
                logger.info(f"Retrieving style documentation for sections: {sections}")

                for section_name in sections:
                    if section_name not in available_sections:
                        logger.warning(f"Unknown section requested: {section_name}")
                        continue

                    file_path = data_dir / available_sections[section_name]
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        result_data[section_name] = {
                            "content": content,
                            "file": available_sections[section_name],
                            "size": len(content),
                        }

                if not result_data:
                    error_msg = f"No valid sections found from: {sections}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "result": ""}

        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(sections, str):
            if not sections.strip():
                return {"success": False, "error": "section cannot be empty or whitespace", "result": ""}
            sections = [sections]

            # Process as single section
            section = sections[0]
            logger.info(f"Retrieving style documentation for section: {section}")

            if section not in available_sections:
                error_msg = f"Unknown section: {section}. Available sections: {list(available_sections.keys())}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "result": ""}

            file_path = data_dir / available_sections[section]
            if not file_path.exists():
                error_msg = f"Section file not found: {file_path}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "result": ""}

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            result_data = {
                "section": section,
                "content": content,
                "file": available_sections[section],
                "size": len(content),
            }
        else:
            actual_type = type(sections).__name__
            return {
                "success": False,
                "error": f"sections must be None or a list of strings, but got {actual_type}: {sections}",
                "result": "",
            }

        # Add metadata
        result_data["metadata"] = {
            "available_sections": list(available_sections.keys()),
            "source": "OmniKit UI Style Documentation v1.0.9",
            "description": "Comprehensive styling guidelines for OmniUI widgets and components",
        }

        result_json = json.dumps(result_data, indent=2)

        logger.info(f"Successfully retrieved style documentation")
        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving style documentation: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_style_docs",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
