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

"""Function to retrieve Kit application examples and templates."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import KIT_VERSION
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)


def load_app_templates() -> Dict[str, Any]:
    """Load the app templates metadata from JSON file."""
    try:
        templates_file = Path(__file__).parent.parent / "data" / KIT_VERSION / "app_templates" / "app_templates.json"
        with open(templates_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load app templates: {e}")
        return {}


def load_template_file(template_id: str, filename: str) -> Optional[str]:
    """
    Load a specific file for a template.

    Args:
        template_id: Template identifier
        filename: File to load (e.g., 'README.md', 'kit_base_editor.kit')

    Returns:
        File content or None if not found
    """
    try:
        file_path = Path(__file__).parent.parent / "data" / KIT_VERSION / "app_templates" / template_id / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Failed to load {filename} for {template_id}: {e}")
    return None


async def list_app_examples() -> Dict[str, Any]:
    """
    List all available Kit application examples.

    Returns:
        Dictionary with available templates and their descriptions
    """
    try:
        templates = load_app_templates()

        if not templates:
            return {"success": False, "error": "No app templates available", "result": None}

        result_lines = ["# Available Kit Application Templates\n"]

        # Group by category
        categories = {}
        for template_id, template_data in templates.items():
            category = template_data.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append((template_id, template_data))

        # Sort categories for consistent output
        for category in sorted(categories.keys()):
            result_lines.append(f"\n## {category.upper()} Applications\n")

            for template_id, template_data in categories[category]:
                result_lines.append(f"### {template_data.get('name', 'Unknown')}")
                result_lines.append(f"**ID**: `{template_id}`")
                result_lines.append(f"**Description**: {template_data.get('description', 'No description')}")

                # Show key use cases
                use_cases = template_data.get("use_cases", [])
                if use_cases:
                    result_lines.append("\n**Primary Use Cases**:")
                    for use_case in use_cases[:3]:
                        result_lines.append(f"- {use_case}")

                # Show metadata
                if template_data.get("supports_streaming"):
                    result_lines.append("**Streaming**: ✓ Supported")
                if template_data.get("dependencies_count"):
                    result_lines.append(f"**Extensions**: {template_data['dependencies_count']} dependencies")

                result_lines.append("")  # Empty line between templates

        result_lines.append("\n## How to Use")
        result_lines.append("- **Search**: Use `search_app_examples(query)` to find templates by use case")
        result_lines.append("- **Get Details**: Use `get_app_examples(template_ids)` to retrieve full template details")
        result_lines.append(
            "- **Example**: `get_app_examples('kit_base_editor')` or `get_app_examples(['kit_base_editor', 'usd_viewer'])`"
        )

        return {"success": True, "result": "\n".join(result_lines), "templates": list(templates.keys())}

    except Exception as e:
        logger.error(f"Failed to list app examples: {e}")
        return {"success": False, "error": f"Failed to list templates: {str(e)}", "result": None}


async def get_app_examples(template_ids: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Retrieve Kit application example templates with full details.

    Args:
        template_ids: Template IDs to retrieve. Can be:
            - None: List all available templates
            - String: Single template ID
            - List: Multiple template IDs

    Returns:
        Dictionary with:
        - success: Whether retrieval was successful
        - result: Formatted template information including README and kit files
        - error: Error message if failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {"template_ids": template_ids}

    success = True
    error_msg = None

    try:
        # Handle None input - list all templates
        if template_ids is None:
            return await list_app_examples()

        # Load templates metadata
        templates = load_app_templates()

        if not templates:
            return {"success": False, "error": "No app templates available", "result": None}

        # Normalize input to list
        if isinstance(template_ids, str):
            template_list = [template_ids]
        elif isinstance(template_ids, list):
            template_list = template_ids
        else:
            return {
                "success": False,
                "error": f"Invalid input type: {type(template_ids).__name__}. Expected string or list.",
                "result": None,
            }

        # Validate template IDs
        invalid_ids = [tid for tid in template_list if tid not in templates]
        if invalid_ids:
            available = list(templates.keys())
            return {
                "success": False,
                "error": f"Unknown template IDs: {', '.join(invalid_ids)}. Available: {', '.join(available)}",
                "result": None,
            }

        # Process each template
        results = []
        for template_id in template_list:
            template_data = templates[template_id]

            # Special handling for streaming_configs
            if template_id == "streaming_configs":
                result_content = format_streaming_configs(template_data)
            else:
                result_content = format_template_details(template_id, template_data)

            results.append(result_content)

        # Combine results
        if len(results) == 1:
            final_result = results[0]
        else:
            # Multiple templates - combine with separators
            combined = []
            combined.append(f"# Kit Application Templates ({len(results)} templates)\n")
            for i, result in enumerate(results, 1):
                if i > 1:
                    combined.append(f"\n{'='*80}\n")
                combined.append(result)
            final_result = "\n".join(combined)

        logger.info(f"Retrieved {len(template_list)} app template(s)")

        return {"success": True, "result": final_result, "templates_retrieved": template_list}

    except Exception as e:
        logger.error(f"Failed to get app examples: {e}")
        error_msg = str(e)
        success = False
        return {"success": False, "error": f"Failed to retrieve templates: {str(e)}", "result": None}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_app_examples",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )


def format_template_details(template_id: str, template_data: Dict[str, Any]) -> str:
    """Format detailed template information including files."""
    lines = []

    # Header
    lines.append(f"# {template_data.get('name', 'Unknown Template')}")
    lines.append(f"**Template ID**: `{template_id}`")
    lines.append(f"**Category**: {template_data.get('category', 'Unknown')}")
    lines.append("")

    # Description
    lines.append("## Description")
    lines.append(template_data.get("description", "No description available"))
    lines.append("")

    # Use Cases
    use_cases = template_data.get("use_cases", [])
    if use_cases:
        lines.append("## Use Cases")
        for use_case in use_cases:
            lines.append(f"- {use_case}")
        lines.append("")

    # Key Features
    features = template_data.get("key_features", [])
    if features:
        lines.append("## Key Features")
        for feature in features:
            lines.append(f"- {feature}")
        lines.append("")

    # Metadata
    lines.append("## Template Metadata")
    lines.append(f"- **Dependencies**: {template_data.get('dependencies_count', 0)} extensions")
    lines.append(f"- **Streaming Support**: {'✓' if template_data.get('supports_streaming') else '✗'}")
    if template_data.get("requires_setup_extension"):
        lines.append("- **Setup Extension**: Required")
    if template_data.get("configuration_layers"):
        lines.append(f"- **Streaming Configs**: {', '.join(template_data['configuration_layers'])}")
    lines.append("")

    # Load README
    readme_file = template_data.get("readme_file", "README.md")
    readme_content = load_template_file(template_id, readme_file)

    if readme_content:
        lines.append("## README Documentation")
        lines.append("```markdown")
        # Truncate if too long
        if len(readme_content) > 8000:
            lines.append(readme_content[:8000])
            lines.append("\n... [README truncated for length]")
        else:
            lines.append(readme_content)
        lines.append("```")
        lines.append("")

    # Load Kit file
    kit_file = template_data.get("kit_file")
    if kit_file:
        kit_content = load_template_file(template_id, kit_file)

        if kit_content:
            lines.append(f"## Kit Configuration File: {kit_file}")
            lines.append("```toml")
            # Show important sections of kit file
            if len(kit_content) > 6000:
                # Extract key sections for long files
                lines.append("# [Showing key sections of kit file]")
                lines.append("")

                # Extract package section
                package_section = extract_kit_section(kit_content, "[package]", "[dependencies]")
                if package_section:
                    lines.append(package_section)
                    lines.append("")

                # Extract first part of dependencies
                deps_section = extract_kit_section(kit_content, "[dependencies]", "[settings", max_lines=30)
                if deps_section:
                    lines.append("[dependencies]")
                    lines.append(deps_section)
                    lines.append("# ... [dependencies truncated] ...")
                    lines.append("")

                # Extract key settings
                settings_section = extract_kit_section(kit_content, "[settings.app]", None, max_lines=20)
                if settings_section:
                    lines.append("[settings.app]")
                    lines.append(settings_section)
            else:
                lines.append(kit_content)
            lines.append("```")

    return "\n".join(lines)


def format_streaming_configs(template_data: Dict[str, Any]) -> str:
    """Format streaming configuration templates."""
    lines = []

    lines.append(f"# {template_data.get('name', 'Streaming Configurations')}")
    lines.append(f"**Category**: {template_data.get('category', 'configuration')}")
    lines.append("")
    lines.append("## Description")
    lines.append(template_data.get("description", ""))
    lines.append("")

    configs = template_data.get("configurations", {})

    for config_id, config_data in configs.items():
        lines.append(f"## {config_data.get('name', config_id)}")
        lines.append(f"**Description**: {config_data.get('description', '')}")
        lines.append("")

        features = config_data.get("features", [])
        if features:
            lines.append("**Features**:")
            for feature in features:
                lines.append(f"- {feature}")
            lines.append("")

        use_cases = config_data.get("use_cases", [])
        if use_cases:
            lines.append("**Use Cases**:")
            for use_case in use_cases:
                lines.append(f"- {use_case}")
            lines.append("")

        # Load the kit file
        kit_file = config_data.get("kit_file")
        if kit_file:
            kit_content = load_template_file("streaming_configs", kit_file)
            if kit_content:
                lines.append(f"### Configuration: {kit_file}")
                lines.append("```toml")
                if len(kit_content) > 3000:
                    lines.append(kit_content[:3000])
                    lines.append("\n... [truncated for length]")
                else:
                    lines.append(kit_content)
                lines.append("```")
                lines.append("")

    return "\n".join(lines)


def extract_kit_section(
    content: str, start_marker: str, end_marker: Optional[str], max_lines: int = 50
) -> Optional[str]:
    """Extract a section from kit file content."""
    lines = content.split("\n")
    start_idx = None

    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    end_idx = len(lines)
    if end_marker:
        for i in range(start_idx, len(lines)):
            if end_marker in lines[i]:
                end_idx = i
                break

    section_lines = lines[start_idx : min(end_idx, start_idx + max_lines)]
    return "\n".join(section_lines)
