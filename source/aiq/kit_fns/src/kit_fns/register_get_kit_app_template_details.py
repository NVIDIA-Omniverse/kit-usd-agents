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

"""Registration wrapper for get_app_examples function."""

import json
import logging
from typing import List, Optional, Union

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_app_examples import get_app_examples
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_template_ids_input(template_ids_str: str) -> Union[List[str], None]:
    """Parse template_ids string input into appropriate format for get_app_examples.

    Args:
        template_ids_str: String input that can be:
            - None/empty: Return None (list all templates)
            - Single template: "kit_base_editor"
            - JSON array: '["kit_base_editor", "usd_viewer"]'
            - Comma-separated: "kit_base_editor, usd_viewer"

    Returns:
        - None: For empty input (list all templates)
        - List[str]: For template IDs

    Raises:
        ValueError: If input format is invalid
    """
    if not template_ids_str or not template_ids_str.strip():
        return None

    template_ids_str = template_ids_str.strip()

    # Try to parse as JSON array first
    if template_ids_str.startswith("[") and template_ids_str.endswith("]"):
        try:
            parsed = json.loads(template_ids_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in template_ids_str:
        return [name.strip() for name in template_ids_str.split(",") if name.strip()]

    # Single template ID
    return [template_ids_str]


class GetKitAppTemplateDetailsInput(BaseModel):
    """Input for get_app_examples function.

    Provide template IDs in any convenient format - the system will handle the conversion automatically.
    """

    template_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Kit application template IDs to retrieve. Accepts multiple flexible formats:
        
        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single template: "kit_base_editor"
        - Native array: ["kit_base_editor", "usd_viewer"] ← WORKS DIRECTLY!
        - JSON string: '["kit_base_editor", "usd_viewer", "usd_composer"]'
        - Comma-separated: "kit_base_editor, usd_viewer, usd_explorer"
        - Empty/null: Lists all available templates with descriptions
        
        📚 AVAILABLE TEMPLATE IDS:
        - "kit_base_editor": Minimal 3D editing application starter
        - "usd_composer": Professional USD content creation suite
        - "usd_explorer": Large-scale environment visualization
        - "usd_viewer": Streaming-optimized RTX viewer
        - "streaming_configs": Streaming configuration layers
        
        💡 TIP: Use search_app_examples first to find the right template!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_KIT_APP_TEMPLATE_DETAILS_DESCRIPTION = """Retrieve complete Kit application template examples including documentation and configuration files.

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- template_ids: Template IDs in ANY convenient format:
  * Single template: "kit_base_editor"
  * Native array: ["kit_base_editor", "usd_viewer"] ← WORKS DIRECTLY!
  * JSON string: '["kit_base_editor", "usd_viewer"]'
  * Comma-separated: "kit_base_editor, usd_viewer"
  * Empty/null: Lists all available templates

AVAILABLE TEMPLATES:
- **kit_base_editor**: Minimal starting point for 3D applications
  - Basic UI, RTX rendering, material library
  - 68 dependencies, streaming support
  
- **usd_composer**: Professional authoring application
  - Full creation suite, animation, materials
  - 147 dependencies, requires setup extension
  
- **usd_explorer**: Large-scale visualization
  - Industrial environments, collaboration
  - 104 dependencies, dual-mode UI
  
- **usd_viewer**: Streaming-optimized viewer
  - Cloud deployment, bi-directional messaging
  - 38 dependencies, headless operation
  
- **streaming_configs**: Configuration layers
  - Default, GDN, and NVCF streaming setups

RETURNS:
- Complete README documentation
- Full .kit configuration files
- Detailed metadata and features
- Use cases and implementation guidance
- Dependency information

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_app_examples(template_ids=["kit_base_editor", "usd_viewer"])
✅ Single string: get_app_examples(template_ids="kit_base_editor")
✅ JSON string: get_app_examples(template_ids='["kit_base_editor", "usd_composer"]')
✅ Comma format: get_app_examples(template_ids="kit_base_editor, usd_viewer, usd_explorer")
✅ List all: get_app_examples() or get_app_examples(template_ids=null)

💡 FOR AI MODELS: You can pass arrays directly like ["kit_base_editor", "usd_viewer"] - no conversion needed!

WORKFLOW:
1. Search for templates: search_app_examples("your use case")
2. Get full details: get_app_examples("template_id")
3. Review README and .kit configuration
4. Use as starting point for your application

BATCH PROCESSING BENEFITS:
- Single API call for multiple templates
- Combined documentation with clear sections
- Efficient context window usage
- Side-by-side comparison capability

The templates provide production-ready starting points for:
- Desktop applications
- Cloud streaming services
- Industrial visualization
- Content creation tools
- Collaborative platforms"""


class GetKitAppTemplateDetailsConfig(FunctionBaseConfig, name="get_kit_app_template_details"):
    """Configuration for get_app_examples function."""

    name: str = "get_kit_app_template_details"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetKitAppTemplateDetailsConfig, framework_wrappers=[])
async def register_get_kit_app_template_details(config: GetKitAppTemplateDetailsConfig, builder: Builder):
    """Register get_app_examples function with AIQ."""

    verbose = config.verbose

    if verbose:
        logger.info("Registering get_kit_app_template_details in verbose mode")

    async def get_kit_app_template_details_wrapper(input: GetKitAppTemplateDetailsInput) -> str:
        """Get Kit application template examples."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.template_ids is None:
                template_ids_to_fetch = None
            elif isinstance(input.template_ids, list):
                # Direct array input - validate and use as-is
                if len(input.template_ids) == 0:
                    template_ids_to_fetch = None  # Empty array = list all
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.template_ids):
                        if not isinstance(item, str):
                            return f"ERROR: All items in template_ids array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in template_ids array"
                    template_ids_to_fetch = [item.strip() for item in input.template_ids]
            elif isinstance(input.template_ids, str):
                # String input - parse using existing logic
                template_ids_to_fetch = _parse_template_ids_input(input.template_ids)
            else:
                return f"ERROR: template_ids must be a string, array, or null, got {type(input.template_ids).__name__}"

            parameters = {"template_ids": input.template_ids}
        except ValueError as e:
            return f"ERROR: Invalid template_ids parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_app_examples(template_ids_to_fetch)

            if verbose:
                if template_ids_to_fetch is None:
                    logger.debug("Listed all available app templates")
                elif isinstance(template_ids_to_fetch, list):
                    logger.debug(f"Retrieved {len(template_ids_to_fetch)} app template(s)")
                else:
                    logger.debug(f"Retrieved template: {template_ids_to_fetch}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve app examples - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_kit_app_template_details",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_app_examples: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_kit_app_template_details_wrapper,
        description=GET_KIT_APP_TEMPLATE_DETAILS_DESCRIPTION,
        input_schema=GetKitAppTemplateDetailsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info
