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

"""Registration wrapper for get_extension_dependencies function."""

import logging

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_extension_dependencies import get_extension_dependencies
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


class GetKitExtensionDependenciesInput(BaseModel):
    """Input for get_extension_dependencies function."""

    extension_id: str = Field(description="Extension ID to analyze dependencies for")
    depth: int = Field(default=2, description="Dependency tree depth to explore")
    include_optional: bool = Field(default=False, description="Include optional dependencies")


# Tool description
GET_KIT_EXTENSION_DEPENDENCIES_DESCRIPTION = """Analyze and visualize Kit extension dependency graphs.

WHAT IT DOES:
- Analyzes dependency tree for specified extension
- Provides hierarchical dependency information
- Shows required and optional dependencies
- Identifies potential dependency conflicts
- Returns version requirements and compatibility info

ANALYSIS FEATURES:
- Recursive dependency traversal
- Configurable depth exploration
- Optional dependency inclusion
- Dependency conflict detection
- Version requirement analysis

ARGUMENTS:
- extension_id (str): Extension ID to analyze (e.g., "omni.ui", "omni.kit.window.console")
- depth (int, optional): Tree depth to explore (default: 2, max recommended: 5)
- include_optional (bool, optional): Include optional dependencies (default: false)

RETURNS:
Dependency tree information including:
- Extension name and version
- Required dependencies at each level
- Optional dependencies (if requested)
- Dependency hierarchy structure
- Version requirements
- Potential conflicts or circular dependencies

USAGE EXAMPLES:
get_extension_dependencies("omni.ui")
get_extension_dependencies("omni.kit.window.console", depth=3)
get_extension_dependencies("omni.ui.scene", include_optional=true)

TIPS:
- Use depth=1 for immediate dependencies only
- Use depth=3+ for complete dependency analysis
- Enable include_optional for comprehensive dependency mapping
- Monitor for circular dependencies in complex extension trees"""


class GetKitExtensionDependenciesConfig(FunctionBaseConfig, name="get_kit_extension_dependencies"):
    """Configuration for get_extension_dependencies function."""

    name: str = "get_kit_extension_dependencies"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetKitExtensionDependenciesConfig, framework_wrappers=[])
async def register_get_kit_extension_dependencies(config: GetKitExtensionDependenciesConfig, builder: Builder):
    """Register get_extension_dependencies function with AIQ."""

    verbose = config.verbose

    if verbose:
        logger.info(f"Registering get_kit_extension_dependencies in verbose mode")

    async def get_kit_extension_dependencies_wrapper(input: GetKitExtensionDependenciesInput) -> str:
        """Wrapper for get_extension_dependencies function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {
            "extension_id": input.extension_id,
            "depth": input.depth,
            "include_optional": input.include_optional,
        }
        error_msg = None
        success = True

        try:
            result = await get_extension_dependencies(
                extension_id=input.extension_id, depth=input.depth, include_optional=input.include_optional
            )

            if verbose:
                logger.debug(f"Analyzed dependencies for extension: {input.extension_id}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to analyze extension dependencies - {error_msg}"
        finally:
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_kit_extension_dependencies",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_extension_dependencies: {log_error}")

    function_info = FunctionInfo.from_fn(
        get_kit_extension_dependencies_wrapper,
        description=GET_KIT_EXTENSION_DEPENDENCIES_DESCRIPTION,
        input_schema=GetKitExtensionDependenciesInput,
    )

    function_info.metadata = {"mcp_exposed": True}

    yield function_info
