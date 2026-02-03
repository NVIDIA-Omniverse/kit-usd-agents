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

"""Usage logging decorator for tracking tool calls."""

import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict

from ..services.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def log_tool_usage(tool_name: str):
    """Decorator to log tool usage.

    Args:
        tool_name (str): Name of the tool being called.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper for logging tool usage."""
            usage_logger = get_usage_logger()
            start_time = time.time()

            # Extract parameters for logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            parameters = dict(bound_args.arguments)

            try:
                # Execute the tool
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log successful usage
                if usage_logger and usage_logger.enabled:
                    try:
                        usage_logger.log_tool_call(
                            tool_name=tool_name,
                            parameters=parameters,
                            success=True,
                            execution_time=execution_time,
                        )
                    except Exception as log_error:
                        logger.warning(f"Failed to log usage for {tool_name}: {log_error}")

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Log failed usage
                if usage_logger and usage_logger.enabled:
                    try:
                        usage_logger.log_tool_call(
                            tool_name=tool_name,
                            parameters=parameters,
                            success=False,
                            error_msg=str(e),
                            execution_time=execution_time,
                        )
                    except Exception as log_error:
                        logger.warning(f"Failed to log usage for {tool_name}: {log_error}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Sync wrapper for logging tool usage."""
            usage_logger = get_usage_logger()
            start_time = time.time()

            # Extract parameters for logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            parameters = dict(bound_args.arguments)

            try:
                # Execute the tool
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log successful usage
                if usage_logger and usage_logger.enabled:
                    try:
                        usage_logger.log_tool_call(
                            tool_name=tool_name,
                            parameters=parameters,
                            success=True,
                            execution_time=execution_time,
                        )
                    except Exception as log_error:
                        logger.warning(f"Failed to log usage for {tool_name}: {log_error}")

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Log failed usage
                if usage_logger and usage_logger.enabled:
                    try:
                        usage_logger.log_tool_call(
                            tool_name=tool_name,
                            parameters=parameters,
                            success=False,
                            error_msg=str(e),
                            execution_time=execution_time,
                        )
                    except Exception as log_error:
                        logger.warning(f"Failed to log usage for {tool_name}: {log_error}")
                raise

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
