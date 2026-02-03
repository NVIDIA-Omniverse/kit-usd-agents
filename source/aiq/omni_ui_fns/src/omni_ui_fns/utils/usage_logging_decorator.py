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

"""Usage logging decorator for OmniUI tools."""

import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict

from .usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def log_tool_usage(tool_name: str):
    """Decorator to log tool usage with timing and error handling.

    Args:
        tool_name: Name of the tool being logged

    Returns:
        Decorated function that logs usage information
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            usage_logger = get_usage_logger()

            if not usage_logger or not usage_logger.enabled:
                # If logging is disabled, just execute the function
                return await func(*args, **kwargs)

            start_time = time.time()
            parameters = {}
            error_msg = None
            success = True

            try:
                # Extract parameters from function arguments
                # For most AIQ functions, the first arg is the input model
                if args and hasattr(args[0], "__dict__"):
                    try:
                        # Convert pydantic model to dict for logging
                        parameters = args[0].dict() if hasattr(args[0], "dict") else {}
                    except Exception:
                        # If conversion fails, just use string representation
                        parameters = {"input": str(args[0])}

                # Execute the original function
                result = await func(*args, **kwargs)

                return result

            except Exception as e:
                success = False
                error_msg = str(e)
                logger.error(f"Error in {tool_name}: {error_msg}")
                raise

            finally:
                # Log the usage regardless of success or failure
                execution_time = time.time() - start_time

                try:
                    usage_logger.log_tool_call(
                        tool_name=tool_name,
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    # Don't let logging errors break the main functionality
                    logger.warning(f"Failed to log usage for {tool_name}: {log_error}")

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            usage_logger = get_usage_logger()

            if not usage_logger or not usage_logger.enabled:
                # If logging is disabled, just execute the function
                return func(*args, **kwargs)

            start_time = time.time()
            parameters = {}
            error_msg = None
            success = True

            try:
                # Extract parameters from function arguments
                if args and hasattr(args[0], "__dict__"):
                    try:
                        parameters = args[0].dict() if hasattr(args[0], "dict") else {}
                    except Exception:
                        parameters = {"input": str(args[0])}

                # Execute the original function
                result = func(*args, **kwargs)

                return result

            except Exception as e:
                success = False
                error_msg = str(e)
                logger.error(f"Error in {tool_name}: {error_msg}")
                raise

            finally:
                # Log the usage regardless of success or failure
                execution_time = time.time() - start_time

                try:
                    usage_logger.log_tool_call(
                        tool_name=tool_name,
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for {tool_name}: {log_error}")

        # Return the appropriate wrapper based on whether the function is async
        if hasattr(func, "_is_coroutine") or func.__name__.startswith("async_"):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
