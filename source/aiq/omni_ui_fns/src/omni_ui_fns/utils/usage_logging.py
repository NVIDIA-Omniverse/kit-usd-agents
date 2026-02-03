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

"""Usage logging utilities for OmniUI tools."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global usage logger instance
_usage_logger: Optional["UsageLogger"] = None


class UsageLogger:
    """Simple usage logger for tracking tool usage."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.usage")

        if enabled:
            # Configure logging format for usage tracking
            formatter = logging.Formatter("%(asctime)s - USAGE - %(levelname)s - %(message)s")

            # Create console handler if not exists
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                self.logger.setLevel(logging.INFO)

    def log_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any] = None,
        success: bool = True,
        error_msg: str = None,
        execution_time: float = None,
    ):
        """Log a tool call with parameters and results."""
        if not self.enabled:
            return

        log_entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "parameters": parameters or {},
            "success": success,
            "execution_time_ms": execution_time * 1000 if execution_time else None,
        }

        if error_msg:
            log_entry["error"] = error_msg

        try:
            self.logger.info(json.dumps(log_entry))
        except Exception as e:
            # Don't let logging failures break the main functionality
            logger.warning(f"Failed to log usage data: {e}")


def create_usage_logger(enabled: bool = True) -> UsageLogger:
    """Create or get the global usage logger instance."""
    global _usage_logger
    if _usage_logger is None:
        _usage_logger = UsageLogger(enabled=enabled)
    return _usage_logger


def get_usage_logger() -> Optional[UsageLogger]:
    """Get the current usage logger instance."""
    return _usage_logger
