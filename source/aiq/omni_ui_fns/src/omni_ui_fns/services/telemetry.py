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
Centralized telemetry service for OmniUI MCP using Redis Streams.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

logger = logging.getLogger(__name__)


class TelemetryService:
    """
    Centralized telemetry service using Redis Streams.

    Captures function calls with timing, parameters, and success status.
    """

    _instance = None
    _redis_client = None
    _enabled = True

    # Redis configuration - configurable via environment variables
    # Defaults are safe for development; production should override via env vars
    REDIS_HOST = os.getenv("OMNI_UI_MCP_REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("OMNI_UI_MCP_REDIS_PORT", "6379"))
    KEY_PREFIX = os.getenv("OMNI_UI_MCP_TELEMETRY_PREFIX", "omni_ui_mcp:telemetry")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelemetryService, cls).__new__(cls)
        return cls._instance

    async def initialize(self):
        """Initialize Redis connection."""
        # Check if telemetry is disabled via environment variable
        telemetry_enabled = os.getenv("TELEMETRY_ENABLED", "false").lower()
        if telemetry_enabled in ("false", "0", "no", "disabled"):
            logger.info("Telemetry disabled via TELEMETRY_ENABLED environment variable")
            self._enabled = False
            return

        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available, telemetry disabled")
            self._enabled = False
            return

        if self._redis_client is None:
            try:
                self._redis_client = aioredis.Redis(
                    host=self.REDIS_HOST,
                    port=self.REDIS_PORT,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                # Test connection
                await self._redis_client.ping()
                logger.info(f"Telemetry service connected to Redis at {self.REDIS_HOST}:{self.REDIS_PORT}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._enabled = False
                self._redis_client = None

    async def capture_call(
        self,
        function_name: str,
        request_data: Dict[str, Any],
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Capture a function call to Redis as a regular key-value entry.

        Args:
            function_name: Name of the function being called
            request_data: Input parameters passed to the function
            duration_ms: Time taken for the function execution in milliseconds
            success: Whether the function call was successful
            error: Error message if the call failed
            session_id: Optional session identifier for grouping calls

        Returns:
            bool: True if telemetry was captured successfully, False otherwise
        """
        if not self._enabled or self._redis_client is None:
            return False

        try:
            # Generate unique call ID and timestamp
            call_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc)

            # Create Redis key with timestamp for easy sorting
            # Format: omni_ui_mcp:telemetry:YYYY-MM-DD:HH-MM-SS-microseconds:call_id
            timestamp_str = timestamp.strftime("%Y-%m-%d:%H-%M-%S") + f"-{timestamp.microsecond:06d}"
            redis_key = f"{self.KEY_PREFIX}:{timestamp_str}:{call_id}"

            # Prepare telemetry data with anonymized request data
            # This prevents leaking sensitive user queries to telemetry
            anonymized_request = self._anonymize_request_data(request_data)

            telemetry_data = {
                "service": "omni_ui_mcp",
                "function_name": function_name,
                "call_id": call_id,
                "timestamp": timestamp.isoformat(),
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "request_metadata": anonymized_request,
                "session_id": self._hash_session_id(session_id) if session_id else "unknown",
            }

            if error:
                telemetry_data["error"] = error

            # Store as JSON in Redis with expiration (optional - can be removed for permanent storage)
            await self._redis_client.set(
                redis_key,
                json.dumps(telemetry_data, default=str),
                # ex=None  # No expiration for permanent storage
            )

            logger.debug(f"Telemetry captured for {function_name}: {redis_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to capture telemetry: {e}")
            return False

    @asynccontextmanager
    async def track_call(self, function_name: str, request_data: Dict[str, Any], session_id: Optional[str] = None):
        """
        Context manager for tracking function calls with automatic timing.

        Usage:
            async with telemetry.track_call("my_function", {"param": "value"}):
                # Your function logic here
                result = await do_something()
        """
        start_time = time.perf_counter()
        success = True
        error = None

        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            await self.capture_call(
                function_name=function_name,
                request_data=request_data,
                duration_ms=duration_ms,
                success=success,
                error=error,
                session_id=session_id,
            )

    async def get_telemetry_keys_count(self) -> int:
        """Get count of telemetry keys in Redis."""
        if not self._enabled or self._redis_client is None:
            return 0

        try:
            pattern = f"{self.KEY_PREFIX}:*"
            keys = await self._redis_client.keys(pattern)
            return len(keys)
        except Exception as e:
            logger.error(f"Failed to get telemetry keys count: {e}")
            return 0

    async def get_recent_telemetry_keys(self, limit: int = 10) -> list:
        """Get the most recent telemetry keys (sorted by timestamp)."""
        if not self._enabled or self._redis_client is None:
            return []

        try:
            pattern = f"{self.KEY_PREFIX}:*"
            keys = await self._redis_client.keys(pattern)
            # Keys are naturally sorted by timestamp due to our naming convention
            return sorted(keys, reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Failed to get recent telemetry keys: {e}")
            return []

    async def close(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None

    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._enabled and self._redis_client is not None

    def _anonymize_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize request data to prevent leaking sensitive user information.

        Only captures metadata about the request structure, not actual content.

        Args:
            request_data: Original request data

        Returns:
            Anonymized metadata about the request
        """
        if not request_data:
            return {}

        anonymized = {}

        for key, value in request_data.items():
            if isinstance(value, str):
                # Only capture length and type, not actual content
                anonymized[key] = {
                    "type": "string",
                    "length": len(value),
                    "empty": len(value) == 0,
                }
            elif isinstance(value, list):
                anonymized[key] = {
                    "type": "list",
                    "count": len(value),
                    "empty": len(value) == 0,
                }
            elif isinstance(value, dict):
                anonymized[key] = {
                    "type": "dict",
                    "key_count": len(value),
                    "empty": len(value) == 0,
                }
            elif isinstance(value, bool):
                # Booleans are safe to include
                anonymized[key] = {"type": "bool", "value": value}
            elif isinstance(value, (int, float)):
                # Numeric values are generally safe for telemetry
                anonymized[key] = {"type": type(value).__name__, "value": value}
            elif value is None:
                anonymized[key] = {"type": "null"}
            else:
                anonymized[key] = {"type": type(value).__name__}

        return anonymized

    def _hash_session_id(self, session_id: str) -> str:
        """Hash session ID to anonymize it while maintaining correlation ability.

        Args:
            session_id: Original session identifier

        Returns:
            Hashed session ID
        """
        if not session_id or session_id == "unknown":
            return "unknown"
        return hashlib.sha256(session_id.encode()).hexdigest()[:16]


# Global telemetry instance
telemetry = TelemetryService()


async def ensure_telemetry_initialized():
    """Ensure telemetry service is initialized."""
    if telemetry._redis_client is None:
        await telemetry.initialize()
