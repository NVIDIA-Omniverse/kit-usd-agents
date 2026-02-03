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

"""Rate limiting utilities for the USD Code MCP server.

Provides simple token bucket rate limiting to prevent API abuse and excessive costs.
Rate limiting at this level is supplementary to upstream service limits.
"""

import logging
import os
import threading
import time
from functools import wraps
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Configurable rate limits via environment variables
# Default: 60 requests per minute (1 per second average)
DEFAULT_RATE_LIMIT = int(os.getenv("USD_MCP_RATE_LIMIT", "60"))
DEFAULT_RATE_WINDOW = int(os.getenv("USD_MCP_RATE_WINDOW", "60"))  # seconds

# Feature flag to enable/disable rate limiting (disabled by default per feedback)
RATE_LIMITING_ENABLED = os.getenv("USD_MCP_RATE_LIMITING_ENABLED", "false").lower() in ("true", "1", "yes")


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.1f} seconds.")


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling request rates.

    The token bucket algorithm allows for burst traffic while maintaining
    an average rate limit over time.
    """

    def __init__(
        self,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        window_seconds: int = DEFAULT_RATE_WINDOW,
    ):
        """Initialize the rate limiter.

        Args:
            rate_limit: Maximum number of requests allowed per window
            window_seconds: Time window in seconds
        """
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.tokens = float(rate_limit)
        self.last_update = time.time()
        self._lock = threading.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_update
        tokens_to_add = elapsed * (self.rate_limit / self.window_seconds)
        self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
        self.last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if rate limited
        """
        if not RATE_LIMITING_ENABLED:
            return True

        with self._lock:
            self._refill_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_retry_after(self) -> float:
        """Get the time until tokens are available.

        Returns:
            Seconds until the next token is available
        """
        with self._lock:
            if self.tokens >= 1:
                return 0.0
            tokens_needed = 1 - self.tokens
            seconds_per_token = self.window_seconds / self.rate_limit
            return tokens_needed * seconds_per_token


# Global rate limiter instance
_global_rate_limiter: Optional[TokenBucketRateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter() -> TokenBucketRateLimiter:
    """Get the global rate limiter instance.

    Uses lazy initialization with thread safety.
    """
    global _global_rate_limiter

    if _global_rate_limiter is None:
        with _rate_limiter_lock:
            if _global_rate_limiter is None:
                _global_rate_limiter = TokenBucketRateLimiter()
                if RATE_LIMITING_ENABLED:
                    logger.info(
                        f"Rate limiter initialized: {DEFAULT_RATE_LIMIT} requests per {DEFAULT_RATE_WINDOW} seconds"
                    )
                else:
                    logger.info("Rate limiting is disabled")

    return _global_rate_limiter


def rate_limit(func: Callable) -> Callable:
    """Decorator to apply rate limiting to a function.

    If rate limit is exceeded, raises RateLimitExceeded with retry_after.

    Usage:
        @rate_limit
        async def my_function():
            ...
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        limiter = get_rate_limiter()
        if not limiter.acquire():
            retry_after = limiter.get_retry_after()
            logger.warning(f"Rate limit exceeded for {func.__name__}. Retry after {retry_after:.1f}s")
            raise RateLimitExceeded(retry_after)
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        limiter = get_rate_limiter()
        if not limiter.acquire():
            retry_after = limiter.get_retry_after()
            logger.warning(f"Rate limit exceeded for {func.__name__}. Retry after {retry_after:.1f}s")
            raise RateLimitExceeded(retry_after)
        return func(*args, **kwargs)

    # Return appropriate wrapper based on function type
    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def check_rate_limit() -> Dict[str, any]:
    """Check current rate limit status.

    Returns:
        Dictionary with rate limit status information
    """
    limiter = get_rate_limiter()
    return {
        "enabled": RATE_LIMITING_ENABLED,
        "rate_limit": limiter.rate_limit,
        "window_seconds": limiter.window_seconds,
        "tokens_available": limiter.tokens if RATE_LIMITING_ENABLED else float("inf"),
        "retry_after": limiter.get_retry_after() if RATE_LIMITING_ENABLED else 0.0,
    }
