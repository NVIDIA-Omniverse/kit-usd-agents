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

"""Input sanitization utilities for Kit MCP tools."""

import html
import re
from typing import Optional


def sanitize_query(query: str) -> str:
    """Sanitize user query before sending to external APIs.

    This function removes potentially malicious content like HTML/script tags
    and encodes special characters to prevent XSS and injection attacks.

    Args:
        query: The raw user query string.

    Returns:
        Sanitized query string safe for external API calls.
    """
    if not query:
        return ""

    # Remove HTML/XML tags
    sanitized = re.sub(r"<[^>]+>", "", query)

    # Escape HTML entities for any remaining special characters
    sanitized = html.escape(sanitized)

    # Remove null bytes and other control characters
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sanitized)

    # Normalize whitespace
    sanitized = " ".join(sanitized.split())

    return sanitized.strip()


def sanitize_identifier(identifier: str, max_length: int = 200) -> Optional[str]:
    """Sanitize an identifier (class name, module name, etc.) for safe use.

    Args:
        identifier: The identifier string to sanitize.
        max_length: Maximum allowed length for the identifier.

    Returns:
        Sanitized identifier or None if the input is invalid.
    """
    if not identifier or not isinstance(identifier, str):
        return None

    # Remove any HTML tags
    sanitized = re.sub(r"<[^>]+>", "", identifier)

    # Remove path traversal patterns
    sanitized = re.sub(r"\.\./", "", sanitized)
    sanitized = re.sub(r"\\", "", sanitized)

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized.strip() if sanitized.strip() else None
