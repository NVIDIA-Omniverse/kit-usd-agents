"""Shared input sanitization (REQ-RG-3).

Ported from the per-MCP copies under
``kit-usd-agents-master/source/aiq/*_fns/src/*/utils/input_sanitization.py``.
Behavior is unchanged so MCP test suites keep passing once callers switch
their imports to this module.
"""

from __future__ import annotations

import html
import re
from typing import Optional

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PATH_TRAVERSAL_RE = re.compile(r"\.\./")


def sanitize_query(query: str) -> str:
    """Scrub a raw user query before it reaches an external API or retriever.

    Strips HTML/XML tags, escapes residual HTML entities, removes control
    characters and null bytes, and collapses whitespace.
    """
    if not query:
        return ""
    sanitized = _HTML_TAG_RE.sub("", query)
    sanitized = html.escape(sanitized)
    sanitized = _CONTROL_CHAR_RE.sub("", sanitized)
    sanitized = " ".join(sanitized.split())
    return sanitized.strip()


def sanitize_identifier(identifier: str, max_length: int = 200) -> Optional[str]:
    """Scrub a class / module / extension identifier for safe interpolation.

    Returns ``None`` when the cleaned result is empty or the input is not a
    string. Removes HTML tags, path-traversal sequences, and backslashes; then
    truncates to ``max_length``.
    """
    if not identifier or not isinstance(identifier, str):
        return None
    sanitized = _HTML_TAG_RE.sub("", identifier)
    sanitized = _PATH_TRAVERSAL_RE.sub("", sanitized)
    sanitized = sanitized.replace("\\", "")
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    stripped = sanitized.strip()
    return stripped or None
