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

"""ovgenai-retrieval compatibility shim.

Re-exports the small set of helpers (``sanitize_query``,
``empty_result_sentinel``, ``truncate_text_with_guidance``) used by the
register_search_*.py modules through one import point that works in
two deployment shapes:

1. Docker images — ``ovgenai-retrieval`` is installed from the wheel
   copied into ``source/mcp/<mcp>/dist/`` (see commit V-11). Imports
   resolve to the real library and behavior is identical.

2. Local poetry-only dev — commit afa824f6 deliberately removed
   ``ovgenai-retrieval`` from the poetry pyprojects so a fresh
   ``poetry install`` doesn't need network access. In that environment
   the wheel isn't installed and the bare ``from ovgenai_retrieval ...``
   imports at module load time would crash the MCP register entry
   points. This shim provides equivalent fallbacks so the MCP starts.

Behavior parity: the fallbacks below were ported from the original
``utils/input_sanitization.py`` and ``ovgenai_retrieval.guardrails``
modules. Existing test fixtures continue to pass against either path.
"""

from __future__ import annotations

try:
    from ovgenai_retrieval import sanitize_query
    from ovgenai_retrieval.guardrails import empty_result_sentinel, truncate_text_with_guidance
except ImportError:
    import html
    import re

    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    _EMPTY_RESULT_SENTINEL = "No relevant documentation found for this query."
    _DEFAULT_TRUNCATION_HINT = (
        "[Results truncated to fit the context window. "
        "Refine the query or narrow the --subfolder scope to see more specific matches.]"
    )

    def sanitize_query(query: str) -> str:  # type: ignore[no-redef]
        """Local fallback ported from the pre-R2 input_sanitization.py."""
        if not query:
            return ""
        sanitized = _HTML_TAG_RE.sub("", query)
        sanitized = html.escape(sanitized)
        sanitized = _CONTROL_CHAR_RE.sub("", sanitized)
        sanitized = " ".join(sanitized.split())
        return sanitized.strip()

    def empty_result_sentinel() -> str:  # type: ignore[no-redef]
        """Canonical "no hits" message (REQ-RG-2)."""
        return _EMPTY_RESULT_SENTINEL

    def truncate_text_with_guidance(  # type: ignore[no-redef]
        text: str,
        max_chars: int,
        hint: str = _DEFAULT_TRUNCATION_HINT,
    ) -> tuple[str, str | None]:
        """Cap a pre-formatted result string at ``max_chars`` (REQ-RG-1)."""
        if max_chars <= 0:
            return "", hint
        if len(text) <= max_chars:
            return text, None
        head = text[:max_chars]
        nl = head.rfind("\n")
        if nl >= int(max_chars * 0.8):
            head = head[:nl]
        return head, hint


__all__ = [
    "sanitize_query",
    "empty_result_sentinel",
    "truncate_text_with_guidance",
]
