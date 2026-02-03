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

"""Patching utilities for the USD RAG MCP server."""

import logging

logger = logging.getLogger(__name__)


def patch_information(s: str) -> str:
    """Apply patches to information strings before returning to users.

    This function applies various string replacements to normalize terminology
    and fix inconsistencies in USD documentation and code examples.

    Args:
        s: The string to patch

    Returns:
        The patched string with replacements applied
    """
    if not isinstance(s, str):
        logger.warning(f"patch_information received non-string input: {type(s)}")
        return s

    try:
        # Apply the specified replacements
        patched = s.replace("UsdPrimCompositionQueryArc", "CompositionArc")
        patched = patched.replace("PrimCompositionQueryArc", "CompositionArc")

        logger.debug("Applied information patches")
        return patched

    except Exception as e:
        logger.error(f"Error applying patches to information: {e}")
        return s  # Return original string if patching fails
