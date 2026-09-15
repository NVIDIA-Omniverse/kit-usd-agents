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

"""Utility modules for the USD RAG MCP server.

``sanitize_query`` previously lived here as a local html-escaping
re-implementation that silently corrupted queries containing
``& < > ' "`` at embed time (audit finding R2). All call sites now
import the canonical version directly from ``ovgenai_retrieval``;
the local sanitizer module has been removed. The re-exports below
(input validation, patching, rate limiting) are unrelated and stay.
"""

from .input_validation import (
    InputValidationError,
    validate_integer_range,
    validate_list_items,
    validate_query,
    validate_string_length,
)
from .patching import patch_information
from .rate_limiting import RateLimitExceeded, check_rate_limit, get_rate_limiter, rate_limit

__all__ = [
    "patch_information",
    "InputValidationError",
    "validate_string_length",
    "validate_query",
    "validate_list_items",
    "validate_integer_range",
    "RateLimitExceeded",
    "rate_limit",
    "get_rate_limiter",
    "check_rate_limit",
]
