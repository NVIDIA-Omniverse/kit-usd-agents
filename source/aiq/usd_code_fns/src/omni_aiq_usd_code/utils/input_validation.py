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

"""Input validation utilities for the USD Code MCP server."""

import os
from typing import List, Optional, Union

# Configurable limits via environment variables
MAX_STRING_LENGTH = int(os.getenv("USD_MCP_MAX_STRING_LENGTH", "10000"))
MAX_LIST_ITEMS = int(os.getenv("USD_MCP_MAX_LIST_ITEMS", "100"))
MAX_QUERY_LENGTH = int(os.getenv("USD_MCP_MAX_QUERY_LENGTH", "5000"))


class InputValidationError(ValueError):
    """Exception raised when input validation fails."""

    pass


def validate_string_length(
    value: str,
    field_name: str = "input",
    max_length: int = MAX_STRING_LENGTH,
) -> str:
    """Validate that a string is within the allowed length.

    Args:
        value: The string to validate
        field_name: Name of the field for error messages
        max_length: Maximum allowed length

    Returns:
        The validated string

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise InputValidationError(f"{field_name} must be a string, got {type(value).__name__}")

    if len(value) > max_length:
        raise InputValidationError(f"{field_name} exceeds maximum length of {max_length} characters (got {len(value)})")

    return value


def validate_query(
    query: str,
    field_name: str = "query",
    max_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Validate a search query string.

    Args:
        query: The query string to validate
        field_name: Name of the field for error messages
        max_length: Maximum allowed length for queries

    Returns:
        The validated and stripped query

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(query, str):
        raise InputValidationError(f"{field_name} must be a string, got {type(query).__name__}")

    query = query.strip()

    if not query:
        raise InputValidationError(f"{field_name} cannot be empty")

    if len(query) > max_length:
        raise InputValidationError(f"{field_name} exceeds maximum length of {max_length} characters (got {len(query)})")

    return query


def validate_list_items(
    items: Union[str, List[str]],
    field_name: str = "items",
    max_items: int = MAX_LIST_ITEMS,
    max_item_length: int = MAX_STRING_LENGTH,
) -> List[str]:
    """Validate a list of items or comma-separated string.

    Args:
        items: Either a list of strings or a comma-separated string
        field_name: Name of the field for error messages
        max_items: Maximum number of items allowed
        max_item_length: Maximum length of each item

    Returns:
        Validated list of items

    Raises:
        InputValidationError: If validation fails
    """
    # Handle comma-separated string
    if isinstance(items, str):
        item_list = [item.strip() for item in items.split(",") if item.strip()]
    elif isinstance(items, list):
        item_list = items
    else:
        raise InputValidationError(f"{field_name} must be a string or list, got {type(items).__name__}")

    if not item_list:
        raise InputValidationError(f"{field_name} cannot be empty")

    if len(item_list) > max_items:
        raise InputValidationError(f"{field_name} exceeds maximum of {max_items} items (got {len(item_list)})")

    # Validate each item
    for i, item in enumerate(item_list):
        if not isinstance(item, str):
            raise InputValidationError(f"{field_name}[{i}] must be a string, got {type(item).__name__}")
        if len(item) > max_item_length:
            raise InputValidationError(f"{field_name}[{i}] exceeds maximum length of {max_item_length} characters")

    return item_list


def validate_integer_range(
    value: int,
    field_name: str = "value",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Validate that an integer is within an allowed range.

    Args:
        value: The integer to validate
        field_name: Name of the field for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        The validated integer

    Raises:
        InputValidationError: If validation fails
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputValidationError(f"{field_name} must be an integer, got {type(value).__name__}")

    if min_value is not None and value < min_value:
        raise InputValidationError(f"{field_name} must be at least {min_value} (got {value})")

    if max_value is not None and value > max_value:
        raise InputValidationError(f"{field_name} must be at most {max_value} (got {value})")

    return value
