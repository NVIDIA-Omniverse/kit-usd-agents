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

"""Fuzzy matching utilities for finding best matches in OmniUI Atlas data."""

from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score between 0 and 1
    """
    # Convert to lowercase for case-insensitive comparison
    str1_lower = str1.lower()
    str2_lower = str2.lower()

    # Exact match (case-insensitive)
    if str1_lower == str2_lower:
        return 1.0

    # Check if one is contained in the other
    if str1_lower in str2_lower or str2_lower in str1_lower:
        # Give higher score for containment
        return 0.9 * min(len(str1_lower), len(str2_lower)) / max(len(str1_lower), len(str2_lower))

    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, str1_lower, str2_lower).ratio()


def find_best_matches(
    query: str,
    candidates: Dict[str, Any],
    key_func: Callable[[str], str],
    value_func: Callable[[str], Any],
    name_func: Callable[[Any], str],
    threshold: float = 0.5,
    max_results: int = 5,
) -> List[Tuple[str, Any, float]]:
    """Find best matches for a query in a dictionary of candidates.

    Args:
        query: Search query string
        candidates: Dictionary of candidates to search
        key_func: Function to extract key from candidate
        value_func: Function to extract value from candidate
        name_func: Function to extract name from value for comparison
        threshold: Minimum similarity score to include in results
        max_results: Maximum number of results to return

    Returns:
        List of tuples containing (key, value, similarity_score)
    """
    matches = []

    for key in candidates:
        value = value_func(key)
        name = name_func(value)

        # Calculate similarity with the full name
        similarity = calculate_similarity(query, name)

        # Also check against just the class/module name (last part)
        if "." in name:
            short_name = name.split(".")[-1]
            short_similarity = calculate_similarity(query, short_name)
            similarity = max(similarity, short_similarity)

        # Also check against the key itself
        key_similarity = calculate_similarity(query, key)
        similarity = max(similarity, key_similarity)

        if similarity >= threshold:
            matches.append((key, value, similarity))

    # Sort by similarity score (descending)
    matches.sort(key=lambda x: x[2], reverse=True)

    # Return top results
    return matches[:max_results]


def find_best_match(query: str, candidates: List[str], threshold: float = 0.5) -> Optional[Tuple[str, float]]:
    """Find the best match for a query in a list of candidates.

    Args:
        query: Search query string
        candidates: List of candidate strings
        threshold: Minimum similarity score to consider a match

    Returns:
        Tuple of (best_match, similarity_score) or None if no match found
    """
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        score = calculate_similarity(query, candidate)
        if score > best_score and score >= threshold:
            best_match = candidate
            best_score = score

    if best_match:
        return (best_match, best_score)
    return None
