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

"""Fuzzy matching utilities for USD RAG MCP server."""

import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple


def normalize_name(name: str) -> str:
    """Normalize a name for fuzzy matching by removing dots and converting to lowercase.

    Args:
        name: The name to normalize

    Returns:
        Normalized name with dots removed and lowercased
    """
    if not name:
        return ""

    # Remove dots to allow matching across module boundaries
    # e.g., "USDGeomPrimVar" matches "pxr.UsdGeom.PrimVar"
    return name.replace(".", "").lower()


def _extract_usd_components(query: str) -> Tuple[str, str]:
    """Extract USD module and class components from a query.

    Args:
        query: Query string like "UsdGeomMesh"

    Returns:
        Tuple of (module_part, class_part) - e.g., ("geom", "mesh")
    """
    if not query.lower().startswith("usd") or len(query) <= 3:
        return "", ""

    suffix = query[3:]
    # For patterns like "UsdGeomMesh", try to split into module and class parts
    # This is heuristic-based since USD naming isn't perfectly consistent

    # Common USD module patterns
    common_modules = ["geom", "shade", "skel", "media", "render", "vol", "physics"]

    for module in common_modules:
        if suffix.lower().startswith(module):
            module_part = module
            class_part = suffix[len(module) :].lower()
            return module_part, class_part

    # If no common module found, treat entire suffix as class name
    return "", suffix.lower()


def _check_usd_pattern_match(query: str, target_name: str, target_full_name: str) -> Tuple[bool, float]:
    """Check USD naming pattern matches like "UsdGeomMesh" -> "Mesh" from "pxr.UsdGeom".

    Args:
        query: The search query
        target_name: The class name (e.g., "Mesh")
        target_full_name: The full class name (e.g., "pxr.UsdGeom.Mesh")

    Returns:
        Tuple of (is_match, score)
    """
    if not target_full_name or "pxr.usd" not in target_full_name.lower():
        return False, 0.0

    module_part, class_part = _extract_usd_components(query)

    # Check if target name matches the class part
    if class_part and class_part == target_name.lower():
        # Check if module part matches the full name
        if module_part:
            expected_module = f"pxr.usd{module_part}"
            if expected_module in target_full_name.lower():
                return True, 0.85
        return True, 0.9

    # Check direct suffix match (e.g., "UsdStage" -> "Stage")
    query_suffix = query[3:].lower()
    if query_suffix == target_name.lower():
        return True, 0.9

    return False, 0.0


def is_usd_class_match(query: str, target_name: str, target_full_name: str = "") -> Tuple[bool, float]:
    """Check if query matches a USD class using USD naming conventions.

    USD classes often have the pattern:
    - Query: "UsdStage" -> Target: "Stage" (from pxr.Usd module)
    - Query: "UsdGeomMesh" -> Target: "Mesh" (from pxr.UsdGeom module)
    - Query: "Stage" -> Target: "Stage" (exact match)

    Args:
        query: The search query
        target_name: The class name (e.g., "Stage")
        target_full_name: The full class name (e.g., "pxr.Usd.Stage")

    Returns:
        Tuple of (is_match, score)
    """
    if not query or not target_name:
        return False, 0.0

    query_lower = query.lower()
    target_lower = target_name.lower()
    full_name_lower = target_full_name.lower() if target_full_name else ""

    # Exact matches get highest scores
    if query_lower == target_lower:
        return True, 1.0

    if query_lower == full_name_lower:
        return True, 0.95

    # Check USD naming patterns for queries starting with "Usd"
    if query_lower.startswith("usd") and len(query) > 3:
        is_match, score = _check_usd_pattern_match(query, target_name, target_full_name)
        if is_match:
            return True, score

        # Check if query without "Usd" prefix matches target
        query_without_usd = query[3:].lower()
        if query_without_usd == target_lower:
            return True, 0.8

    return False, 0.0


def _calculate_substring_score(query: str, target: str, is_target_in_query: bool) -> float:
    """Calculate score for substring matches.

    Args:
        query: The query string
        target: The target string
        is_target_in_query: True if target is contained in query, False if query in target

    Returns:
        Match score between 0.0 and 1.0
    """
    if is_target_in_query:
        # Target name is contained in query (e.g., "Stage" in "UsdStage")
        # Give higher score if target is a significant portion of the query
        ratio = len(target) / len(query)
        return 0.75 if ratio >= 0.5 else 0.65
    else:
        # Query is contained in target (lower priority to avoid utility class matches)
        ratio = len(query) / len(target)
        return 0.7 if ratio >= 0.7 else 0.5


def fuzzy_match_score(query: str, target: str, target_full_name: str = "") -> float:
    """Calculate fuzzy match score between query and target strings.

    Args:
        query: The query string
        target: The target string to match against
        target_full_name: The full name of the target (for USD-aware matching)

    Returns:
        Match score between 0.0 and 1.0
    """
    if not query or not target:
        return 0.0

    # First try USD-aware class matching
    is_usd_match, usd_score = is_usd_class_match(query, target, target_full_name)
    if is_usd_match:
        return usd_score

    query_lower = query.lower()
    target_lower = target.lower()

    # Exact matches
    if query_lower == target_lower:
        return 1.0

    if target_full_name and query_lower == target_full_name.lower():
        return 0.95

    # Check normalized versions (dots removed)
    norm_query = normalize_name(query)
    norm_target = normalize_name(target)
    norm_full_name = normalize_name(target_full_name) if target_full_name else ""

    # Exact match after normalization
    if norm_query == norm_target:
        return 0.85

    if norm_full_name and norm_query == norm_full_name:
        return 0.82

    # Substring matches with original strings
    if target_lower in query_lower and len(target_lower) >= 3:
        return _calculate_substring_score(query_lower, target_lower, True)

    if query_lower in target_lower:
        return _calculate_substring_score(query_lower, target_lower, False)

    # Substring matches with normalized strings
    if norm_query and norm_target:
        if norm_target in norm_query and len(norm_target) >= 3:
            ratio = len(norm_target) / len(norm_query)
            return 0.6 if ratio >= 0.5 else 0.45

        if norm_query in norm_target:
            ratio = len(norm_query) / len(norm_target)
            return 0.55 if ratio >= 0.7 else 0.4

    # Use sequence matcher for similarity on normalized strings
    if norm_query and norm_target:
        similarity = SequenceMatcher(None, norm_query, norm_target).ratio()
        if similarity > 0.7:  # Only return good similarities
            return similarity * 0.6  # Scale down but keep reasonable scores

    return 0.0  # No good match found


def find_best_matches(
    query: str,
    candidates: Dict[str, Any],
    key_func: Optional[Callable[[str, Any], List[str]]] = None,
    threshold: float = 0.3,
) -> List[Tuple[str, Any, float]]:
    """Find best matching candidates using fuzzy matching.

    Args:
        query: The search query
        candidates: Dictionary of candidates to search through
        key_func: Function to extract search terms from each candidate.
                 If None, uses default extraction logic.
        threshold: Minimum match score threshold (default 0.3 for more permissive matching)

    Returns:
        List of tuples (key, value, score) sorted by score descending
    """
    if key_func is None:
        key_func = lambda k, v: [k, v.get("name", ""), v.get("full_name", "")]

    matches = []
    for key, value in candidates.items():
        search_terms = key_func(key, value)
        max_score = 0.0

        # Get full name for USD-aware matching
        full_name = value.get("full_name", "") if isinstance(value, dict) else ""

        for term in search_terms:
            if term:
                score = fuzzy_match_score(query, term, full_name)
                max_score = max(max_score, score)

        if max_score >= threshold:
            matches.append((key, value, max_score))

    # Sort by score descending
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches
