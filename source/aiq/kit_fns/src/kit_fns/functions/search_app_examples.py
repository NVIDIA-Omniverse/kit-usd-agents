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

"""Function to search for Kit application examples using semantic search."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import KIT_VERSION
from ..services.telemetry import ensure_telemetry_initialized, telemetry

logger = logging.getLogger(__name__)


# Load app templates metadata
def load_app_templates() -> Dict[str, Any]:
    """Load the app templates metadata from JSON file."""
    try:
        templates_file = Path(__file__).parent.parent / "data" / KIT_VERSION / "app_templates" / "app_templates.json"
        with open(templates_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load app templates: {e}")
        return {}


def calculate_relevance_score(query: str, template_data: Dict[str, Any]) -> float:
    """
    Calculate relevance score for a template based on the search query.

    Args:
        query: Search query string
        template_data: Template metadata dictionary

    Returns:
        Relevance score (0.0 to 1.0)
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    score = 0.0

    # Check name match (highest weight)
    if query_lower in template_data.get("name", "").lower():
        score += 0.3

    # Check description match
    description = template_data.get("description", "").lower()
    if query_lower in description:
        score += 0.25
    else:
        # Check individual words
        desc_words = set(description.split())
        word_matches = len(query_words.intersection(desc_words))
        score += min(0.15, word_matches * 0.03)

    # Check category match
    if query_lower in template_data.get("category", "").lower():
        score += 0.15

    # Check use cases
    use_cases = template_data.get("use_cases", [])
    for use_case in use_cases:
        if query_lower in use_case.lower():
            score += 0.15
            break
    else:
        # Check individual words in use cases
        for use_case in use_cases:
            use_case_words = set(use_case.lower().split())
            if query_words.intersection(use_case_words):
                score += 0.08
                break

    # Check key features
    features = template_data.get("key_features", [])
    for feature in features:
        if query_lower in feature.lower():
            score += 0.1
            break
    else:
        # Check individual words in features
        for feature in features:
            feature_words = set(feature.lower().split())
            if query_words.intersection(feature_words):
                score += 0.05
                break

    # Special keywords boost
    special_keywords = {
        "streaming": ["streaming", "cloud", "remote", "webrtc"],
        "large": ["large", "scale", "warehouse", "factory", "industrial"],
        "edit": ["edit", "author", "create", "compose"],
        "view": ["view", "explore", "visualiz", "inspect"],
        "collab": ["collab", "team", "multi", "presence", "live"],
    }

    for keyword_group, keywords in special_keywords.items():
        if any(kw in query_lower for kw in keywords):
            # Check if template has related features
            template_text = f"{description} {' '.join(features)} {' '.join(use_cases)}".lower()
            if any(kw in template_text for kw in keywords):
                score += 0.05

    return min(1.0, score)


async def search_app_examples(query: str, top_k: int = 5, category_filter: str = "") -> Dict[str, Any]:
    """
    Search for Kit application examples using semantic search.

    Args:
        query: Search query describing the desired application type or use case
        top_k: Number of top results to return (default: 5)
        category_filter: Optional category filter (editor, authoring, visualization, streaming, configuration)

    Returns:
        Dictionary with:
        - success: Whether search was successful
        - results: List of matching app templates with scores
        - error: Error message if failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {"query": query, "top_k": top_k, "category_filter": category_filter}

    success = True
    error_msg = None

    try:
        # Load app templates
        templates = load_app_templates()

        if not templates:
            return {"success": False, "error": "No app templates available", "results": []}

        # Calculate relevance scores for each template
        scored_templates = []

        for template_id, template_data in templates.items():
            # Apply category filter if specified
            if category_filter:
                template_category = template_data.get("category", "")
                if category_filter.lower() != template_category.lower():
                    continue

            # Calculate relevance score
            score = calculate_relevance_score(query, template_data)

            # Only include templates with non-zero scores
            if score > 0:
                scored_templates.append(
                    {
                        "id": template_id,
                        "name": template_data.get("name", "Unknown"),
                        "category": template_data.get("category", ""),
                        "description": template_data.get("description", ""),
                        "score": round(score, 3),
                        "use_cases": template_data.get("use_cases", [])[:3],  # Top 3 use cases
                        "key_features": template_data.get("key_features", [])[:5],  # Top 5 features
                        "supports_streaming": template_data.get("supports_streaming", False),
                        "dependencies_count": template_data.get("dependencies_count", 0),
                    }
                )

        # Sort by score (descending)
        scored_templates.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top_k results
        if top_k:
            scored_templates = scored_templates[:top_k]

        # Format results
        if not scored_templates:
            result_text = f"No Kit application templates found matching '{query}'"
            if category_filter:
                result_text += f" in category '{category_filter}'"
        else:
            result_lines = [f"# Kit Application Templates Search Results\n"]
            result_lines.append(f"Query: '{query}'")
            if category_filter:
                result_lines.append(f"Category Filter: {category_filter}")
            result_lines.append(f"Found {len(scored_templates)} matching template(s)\n")

            for i, template in enumerate(scored_templates, 1):
                result_lines.append(f"\n## {i}. {template['name']} (Score: {template['score']})")
                result_lines.append(f"**Category**: {template['category']}")
                result_lines.append(f"**Description**: {template['description']}")

                if template["use_cases"]:
                    result_lines.append("\n**Use Cases**:")
                    for use_case in template["use_cases"]:
                        result_lines.append(f"- {use_case}")

                if template["key_features"]:
                    result_lines.append("\n**Key Features**:")
                    for feature in template["key_features"][:3]:  # Show top 3 features in search
                        result_lines.append(f"- {feature}")

                result_lines.append(f"\n**Template ID**: `{template['id']}`")
                result_lines.append(f"**Dependencies**: {template['dependencies_count']} extensions")
                if template["supports_streaming"]:
                    result_lines.append("**Streaming**: ✓ Supported")

            result_lines.append(
                f"\n\n💡 **Tip**: Use `get_app_examples` with the template ID to get complete details including README and .kit file contents"
            )

            result_text = "\n".join(result_lines)

        logger.info(f"Search for '{query}' returned {len(scored_templates)} results")

        return {
            "success": True,
            "result": result_text,
            "results": scored_templates,
            "total_found": len(scored_templates),
        }

    except Exception as e:
        logger.error(f"Failed to search app examples: {e}")
        error_msg = str(e)
        success = False
        return {"success": False, "error": f"Search failed: {str(e)}", "results": []}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="search_app_examples",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
