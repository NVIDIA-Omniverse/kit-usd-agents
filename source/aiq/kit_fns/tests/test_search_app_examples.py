#!/usr/bin/env python
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

"""Test script for the search_app_examples functionality."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import functions
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_fns.functions.search_app_examples import calculate_relevance_score, load_app_templates, search_app_examples


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def test_load_app_templates():
    """Test loading app templates data."""
    print_section("Testing load_app_templates")

    templates = load_app_templates()

    if not templates:
        print("[FAIL] No templates loaded")
        return False

    print(f"[OK] Loaded {len(templates)} templates")

    # Check template structure
    expected_templates = ["kit_base_editor", "usd_composer", "usd_explorer", "usd_viewer", "streaming_configs"]
    for template_id in expected_templates:
        if template_id in templates:
            print(f"  [OK] Template '{template_id}' found")
            template_data = templates[template_id]

            # Verify required fields
            required_fields = ["name", "category", "description"]
            for field in required_fields:
                if field in template_data:
                    print(f"    [OK] Field '{field}': {template_data[field][:50]}...")
                else:
                    print(f"    [FAIL] Missing field '{field}'")
                    return False
        else:
            print(f"  [FAIL] Template '{template_id}' not found")
            return False

    return True


def test_calculate_relevance_score():
    """Test the relevance score calculation."""
    print_section("Testing calculate_relevance_score")

    # Mock template data
    template_data = {
        "name": "USD Explorer",
        "category": "visualization",
        "description": "Robust viewer for large-scale environments like factories and warehouses",
        "use_cases": [
            "Industrial facility visualization",
            "Digital twin applications",
            "Large-scale scene exploration",
        ],
        "key_features": ["Large scene optimization", "Annotation tools", "Live collaboration"],
    }

    test_cases = [
        # (query, expected_min_score, test_description)
        ("USD Explorer", 0.3, "Exact name match"),
        ("viewer", 0.15, "Description word match"),
        ("visualization", 0.15, "Category match"),
        ("factory", 0.03, "Use case word match"),
        ("large scene", 0.1, "Feature match"),
        ("streaming", 0.0, "No match expected"),
    ]

    all_passed = True
    for query, expected_min_score, description in test_cases:
        score = calculate_relevance_score(query, template_data)

        if score >= expected_min_score:
            print(f"[OK] {description}: '{query}' -> score {score:.3f} (>= {expected_min_score})")
        else:
            print(f"[FAIL] {description}: '{query}' -> score {score:.3f} (expected >= {expected_min_score})")
            all_passed = False

    return all_passed


async def test_search_basic():
    """Test basic search functionality."""
    print_section("Testing Basic Search")

    test_queries = ["editor", "streaming", "large scale", "authoring", "cloud"]

    all_passed = True
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        result = await search_app_examples(query, top_k=3)

        if not result["success"]:
            print(f"  [FAIL] Search failed: {result.get('error', 'Unknown error')}")
            all_passed = False
            continue

        # Check result structure
        if "results" not in result:
            print(f"  [FAIL] Missing 'results' field")
            all_passed = False
            continue

        if "total_found" not in result:
            print(f"  [FAIL] Missing 'total_found' field")
            all_passed = False
            continue

        total_found = result["total_found"]
        results = result["results"]

        print(f"  [OK] Found {total_found} template(s)")

        # Verify results structure
        for i, template in enumerate(results, 1):
            required_fields = ["id", "name", "category", "description", "score"]
            missing_fields = [field for field in required_fields if field not in template]

            if missing_fields:
                print(f"  [FAIL] Result {i} missing fields: {missing_fields}")
                all_passed = False
            else:
                print(f"  [OK] Result {i}: {template['name']} (Score: {template['score']})")

        # Verify scores are in descending order
        if len(results) > 1:
            scores = [r["score"] for r in results]
            if scores == sorted(scores, reverse=True):
                print(f"  [OK] Results sorted by score (descending)")
            else:
                print(f"  [FAIL] Results not properly sorted by score")
                all_passed = False

        # Verify result text is generated
        if "result" in result and result["result"]:
            print(f"  [OK] Result text generated ({len(result['result'])} chars)")
        else:
            print(f"  [FAIL] Result text not generated")
            all_passed = False

    return all_passed


async def test_search_with_filters():
    """Test search with category filters."""
    print_section("Testing Search with Category Filter")

    test_cases = [
        ("viewer", "visualization", "Should find USD Explorer"),
        ("editor", "authoring", "Should find USD Composer"),
        ("streaming", "streaming", "Should find USD Viewer"),
    ]

    all_passed = True
    for query, category_filter, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Query: '{query}', Category: '{category_filter}'")

        result = await search_app_examples(query, top_k=5, category_filter=category_filter)

        if not result["success"]:
            print(f"  [FAIL] Search failed: {result.get('error', 'Unknown error')}")
            all_passed = False
            continue

        # Verify all results match the category filter
        results = result["results"]
        if results:
            category_matches = all(r["category"].lower() == category_filter.lower() for r in results)
            if category_matches:
                print(f"  [OK] All {len(results)} result(s) match category '{category_filter}'")
                for r in results:
                    print(f"    - {r['name']} ({r['category']})")
            else:
                print(f"  [FAIL] Some results don't match category filter")
                all_passed = False
        else:
            print(f"  [OK] No results found (expected for strict filter)")

    return all_passed


async def test_search_top_k():
    """Test top_k parameter."""
    print_section("Testing top_k Parameter")

    test_cases = [1, 3, 5, 10]

    all_passed = True
    for top_k in test_cases:
        print(f"\nTesting top_k={top_k}")
        result = await search_app_examples("application", top_k=top_k)

        if not result["success"]:
            print(f"  [FAIL] Search failed: {result.get('error', 'Unknown error')}")
            all_passed = False
            continue

        results = result["results"]
        total_found = result["total_found"]

        # Verify we don't get more than top_k results
        if len(results) <= top_k:
            print(f"  [OK] Returned {len(results)} results (top_k={top_k})")
        else:
            print(f"  [FAIL] Returned {len(results)} results (expected <= {top_k})")
            all_passed = False

    return all_passed


async def test_search_no_results():
    """Test search with queries that should return no results."""
    print_section("Testing No Results Scenarios")

    test_queries = [
        "xyzabc123impossible",
        "nonexistent_feature_12345",
    ]

    all_passed = True
    for query in test_queries:
        print(f"\nSearching for unlikely query: '{query}'")
        result = await search_app_examples(query, top_k=5)

        if not result["success"]:
            print(f"  [FAIL] Search failed: {result.get('error', 'Unknown error')}")
            all_passed = False
            continue

        results = result["results"]
        total_found = result["total_found"]

        if total_found == 0 and len(results) == 0:
            print(f"  [OK] Correctly returned no results")

            # Verify the result text mentions no results
            if "result" in result and "No Kit application templates found" in result["result"]:
                print(f"  [OK] Result text indicates no matches")
            else:
                print(f"  [FAIL] Result text doesn't indicate no matches")
                all_passed = False
        else:
            print(f"  [FAIL] Expected no results, got {total_found}")
            all_passed = False

    return all_passed


async def test_search_result_format():
    """Test the format of search results."""
    print_section("Testing Result Format")

    result = await search_app_examples("editor", top_k=2)

    if not result["success"]:
        print(f"[FAIL] Search failed: {result.get('error', 'Unknown error')}")
        return False

    all_passed = True

    # Check top-level structure
    required_top_fields = ["success", "result", "results", "total_found"]
    for field in required_top_fields:
        if field in result:
            print(f"[OK] Top-level field '{field}' present")
        else:
            print(f"[FAIL] Missing top-level field '{field}'")
            all_passed = False

    # Check result text format
    result_text = result.get("result", "")
    expected_markers = ["# Kit Application Templates Search Results", "Query:", "**Template ID**:"]

    for marker in expected_markers:
        if marker in result_text:
            print(f"[OK] Result text contains '{marker}'")
        else:
            print(f"[FAIL] Result text missing '{marker}'")
            all_passed = False

    # Check results array structure
    if result["results"]:
        template = result["results"][0]
        required_template_fields = [
            "id",
            "name",
            "category",
            "description",
            "score",
            "use_cases",
            "key_features",
            "supports_streaming",
            "dependencies_count",
        ]

        for field in required_template_fields:
            if field in template:
                print(f"[OK] Template field '{field}' present: {type(template[field]).__name__}")
            else:
                print(f"[FAIL] Template missing field '{field}'")
                all_passed = False

    return all_passed


async def test_search_specific_templates():
    """Test searches that should find specific templates."""
    print_section("Testing Specific Template Searches")

    test_cases = [
        ("factory warehouse", "usd_explorer", "USD Explorer should match factory/warehouse"),
        ("cloud streaming", "usd_viewer", "USD Viewer should match cloud streaming"),
        ("authoring professional", "usd_composer", "USD Composer should match professional authoring"),
        ("minimal editor", "kit_base_editor", "Kit Base Editor should match minimal editor"),
    ]

    all_passed = True
    for query, expected_template_id, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Query: '{query}'")
        print(f"  Expected to find: '{expected_template_id}'")

        result = await search_app_examples(query, top_k=5)

        if not result["success"]:
            print(f"  [FAIL] Search failed: {result.get('error', 'Unknown error')}")
            all_passed = False
            continue

        # Check if expected template is in results
        template_ids = [r["id"] for r in result["results"]]

        if expected_template_id in template_ids:
            position = template_ids.index(expected_template_id) + 1
            print(f"  [OK] Found '{expected_template_id}' at position {position}")
        else:
            print(f"  [FAIL] Expected template '{expected_template_id}' not found")
            print(f"  Found templates: {template_ids}")
            all_passed = False

    return all_passed


async def run_async_tests():
    """Run all async tests."""
    tests = [
        ("Basic Search", test_search_basic),
        ("Search with Filters", test_search_with_filters),
        ("Top K Parameter", test_search_top_k),
        ("No Results Scenarios", test_search_no_results),
        ("Result Format", test_search_result_format),
        ("Specific Template Searches", test_search_specific_templates),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = await test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))

    return results


def main():
    """Run all tests."""
    print("=" * 60)
    print(" Kit App Examples Search Test Suite")
    print("=" * 60)

    # Run tests
    test_results = []

    # Sync tests
    test_results.append(("Load App Templates", test_load_app_templates()))
    test_results.append(("Calculate Relevance Score", test_calculate_relevance_score()))

    # Async tests
    async_results = asyncio.run(run_async_tests())
    test_results.extend(async_results)

    # Print summary
    print_section("Test Summary")

    all_passed = True
    for test_name, passed in test_results:
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
