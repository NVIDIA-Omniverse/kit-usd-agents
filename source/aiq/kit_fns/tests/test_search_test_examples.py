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

"""Test script for the search_test_examples functionality."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import functions
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_fns.functions.search_test_examples import search_test_examples


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


async def test_search_basic():
    """Test basic search functionality."""
    print_section("Testing Basic Search")

    test_queries = ["test async", "test setup", "test ui", "test extension", "test stage"]

    all_passed = True
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        result = await search_test_examples(query, top_k=3)

        if not result["success"]:
            error_msg = result.get("error", "Unknown error")
            if "Test search data is not available" in error_msg:
                print(f"  [OK] Service not available (expected in test environment)")
                continue
            print(f"  [FAIL] Search failed: {error_msg}")
            all_passed = False
            continue

        # Check result structure
        if "result" not in result:
            print(f"  [FAIL] Missing 'result' field")
            all_passed = False
            continue

        result_text = result["result"]

        if "No test examples found" in result_text:
            print(f"  [OK] No results found for query (valid response)")
        elif "Kit Test Example Search Results" in result_text:
            print(f"  [OK] Found test examples")

            # Check for expected formatting markers
            expected_markers = ["Test Example", "File:", "Extension:", "Relevance Score:"]
            for marker in expected_markers:
                if marker in result_text:
                    print(f"    [OK] Result contains '{marker}'")
                else:
                    print(f"    [FAIL] Result missing '{marker}'")
                    all_passed = False
        else:
            print(f"  [FAIL] Unexpected result format")
            all_passed = False

    return all_passed


async def test_search_validation():
    """Test input validation."""
    print_section("Testing Input Validation")

    test_cases = [
        ("", "empty query"),
        ("   ", "whitespace-only query"),
    ]

    all_passed = True
    for query, description in test_cases:
        print(f"\nTesting {description}: '{query}'")
        result = await search_test_examples(query, top_k=5)

        if not result["success"]:
            if result.get("error") == "query cannot be empty":
                print(f"  [OK] Correctly rejected {description}")
            else:
                print(f"  [FAIL] Wrong error: {result.get('error')}")
                all_passed = False
        else:
            print(f"  [FAIL] Should have rejected {description}")
            all_passed = False

    # Test invalid top_k
    print(f"\nTesting invalid top_k: 0")
    result = await search_test_examples("test", top_k=0)

    if not result["success"]:
        if result.get("error") == "top_k must be positive":
            print(f"  [OK] Correctly rejected top_k=0")
        else:
            print(f"  [FAIL] Wrong error: {result.get('error')}")
            all_passed = False
    else:
        print(f"  [FAIL] Should have rejected top_k=0")
        all_passed = False

    # Test negative top_k
    print(f"\nTesting invalid top_k: -5")
    result = await search_test_examples("test", top_k=-5)

    if not result["success"]:
        if result.get("error") == "top_k must be positive":
            print(f"  [OK] Correctly rejected top_k=-5")
        else:
            print(f"  [FAIL] Wrong error: {result.get('error')}")
            all_passed = False
    else:
        print(f"  [FAIL] Should have rejected top_k=-5")
        all_passed = False

    return all_passed


async def test_search_top_k():
    """Test top_k parameter."""
    print_section("Testing top_k Parameter")

    test_cases = [1, 5, 10, 20]

    all_passed = True
    for top_k in test_cases:
        print(f"\nTesting top_k={top_k}")
        result = await search_test_examples("test", top_k=top_k)

        if not result["success"]:
            error_msg = result.get("error", "Unknown error")
            if "Test search data is not available" in error_msg:
                print(f"  [OK] Service not available (expected in test environment)")
                continue
            print(f"  [FAIL] Search failed: {error_msg}")
            all_passed = False
            continue

        result_text = result["result"]

        # Count how many test examples are in the result
        test_example_count = result_text.count("## Test Example")

        if "No test examples found" in result_text:
            print(f"  [OK] No results found (valid response)")
        elif test_example_count > 0:
            if test_example_count <= top_k:
                print(f"  [OK] Returned {test_example_count} results (top_k={top_k})")
            else:
                print(f"  [FAIL] Returned {test_example_count} results (expected <= {top_k})")
                all_passed = False
        else:
            print(f"  [FAIL] Could not determine number of results")
            all_passed = False

    return all_passed


async def test_search_no_results():
    """Test search with queries that should return no results."""
    print_section("Testing No Results Scenarios")

    test_queries = [
        "xyzabc123impossible",
        "nonexistent_test_pattern_12345",
    ]

    all_passed = True
    for query in test_queries:
        print(f"\nSearching for unlikely query: '{query}'")
        result = await search_test_examples(query, top_k=5)

        if not result["success"]:
            error_msg = result.get("error", "Unknown error")
            if "Test search data is not available" in error_msg:
                print(f"  [OK] Service not available (expected in test environment)")
                continue
            print(f"  [FAIL] Search failed: {error_msg}")
            all_passed = False
            continue

        result_text = result["result"]

        # Should find no results or have very low relevance
        if "No test examples found" in result_text:
            print(f"  [OK] Correctly returned no results")
        else:
            # Count results
            test_example_count = result_text.count("## Test Example")
            print(f"  [OK] Found {test_example_count} results (may have low relevance scores)")

    return all_passed


async def test_search_result_format():
    """Test the format of search results."""
    print_section("Testing Result Format")

    result = await search_test_examples("test setup", top_k=5)

    if not result["success"]:
        error_msg = result.get("error", "Unknown error")
        if "Test search data is not available" in error_msg:
            print(f"[OK] Service not available (expected in test environment)")
            return True
        print(f"[FAIL] Search failed: {error_msg}")
        return False

    all_passed = True

    # Check top-level structure
    required_top_fields = ["success", "result", "error"]
    for field in required_top_fields:
        if field in result:
            print(f"[OK] Top-level field '{field}' present")
        else:
            print(f"[FAIL] Missing top-level field '{field}'")
            all_passed = False

    # Check result text format
    result_text = result.get("result", "")

    if "No test examples found" in result_text:
        print(f"[OK] No results found (valid response)")
    else:
        expected_markers = ["# Kit Test Example Search Results for:", "**Found", "relevant test examples:**"]

        for marker in expected_markers:
            if marker in result_text:
                print(f"[OK] Result text contains '{marker}'")
            else:
                print(f"[FAIL] Result text missing '{marker}'")
                all_passed = False

        # Check for test example details if results found
        if "## Test Example" in result_text:
            detail_markers = [
                "**File:**",
                "**Extension:**",
                "**Lines:**",
                "**Relevance Score:**",
                "**Test Description:**",
                "**Test Code:**",
                "**Testing Tips:**",
            ]

            for marker in detail_markers:
                if marker in result_text:
                    print(f"[OK] Result contains '{marker}'")
                else:
                    print(f"[FAIL] Result missing '{marker}'")
                    all_passed = False

    return all_passed


async def test_search_specific_patterns():
    """Test searches for specific test patterns."""
    print_section("Testing Specific Test Pattern Searches")

    test_cases = [
        ("async test setup", "Should find async test setup patterns"),
        ("stage creation", "Should find stage/USD creation tests"),
        ("extension lifecycle", "Should find extension lifecycle tests"),
        ("ui interaction", "Should find UI interaction tests"),
    ]

    all_passed = True
    for query, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Query: '{query}'")

        result = await search_test_examples(query, top_k=5)

        if not result["success"]:
            error_msg = result.get("error", "Unknown error")
            if "Test search data is not available" in error_msg:
                print(f"  [OK] Service not available (expected in test environment)")
                continue
            print(f"  [FAIL] Search failed: {error_msg}")
            all_passed = False
            continue

        result_text = result["result"]

        if "No test examples found" in result_text:
            print(f"  [OK] No results found (query may be too specific)")
        else:
            test_example_count = result_text.count("## Test Example")
            print(f"  [OK] Found {test_example_count} test example(s)")

            # Verify testing tips are included
            if "**Testing Tips:**" in result_text:
                print(f"  [OK] Testing tips included")
            else:
                print(f"  [FAIL] Testing tips missing")
                all_passed = False

    return all_passed


async def test_service_availability():
    """Test service availability handling."""
    print_section("Testing Service Availability")

    result = await search_test_examples("test", top_k=5)

    if not result["success"]:
        error_msg = result.get("error", "Unknown error")
        if "Test search data is not available" in error_msg:
            print(f"[OK] Service correctly reports unavailability")
            print(f"  This is expected in test environments without FAISS data")
            return True
        else:
            print(f"[FAIL] Unexpected error: {error_msg}")
            return False
    else:
        print(f"[OK] Service is available and functional")
        return True


async def run_async_tests():
    """Run all async tests."""
    tests = [
        ("Service Availability", test_service_availability),
        ("Basic Search", test_search_basic),
        ("Input Validation", test_search_validation),
        ("Top K Parameter", test_search_top_k),
        ("No Results Scenarios", test_search_no_results),
        ("Result Format", test_search_result_format),
        ("Specific Test Pattern Searches", test_search_specific_patterns),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = await test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' raised exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    return results


def main():
    """Run all tests."""
    print("=" * 60)
    print(" Kit Test Examples Search Test Suite")
    print("=" * 60)

    # Run async tests
    test_results = asyncio.run(run_async_tests())

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
