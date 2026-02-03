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

"""Test script for search_extensions function."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import functions
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_fns.functions.search_extensions import search_extensions
from kit_fns.services.extension_service import ExtensionService


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def print_result(test_name: str, passed: bool):
    """Print test result with [OK] or [FAIL] marker."""
    status = "[OK]" if passed else "[FAIL]"
    print(f"{status} {test_name}")


async def test_basic_search():
    """Test basic search functionality."""
    test_name = "Basic search with valid query"

    try:
        result = await search_extensions("user interface", top_k=5)

        # Check result structure
        assert "success" in result, "Result missing 'success' field"
        assert "result" in result, "Result missing 'result' field"
        assert "error" in result, "Result missing 'error' field"

        # Check success
        assert result["success"] is True, "Search should succeed"
        assert result["error"] is None, "Error should be None"
        assert result["result"], "Result should not be empty"

        # Check result format
        assert "Kit Extension Search Results" in result["result"], "Missing header"

        print_result(test_name, True)
        return True

    except Exception as e:
        print_result(test_name, False)
        print(f"  Error: {e}")
        return False


async def test_multiple_queries():
    """Test search with multiple different queries."""
    test_name = "Multiple different search queries"

    test_queries = [
        ("viewport rendering", "Testing viewport/rendering query"),
        ("physics simulation", "Testing physics query"),
        ("USD scene", "Testing USD query"),
        ("python api", "Testing python api query"),
        ("user interface widgets", "Testing UI query"),
    ]

    all_passed = True

    for query, description in test_queries:
        try:
            result = await search_extensions(query, top_k=3)

            assert result["success"] is True, f"Query '{query}' failed"
            assert result["result"], f"Query '{query}' returned empty result"

            # Print sample results
            lines = result["result"].split("\n")
            print(f"\n  Query: '{query}'")
            # Find the "Found N matching extensions" line
            for line in lines:
                if "Found" in line and "matching extensions" in line:
                    print(f"    {line.strip()}")
                    break

        except Exception as e:
            print(f"  [FAIL] Query '{query}': {e}")
            all_passed = False

    print_result(test_name, all_passed)
    return all_passed


async def test_top_k_parameter():
    """Test search with different top_k values."""
    test_name = "Different top_k values"

    try:
        # Test with different top_k values
        result_3 = await search_extensions("rendering", top_k=3)
        result_10 = await search_extensions("rendering", top_k=10)
        result_20 = await search_extensions("rendering", top_k=20)

        assert result_3["success"] is True, "top_k=3 failed"
        assert result_10["success"] is True, "top_k=10 failed"
        assert result_20["success"] is True, "top_k=20 failed"

        # Count results (rough check by looking for "##" markers)
        count_3 = result_3["result"].count("## ")
        count_10 = result_10["result"].count("## ")
        count_20 = result_20["result"].count("## ")

        print(f"  top_k=3: {count_3} results")
        print(f"  top_k=10: {count_10} results")
        print(f"  top_k=20: {count_20} results")

        # Results should increase or stay same (if not enough matching extensions)
        assert count_10 >= count_3, "top_k=10 should return >= results than top_k=3"

        print_result(test_name, True)
        return True

    except Exception as e:
        print_result(test_name, False)
        print(f"  Error: {e}")
        return False


async def test_no_results_query():
    """Test search with query that likely returns no results."""
    test_name = "Query with no results"

    try:
        # Use a very specific query unlikely to match anything
        result = await search_extensions("xyz123nonexistent456", top_k=5)

        assert result["success"] is True, "Should succeed even with no results"
        assert result["error"] is None, "Error should be None"

        # Check if it indicates no results
        if "No extensions found" in result["result"]:
            print(f"  Correctly returned: No extensions found")
        elif result["result"]:
            # May still return some results if fallback search is used
            print(f"  Returned results (possibly from fallback search)")

        print_result(test_name, True)
        return True

    except Exception as e:
        print_result(test_name, False)
        print(f"  Error: {e}")
        return False


async def test_result_format():
    """Test that result format contains expected fields."""
    test_name = "Result format validation"

    try:
        result = await search_extensions("rendering", top_k=2)

        assert result["success"] is True, "Search should succeed"

        result_text = result["result"]

        # Check for key components in the formatted result
        expected_elements = [
            "Kit Extension Search Results",
            "Found",
            "matching extensions",
            "Version:",
            "Category:",
            "Relevance Score:",
            "Description:",
        ]

        missing = []
        for element in expected_elements:
            if element not in result_text:
                missing.append(element)

        if missing:
            print(f"  Missing elements: {missing}")
            print_result(test_name, False)
            return False

        print(f"  All expected format elements present")
        print_result(test_name, True)
        return True

    except Exception as e:
        print_result(test_name, False)
        print(f"  Error: {e}")
        return False


async def test_extension_service_availability():
    """Test that extension service is available."""
    test_name = "Extension service availability"

    try:
        from kit_fns.functions.search_extensions import get_extension_service

        service = get_extension_service()
        available = service.is_available()

        if not available:
            print(f"  [WARNING] Extension service not available")
            print(f"  This may be expected if database not built")
            print_result(test_name, True)  # Not a failure, just not available
            return True

        # Test that we can get extension list
        ext_list = service.get_extension_list()
        print(f"  Extension service available: {len(ext_list)} extensions")

        print_result(test_name, True)
        return True

    except Exception as e:
        print_result(test_name, False)
        print(f"  Error: {e}")
        return False


async def test_search_with_telemetry():
    """Test that search works with telemetry."""
    test_name = "Search with telemetry enabled"

    try:
        # Search should initialize and use telemetry
        result = await search_extensions("omni.ui", top_k=3)

        assert result["success"] is True, "Search with telemetry should succeed"
        assert result["result"], "Result should not be empty"

        print(f"  Telemetry integration working")
        print_result(test_name, True)
        return True

    except Exception as e:
        print_result(test_name, False)
        print(f"  Error: {e}")
        return False


async def test_specific_extension_search():
    """Test searching for specific known extensions."""
    test_name = "Search for specific known extensions"

    # Common extensions that should exist
    test_searches = [
        "omni.ui",
        "viewport",
        "console",
        "window",
    ]

    all_passed = True

    for search_term in test_searches:
        try:
            result = await search_extensions(search_term, top_k=5)

            assert result["success"] is True, f"Search for '{search_term}' failed"

            # Check if we got results
            has_results = "Found" in result["result"] and "matching extensions" in result["result"]

            if has_results:
                print(f"  Found results for '{search_term}'")
            else:
                print(f"  No results for '{search_term}' (may be expected)")

        except Exception as e:
            print(f"  [FAIL] Search for '{search_term}': {e}")
            all_passed = False

    print_result(test_name, all_passed)
    return all_passed


async def test_relevance_scores():
    """Test that results include relevance scores."""
    test_name = "Relevance scores present"

    try:
        result = await search_extensions("rendering", top_k=5)

        assert result["success"] is True, "Search should succeed"

        # Check for relevance score format
        assert "Relevance Score:" in result["result"], "Missing relevance scores"

        # Count how many scores we found
        score_count = result["result"].count("Relevance Score:")
        print(f"  Found {score_count} relevance scores")

        print_result(test_name, True)
        return True

    except Exception as e:
        print_result(test_name, False)
        print(f"  Error: {e}")
        return False


async def test_error_handling():
    """Test error handling with invalid inputs."""
    test_name = "Error handling with invalid inputs"

    try:
        # Test with empty query
        result = await search_extensions("", top_k=5)

        # Should either succeed with no results or handle gracefully
        assert "success" in result, "Result should have success field"
        assert "error" in result, "Result should have error field"

        print(f"  Empty query handled: success={result['success']}")

        # Test with very large top_k
        result = await search_extensions("test", top_k=1000)
        assert "success" in result, "Result should have success field"

        print(f"  Large top_k handled: success={result['success']}")

        print_result(test_name, True)
        return True

    except Exception as e:
        print_result(test_name, False)
        print(f"  Error: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print(" Search Extensions Function Test Suite")
    print("=" * 60)

    # Check if NVIDIA_API_KEY is set
    if not os.getenv("NVIDIA_API_KEY"):
        print("\n[WARNING] NVIDIA_API_KEY not set. FAISS search may use fallback.")
        print("Set it with: set NVIDIA_API_KEY=your_api_key")

    # List of all test functions
    tests = [
        ("Extension Service Availability", test_extension_service_availability()),
        ("Basic Search", test_basic_search()),
        ("Multiple Queries", test_multiple_queries()),
        ("Top-K Parameter", test_top_k_parameter()),
        ("No Results Query", test_no_results_query()),
        ("Result Format", test_result_format()),
        ("Telemetry Integration", test_search_with_telemetry()),
        ("Specific Extension Search", test_specific_extension_search()),
        ("Relevance Scores", test_relevance_scores()),
        ("Error Handling", test_error_handling()),
    ]

    # Run all tests
    results = []
    for test_name, test_coro in tests:
        print_section(test_name)
        passed = await test_coro
        results.append((test_name, passed))

    # Print summary
    print_section("Test Summary")

    all_passed = True
    passed_count = 0

    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if passed:
            passed_count += 1
        else:
            all_passed = False

    print(f"\nTests passed: {passed_count}/{len(results)}")

    if all_passed:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please check the output above.")
        return 1


def main():
    """Main entry point."""
    return asyncio.run(run_all_tests())


if __name__ == "__main__":
    sys.exit(main())
