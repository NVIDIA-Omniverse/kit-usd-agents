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

"""Test script for the search_settings functionality."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import services
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_mcp.functions.search_settings import search_settings
from kit_mcp.services.settings_service import SettingsService


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def test_settings_service():
    """Test the Settings service directly."""
    print_section("Testing Settings Service")

    settings_service = SettingsService()

    # Test availability
    print(f"Settings service available: {settings_service.is_available()}")

    if not settings_service.is_available():
        print("ERROR: Settings service not available. Run the settings pipeline first.")
        return False

    # Get statistics
    stats = settings_service.get_settings_statistics()
    print(f"Total settings: {stats.get('total_settings', 0)}")
    print(f"Extensions scanned: {stats.get('total_extensions_scanned', 0)}")
    print(f"FAISS available: {stats.get('faiss_available', False)}")
    print(
        f"Documented settings: {stats.get('documented_settings', 0)} ({stats.get('documentation_percentage', 0):.1f}%)"
    )

    # Show type distribution
    if "type_distribution" in stats:
        print("\nSetting types:")
        for type_name, count in sorted(stats["type_distribution"].items(), key=lambda x: (x[0] is None, x[0] or "")):
            print(f"  {type_name}: {count}")

    # Show prefix distribution
    if "prefix_distribution" in stats:
        print("\nSetting prefixes:")
        for prefix, count in sorted(stats["prefix_distribution"].items()):
            print(f"  /{prefix}/: {count} settings")

    # Test search
    print("\n" + "-" * 40)
    print("Testing semantic search...")

    test_queries = [
        ("viewport rendering", None, None),
        ("enable debug", None, "bool"),
        ("window settings", "app", None),
        ("persistent preferences", "persistent", None),
        ("ray tracing", "rtx", None),
    ]

    for query, prefix, type_filter in test_queries:
        filter_desc = []
        if prefix:
            filter_desc.append(f"prefix={prefix}")
        if type_filter:
            filter_desc.append(f"type={type_filter}")
        filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""

        print(f"\nSearching for: '{query}'{filter_str}")
        results = settings_service.search_settings(query, top_k=3, prefix_filter=prefix, type_filter=type_filter)

        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['setting']}")
                print(f"     Type: {result.get('type', 'unknown')}, Default: {result.get('default_value', 'None')}")
                print(f"     Score: {result.get('relevance_score', 0):.2f}")
                doc = result.get("documentation", "") or result.get("description", "")
                if doc:
                    print(f"     Doc: {doc[:100]}...")
        else:
            print("  No results found")

    # Test getting settings by extension
    print("\n" + "-" * 40)
    print("Testing get settings by extension...")
    test_ext = "omni.kit.viewport.window"
    ext_settings = settings_service.get_settings_by_extension(test_ext)
    print(f"Settings for {test_ext}: {len(ext_settings)} found")
    if ext_settings:
        for setting in ext_settings[:3]:
            print(f"  - {setting['setting']} ({setting['type']})")

    return True


async def test_search_function():
    """Test the search_settings function."""
    print_section("Testing search_settings Function")

    # Test basic search
    print("Testing basic search...")
    result = await search_settings("viewport grid settings", top_k=5)

    if result["success"]:
        print("✓ Basic search successful")
        # Print first few lines of result
        lines = result["result"].split("\n")[:10]
        for line in lines:
            if line.strip():
                print(f"  {line}")
    else:
        print(f"✗ Basic search failed: {result['error']}")
        return False

    # Test with filters
    print("\nTesting search with filters...")
    result = await search_settings("enable features", top_k=5, prefix_filter="exts", type_filter="bool")

    if result["success"]:
        print("✓ Filtered search successful")
        # Count results
        import re

        matches = re.findall(r"## \d+\.", result["result"])
        print(f"  Found {len(matches)} settings with filters")
    else:
        print(f"✗ Filtered search failed: {result['error']}")

    # Test with no results
    print("\nTesting search with unlikely query...")
    result = await search_settings("xyzabc123impossible", top_k=5)

    if result["success"]:
        if "No settings found" in result["result"]:
            print("✓ No results handled correctly")
        else:
            print("✗ Expected 'No settings found' message")
    else:
        print(f"✗ Search failed: {result['error']}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print(" Kit Settings Search Test Suite")
    print("=" * 60)

    # Check if NVIDIA_API_KEY is set
    if not os.getenv("NVIDIA_API_KEY"):
        print("\nWARNING: NVIDIA_API_KEY not set. FAISS search may not work.")
        print("Falling back to keyword search.")

    # Run tests
    tests_passed = []

    # Test Settings service
    tests_passed.append(("Settings Service", test_settings_service()))

    # Test search function
    tests_passed.append(("Search Function", asyncio.run(test_search_function())))

    # Print summary
    print_section("Test Summary")

    all_passed = True
    for test_name, passed in tests_passed:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
