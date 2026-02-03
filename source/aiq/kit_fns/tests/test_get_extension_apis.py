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

"""Test script for the get_extension_apis functionality."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path to import services
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_fns.functions.get_extension_apis import get_extension_apis
from kit_fns.services.api_service import APIService


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def test_api_service():
    """Test the API service directly."""
    print_section("Testing API Service")

    api_service = APIService()

    # Test availability
    print(f"API service available: {api_service.is_available()}")

    if not api_service.is_available():
        print("[FAIL] API service not available. Run the data collection pipeline first.")
        return False

    # Get API list
    api_list = api_service.get_api_list()
    print(f"Total APIs available: {len(api_list)}")

    if api_list:
        print("\nFirst 5 API references:")
        for api_ref in api_list[:5]:
            print(f"  - {api_ref}")

    # Test with known extension
    print("\n" + "-" * 40)
    print("Testing get_extension_apis with known extensions...")

    test_extensions = ["omni.ui", "omni.kit.commands"]

    for ext_id in test_extensions:
        print(f"\nTesting extension: {ext_id}")
        results = api_service.get_extension_apis([ext_id])

        if results:
            result = results[0]
            print(f"  Extension ID: {result.get('extension_id')}")
            print(f"  API count: {result.get('api_count', 0)}")
            print(f"  Class count: {result.get('class_count', 0)}")
            print(f"  Function count: {result.get('function_count', 0)}")

            if result.get("apis"):
                print(f"\n  First 3 APIs:")
                for api in result["apis"][:3]:
                    print(f"    - {api.get('api_reference')} ({api.get('type')})")
                    docstring = api.get("docstring", "")
                    if docstring:
                        print(f"      {docstring[:80]}...")

            if result.get("error"):
                print(f"  Error: {result['error']}")
            if result.get("note"):
                print(f"  Note: {result['note']}")
        else:
            print(f"  [FAIL] No results returned")

    print("\n[OK] API service tests completed")
    return True


async def test_get_extension_apis_none():
    """Test get_extension_apis with None input."""
    print_section("Test 1: None Input (Get Available APIs Info)")

    result = await get_extension_apis(None)

    if not result["success"]:
        print(f"[FAIL] Request failed: {result['error']}")
        return False

    print("[OK] Request succeeded")

    # Parse result
    try:
        data = json.loads(result["result"])
        print(f"Available API references: {data.get('total_count', 0)}")
        print(f"Usage: {data.get('usage', 'N/A')}")
        print(f"Format: {data.get('format', 'N/A')}")

        if data.get("available_api_references"):
            print(f"\nFirst 5 API references:")
            for api_ref in data["available_api_references"][:5]:
                print(f"  - {api_ref}")

        print("\n[OK] None input test passed")
        return True
    except json.JSONDecodeError as e:
        print(f"[FAIL] Failed to parse JSON result: {e}")
        return False


async def test_get_extension_apis_single():
    """Test get_extension_apis with single extension ID."""
    print_section("Test 2: Single Extension ID")

    # Test with a string
    print("Testing with single string: 'omni.ui'")
    result = await get_extension_apis("omni.ui")

    if not result["success"]:
        print(f"[FAIL] Request failed: {result['error']}")
        return False

    print("[OK] Request succeeded")

    # Parse result
    try:
        data = json.loads(result["result"])
        print(f"Extension ID: {data.get('extension_id')}")
        print(f"API count: {data.get('api_count', 0)}")
        print(f"Class count: {data.get('class_count', 0)}")
        print(f"Function count: {data.get('function_count', 0)}")

        if data.get("apis"):
            print(f"\nFirst 5 APIs:")
            for api in data["apis"][:5]:
                print(f"  - {api.get('api_reference')} ({api.get('type')})")

        if data.get("error"):
            print(f"  Error: {data['error']}")

        print("\n[OK] Single extension test passed")
        return True
    except json.JSONDecodeError as e:
        print(f"[FAIL] Failed to parse JSON result: {e}")
        return False


async def test_get_extension_apis_multiple():
    """Test get_extension_apis with multiple extension IDs."""
    print_section("Test 3: Multiple Extension IDs")

    test_extensions = ["omni.ui", "omni.kit.commands", "omni.usd"]
    print(f"Testing with list: {test_extensions}")

    result = await get_extension_apis(test_extensions)

    if not result["success"]:
        print(f"[FAIL] Request failed: {result['error']}")
        return False

    print("[OK] Request succeeded")

    # Parse result
    try:
        data = json.loads(result["result"])
        print(f"Total requested: {data.get('total_requested', 0)}")
        print(f"Extensions with APIs: {data.get('extensions_with_apis', 0)}")
        print(f"Total APIs: {data.get('total_apis', 0)}")

        if data.get("extensions"):
            print(f"\nExtension details:")
            for ext_data in data["extensions"]:
                print(f"\n  Extension: {ext_data.get('extension_id')}")
                print(f"    API count: {ext_data.get('api_count', 0)}")

                if ext_data.get("apis"):
                    print(f"    First 2 APIs:")
                    for api in ext_data["apis"][:2]:
                        print(f"      - {api.get('api_reference')}")

                if ext_data.get("error"):
                    print(f"    Error: {ext_data['error']}")
                if ext_data.get("note"):
                    print(f"    Note: {ext_data['note']}")

        print("\n[OK] Multiple extensions test passed")
        return True
    except json.JSONDecodeError as e:
        print(f"[FAIL] Failed to parse JSON result: {e}")
        return False


async def test_get_extension_apis_invalid():
    """Test get_extension_apis with invalid inputs."""
    print_section("Test 4: Invalid Inputs")

    test_cases = [
        ("empty string", ""),
        ("empty list", []),
        ("list with empty string", ["omni.ui", "", "omni.usd"]),
        ("invalid type (int)", 123),
        ("non-existent extension", "nonexistent.extension.xyz"),
    ]

    all_passed = True

    for test_name, test_input in test_cases:
        print(f"\nTesting: {test_name}")
        result = await get_extension_apis(test_input)

        # For non-existent extension, we expect success=True but with error in the result
        if test_name == "non-existent extension":
            if result["success"]:
                try:
                    data = json.loads(result["result"])
                    if data.get("error"):
                        print(f"  [OK] Correctly handled: {data['error']}")
                    else:
                        print(f"  [OK] Extension returned with 0 APIs")
                except json.JSONDecodeError:
                    print(f"  [FAIL] Failed to parse result")
                    all_passed = False
            else:
                print(f"  [OK] Request failed as expected: {result['error']}")
        else:
            # For other invalid inputs, we expect success=False
            if not result["success"]:
                print(f"  [OK] Correctly rejected: {result['error']}")
            else:
                print(f"  [FAIL] Should have failed but succeeded")
                all_passed = False

    if all_passed:
        print("\n[OK] All invalid input tests passed")
    else:
        print("\n[FAIL] Some invalid input tests failed")

    return all_passed


async def test_get_extension_apis_edge_cases():
    """Test get_extension_apis with edge cases."""
    print_section("Test 5: Edge Cases")

    # Test with list containing single extension
    print("Testing list with single extension...")
    result = await get_extension_apis(["omni.ui"])

    if result["success"]:
        print("[OK] Single-item list handled correctly")
        data = json.loads(result["result"])
        # Single extension returns the extension object directly (no wrapper)
        print(f"  Extension ID: {data.get('extension_id')}")
        print(f"  API count: {data.get('api_count', 0)}")
    else:
        print(f"[FAIL] Single-item list failed: {result['error']}")
        return False

    # Test with whitespace-only string
    print("\nTesting whitespace-only string...")
    result = await get_extension_apis("   ")

    if not result["success"]:
        print(f"[OK] Whitespace rejected: {result['error']}")
    else:
        print("[FAIL] Whitespace should have been rejected")
        return False

    print("\n[OK] Edge case tests passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print(" Kit Extension APIs Test Suite")
    print("=" * 60)

    # Run tests
    tests_passed = []

    # Test API service
    tests_passed.append(("API Service", test_api_service()))

    # Test async functions
    tests_passed.append(("None Input", asyncio.run(test_get_extension_apis_none())))
    tests_passed.append(("Single Extension", asyncio.run(test_get_extension_apis_single())))
    tests_passed.append(("Multiple Extensions", asyncio.run(test_get_extension_apis_multiple())))
    tests_passed.append(("Invalid Inputs", asyncio.run(test_get_extension_apis_invalid())))
    tests_passed.append(("Edge Cases", asyncio.run(test_get_extension_apis_edge_cases())))

    # Print summary
    print_section("Test Summary")

    all_passed = True
    for test_name, passed in tests_passed:
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
