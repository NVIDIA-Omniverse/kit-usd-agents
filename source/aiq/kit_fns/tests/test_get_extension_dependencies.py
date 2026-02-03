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

"""Test script for get_extension_dependencies function."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path to import services
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_fns.functions.get_extension_dependencies import get_extension_dependencies
from kit_fns.services.extension_service import ExtensionService


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


async def test_basic_dependencies():
    """Test basic dependency retrieval."""
    print_section("Test 1: Basic Dependency Retrieval")

    # Get a test extension that has dependencies
    extension_service = ExtensionService()

    if not extension_service.is_available():
        print("[FAIL] Extension service not available")
        return False

    # Get first few extensions to test with
    extensions = extension_service.get_extension_list()[:10]

    if not extensions:
        print("[FAIL] No extensions available for testing")
        return False

    # Find an extension with dependencies
    test_extension = None
    for ext_id in extensions:
        metadata = extension_service.atlas_service.get_extension_metadata(ext_id)
        if metadata and metadata.get("dependencies"):
            test_extension = ext_id
            print(f"Testing with extension: {test_extension}")
            break

    if not test_extension:
        print("No extensions with dependencies found in first 10 extensions")
        print("Using first available extension as fallback...")
        test_extension = extensions[0]
        print(f"Testing with extension: {test_extension}")

    # Test the function
    result = await get_extension_dependencies(test_extension, depth=2)

    if not result["success"]:
        print(f"[FAIL] Function returned error: {result['error']}")
        return False

    # Parse result
    try:
        data = json.loads(result["result"])
        print(f"[OK] Successfully retrieved dependencies for: {data.get('extension_id')}")
        print(f"     Extension name: {data.get('name')}")
        print(f"     Version: {data.get('version')}")
        print(f"     Depth: {data.get('depth')}")

        deps = data.get("dependencies", {})
        required = deps.get("required", [])
        print(f"     Required dependencies: {len(required)}")

        if required:
            print(f"     First 3 dependencies: {', '.join(required[:3])}")

        children = deps.get("children", {})
        print(f"     Nested dependencies: {len(children)}")

        return True

    except json.JSONDecodeError as e:
        print(f"[FAIL] Failed to parse JSON result: {e}")
        return False


async def test_with_depth():
    """Test dependency retrieval with different depths."""
    print_section("Test 2: Dependency Depth Levels")

    extension_service = ExtensionService()
    extensions = extension_service.get_extension_list()

    if not extensions:
        print("[FAIL] No extensions available")
        return False

    test_extension = extensions[0]
    print(f"Testing with extension: {test_extension}")

    all_passed = True

    # Test different depths
    for depth in [0, 1, 2, 3]:
        print(f"\n  Testing depth={depth}...")
        result = await get_extension_dependencies(test_extension, depth=depth)

        if result["success"]:
            data = json.loads(result["result"])
            print(f"  [OK] Depth {depth} successful")
            print(f"       Result depth: {data.get('depth')}")
        else:
            print(f"  [FAIL] Depth {depth} failed: {result['error']}")
            all_passed = False

    return all_passed


async def test_optional_dependencies():
    """Test with include_optional flag."""
    print_section("Test 3: Optional Dependencies")

    extension_service = ExtensionService()
    extensions = extension_service.get_extension_list()

    if not extensions:
        print("[FAIL] No extensions available")
        return False

    test_extension = extensions[0]
    print(f"Testing with extension: {test_extension}")

    # Test without optional
    print("\n  Testing with include_optional=False...")
    result1 = await get_extension_dependencies(test_extension, depth=1, include_optional=False)

    if not result1["success"]:
        print(f"  [FAIL] Failed without optional: {result1['error']}")
        return False

    data1 = json.loads(result1["result"])
    print(f"  [OK] Without optional: include_optional={data1.get('include_optional')}")

    # Test with optional
    print("\n  Testing with include_optional=True...")
    result2 = await get_extension_dependencies(test_extension, depth=1, include_optional=True)

    if not result2["success"]:
        print(f"  [FAIL] Failed with optional: {result2['error']}")
        return False

    data2 = json.loads(result2["result"])
    print(f"  [OK] With optional: include_optional={data2.get('include_optional')}")

    return True


async def test_invalid_inputs():
    """Test error handling with invalid inputs."""
    print_section("Test 4: Error Handling")

    all_passed = True

    # Test empty extension ID
    print("  Testing empty extension ID...")
    result = await get_extension_dependencies("", depth=1)
    if not result["success"]:
        print(f"  [OK] Correctly rejected empty extension ID")
    else:
        print(f"  [FAIL] Should have rejected empty extension ID")
        all_passed = False

    # Test whitespace-only extension ID
    print("\n  Testing whitespace-only extension ID...")
    result = await get_extension_dependencies("   ", depth=1)
    if not result["success"]:
        print(f"  [OK] Correctly rejected whitespace-only extension ID")
    else:
        print(f"  [FAIL] Should have rejected whitespace-only extension ID")
        all_passed = False

    # Test negative depth
    print("\n  Testing negative depth...")
    result = await get_extension_dependencies("omni.ui", depth=-1)
    if not result["success"]:
        print(f"  [OK] Correctly rejected negative depth")
    else:
        print(f"  [FAIL] Should have rejected negative depth")
        all_passed = False

    # Test non-existent extension
    print("\n  Testing non-existent extension...")
    result = await get_extension_dependencies("non.existent.extension", depth=1)
    if not result["success"]:
        print(f"  [OK] Correctly handled non-existent extension")
        print(f"       Error: {result['error']}")
    else:
        print(f"  [FAIL] Should have returned error for non-existent extension")
        all_passed = False

    return all_passed


async def test_specific_extensions():
    """Test with specific well-known extensions."""
    print_section("Test 5: Specific Extension IDs")

    extension_service = ExtensionService()

    # Get available extensions
    available_extensions = extension_service.get_extension_list()

    # Test extensions (use first available if specific ones don't exist)
    test_extensions = []
    common_extensions = ["omni.ui", "omni.kit.window.console", "omni.usd"]

    for ext in common_extensions:
        if ext in available_extensions:
            test_extensions.append(ext)

    # If no common extensions found, use first 2 available
    if not test_extensions and available_extensions:
        test_extensions = available_extensions[:2]

    if not test_extensions:
        print("[FAIL] No extensions available for testing")
        return False

    all_passed = True

    for ext_id in test_extensions:
        print(f"\n  Testing extension: {ext_id}")
        result = await get_extension_dependencies(ext_id, depth=2)

        if result["success"]:
            data = json.loads(result["result"])
            deps = data.get("dependencies", {})
            required = deps.get("required", [])
            children = deps.get("children", {})

            print(f"  [OK] Extension: {data.get('name')}")
            print(f"       Version: {data.get('version')}")
            print(f"       Direct dependencies: {len(required)}")
            print(f"       Nested dependencies: {len(children)}")

            if required:
                print(f"       Dependencies: {', '.join(required[:3])}")
        else:
            print(f"  [FAIL] Failed: {result['error']}")
            all_passed = False

    return all_passed


async def test_result_format():
    """Test that the result has the expected format."""
    print_section("Test 6: Result Format Validation")

    extension_service = ExtensionService()
    extensions = extension_service.get_extension_list()

    if not extensions:
        print("[FAIL] No extensions available")
        return False

    test_extension = extensions[0]
    print(f"Testing with extension: {test_extension}")

    result = await get_extension_dependencies(test_extension, depth=2)

    if not result["success"]:
        print(f"[FAIL] Function failed: {result['error']}")
        return False

    # Validate result structure
    all_passed = True

    print("\n  Validating result structure...")

    # Check top-level keys
    required_keys = ["success", "result", "error"]
    for key in required_keys:
        if key in result:
            print(f"  [OK] Key '{key}' present")
        else:
            print(f"  [FAIL] Key '{key}' missing")
            all_passed = False

    # Parse and validate JSON structure
    try:
        data = json.loads(result["result"])

        # Check required fields in data
        data_keys = ["extension_id", "name", "version", "dependencies", "depth", "include_optional"]
        for key in data_keys:
            if key in data:
                print(f"  [OK] Data key '{key}' present")
            else:
                print(f"  [FAIL] Data key '{key}' missing")
                all_passed = False

        # Check dependencies structure
        deps = data.get("dependencies", {})
        if "required" in deps:
            print(f"  [OK] Dependencies has 'required' key")
        else:
            print(f"  [FAIL] Dependencies missing 'required' key")
            all_passed = False

        if "children" in deps:
            print(f"  [OK] Dependencies has 'children' key")
        else:
            print(f"  [FAIL] Dependencies missing 'children' key")
            all_passed = False

    except json.JSONDecodeError as e:
        print(f"  [FAIL] Invalid JSON in result: {e}")
        all_passed = False

    return all_passed


async def main():
    """Run all tests."""
    print("=" * 60)
    print(" get_extension_dependencies Test Suite")
    print("=" * 60)

    # Run tests
    tests = [
        ("Basic Dependencies", test_basic_dependencies()),
        ("Depth Levels", test_with_depth()),
        ("Optional Dependencies", test_optional_dependencies()),
        ("Error Handling", test_invalid_inputs()),
        ("Specific Extensions", test_specific_extensions()),
        ("Result Format", test_result_format()),
    ]

    results = []
    for test_name, test_coro in tests:
        try:
            passed = await test_coro
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' raised exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print_section("Test Summary")

    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")
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
    sys.exit(asyncio.run(main()))
