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

"""Test script for the get_api_details function."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path to import the function
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_fns.functions.get_api_details import get_api_details


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def print_result(test_name: str, success: bool, details: str = ""):
    """Print test result with status marker."""
    status = "[OK]" if success else "[FAIL]"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")


async def test_single_valid_api():
    """Test with a single valid API reference."""
    print_section("Test 1: Single Valid API Reference")

    test_ref = "omni.ui@ColorShade"
    print(f"Testing with: {test_ref}")

    result = await get_api_details(test_ref)

    # Check success
    if not result.get("success"):
        print_result("Single valid API", False, f"Error: {result.get('error')}")
        return False

    # Parse result
    try:
        data = json.loads(result["result"])
        print(f"\nAPI Details:")
        print(f"  Symbol: {data.get('symbol', 'N/A')}")
        print(f"  Type: {data.get('type', 'N/A')}")
        print(f"  Extension: {data.get('extension_id', 'N/A')}")
        print(f"  Full Name: {data.get('full_name', 'N/A')}")

        signature = data.get("signature", "N/A")
        if signature != "N/A":
            print(f"  Signature: {signature}")

        docstring = data.get("docstring", "")
        if docstring:
            print(f"  Docstring (first 100 chars): {docstring[:100]}...")

        # Check for methods if it's a class
        if data.get("type") == "class":
            methods = data.get("methods", [])
            print(f"  Methods: {len(methods)} found")
            if methods:
                print(f"    Sample methods:")
                for method in methods[:3]:
                    print(f"      - {method.get('name', 'N/A')}")

        print_result("Single valid API", True, "Successfully retrieved API details")
        return True

    except json.JSONDecodeError as e:
        print_result("Single valid API", False, f"Failed to parse JSON: {e}")
        return False


async def test_multiple_api_references():
    """Test with multiple API references."""
    print_section("Test 2: Multiple API References")

    test_refs = ["omni.ui@ColorShade", "omni.ui@FloatShade", "omni.ui@StringShade"]
    print(f"Testing with: {test_refs}")

    result = await get_api_details(test_refs)

    # Check success
    if not result.get("success"):
        print_result("Multiple API references", False, f"Error: {result.get('error')}")
        return False

    # Parse result
    try:
        data = json.loads(result["result"])

        # For multiple APIs, result should have "apis" array
        if "apis" in data:
            apis = data["apis"]
            total = data.get("total_requested", 0)
            successful = data.get("successful", 0)
            failed = data.get("failed", 0)

            print(f"\nResults:")
            print(f"  Total requested: {total}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")

            print(f"\nAPI Details:")
            for api in apis:
                if "error" in api:
                    print(f"  - {api.get('api_reference', 'N/A')}: ERROR - {api['error']}")
                else:
                    print(f"  - {api.get('symbol', 'N/A')} ({api.get('type', 'N/A')})")

            print_result("Multiple API references", True, f"Retrieved {successful} APIs successfully")
            return successful > 0
        else:
            print_result("Multiple API references", False, "Expected 'apis' array in result")
            return False

    except json.JSONDecodeError as e:
        print_result("Multiple API references", False, f"Failed to parse JSON: {e}")
        return False


async def test_none_input():
    """Test with None input (should list available APIs info)."""
    print_section("Test 3: None Input (List Available APIs)")

    print("Testing with: None")

    result = await get_api_details(None)

    # Check success
    if not result.get("success"):
        print_result("None input", False, f"Error: {result.get('error')}")
        return False

    # Parse result
    try:
        data = json.loads(result["result"])

        print(f"\nAvailable APIs Info:")
        print(f"  Total count: {data.get('total_count', 0)}")
        print(f"  Usage: {data.get('usage', 'N/A')}")
        print(f"  Format: {data.get('format', 'N/A')}")

        api_refs = data.get("available_api_references", [])
        if api_refs:
            print(f"\n  Sample API references (first 5):")
            for ref in api_refs[:5]:
                print(f"    - {ref}")

        print_result("None input", True, f"Retrieved {len(api_refs)} available API references")
        return True

    except json.JSONDecodeError as e:
        print_result("None input", False, f"Failed to parse JSON: {e}")
        return False


async def test_invalid_format():
    """Test with invalid API reference format (missing @)."""
    print_section("Test 4: Invalid Format (Missing @)")

    test_ref = "omni.ui.Window"  # Missing @ separator
    print(f"Testing with: {test_ref}")

    result = await get_api_details(test_ref)

    # Should fail with format error
    if result.get("success"):
        print_result("Invalid format", False, "Expected failure but got success")
        return False

    error = result.get("error", "")
    print(f"Error message: {error}")

    if "Invalid API reference format" in error:
        print_result("Invalid format", True, "Correctly rejected invalid format")
        return True
    else:
        print_result("Invalid format", False, f"Unexpected error: {error}")
        return False


async def test_nonexistent_api():
    """Test with non-existent API reference."""
    print_section("Test 5: Non-existent API")

    test_ref = "omni.ui@NonExistentClass"
    print(f"Testing with: {test_ref}")

    result = await get_api_details(test_ref)

    # Should fail
    if result.get("success"):
        print_result("Non-existent API", False, "Expected failure but got success")
        return False

    error = result.get("error", "")
    print(f"Error message: {error}")

    # Should indicate API not found
    if "not found" in error.lower():
        print_result("Non-existent API", True, "Correctly reported API not found")
        return True
    else:
        print_result("Non-existent API", False, f"Unexpected error: {error}")
        return False


async def test_empty_string():
    """Test with empty string input."""
    print_section("Test 6: Empty String Input")

    test_ref = ""
    print(f"Testing with: (empty string)")

    result = await get_api_details(test_ref)

    # Should fail
    if result.get("success"):
        print_result("Empty string", False, "Expected failure but got success")
        return False

    error = result.get("error", "")
    print(f"Error message: {error}")

    if "cannot be empty" in error.lower():
        print_result("Empty string", True, "Correctly rejected empty string")
        return True
    else:
        print_result("Empty string", False, f"Unexpected error: {error}")
        return False


async def test_empty_list():
    """Test with empty list input."""
    print_section("Test 7: Empty List Input")

    test_refs = []
    print(f"Testing with: []")

    result = await get_api_details(test_refs)

    # Should fail
    if result.get("success"):
        print_result("Empty list", False, "Expected failure but got success")
        return False

    error = result.get("error", "")
    print(f"Error message: {error}")

    if "cannot be empty" in error.lower():
        print_result("Empty list", True, "Correctly rejected empty list")
        return True
    else:
        print_result("Empty list", False, f"Unexpected error: {error}")
        return False


async def test_mixed_valid_invalid():
    """Test with mix of valid and invalid API references."""
    print_section("Test 8: Mixed Valid and Invalid References")

    test_refs = ["omni.ui@ColorShade", "omni.ui@NonExistent", "omni.ui@FloatShade"]
    print(f"Testing with: {test_refs}")

    result = await get_api_details(test_refs)

    # Check success (should succeed even if some fail)
    if not result.get("success"):
        print_result("Mixed references", False, f"Error: {result.get('error')}")
        return False

    # Parse result
    try:
        data = json.loads(result["result"])

        if "apis" in data:
            apis = data["apis"]
            successful = data.get("successful", 0)
            failed = data.get("failed", 0)

            print(f"\nResults:")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")

            # Check we have at least some successes and some failures
            has_success = any("error" not in api for api in apis)
            has_failure = any("error" in api for api in apis)

            if has_success and has_failure:
                print_result("Mixed references", True, "Correctly handled mix of valid/invalid APIs")
                return True
            else:
                print_result("Mixed references", False, "Expected both successes and failures")
                return False
        else:
            print_result("Mixed references", False, "Expected 'apis' array in result")
            return False

    except json.JSONDecodeError as e:
        print_result("Mixed references", False, f"Failed to parse JSON: {e}")
        return False


async def test_class_with_methods():
    """Test retrieving a class and examining its methods."""
    print_section("Test 9: Class with Methods Details")

    test_ref = "omni.ui@ColorShade"
    print(f"Testing with: {test_ref}")

    result = await get_api_details(test_ref)

    if not result.get("success"):
        print_result("Class with methods", False, f"Error: {result.get('error')}")
        return False

    try:
        data = json.loads(result["result"])

        # Check if it's a class
        if data.get("type") != "class":
            print_result("Class with methods", False, f"Expected class, got {data.get('type')}")
            return False

        methods = data.get("methods", [])
        print(f"\nClass: {data.get('symbol', 'N/A')}")
        print(f"Total methods: {len(methods)}")

        if methods:
            print(f"\nSample methods (first 5):")
            for i, method in enumerate(methods[:5], 1):
                method_name = method.get("name", "N/A")
                signature = method.get("signature", "N/A")
                print(f"  {i}. {method_name}")
                if signature != "N/A" and signature != f"{method_name}(...)":
                    print(f"     Signature: {signature}")

            print_result("Class with methods", True, f"Successfully retrieved class with {len(methods)} methods")
            return True
        else:
            print_result("Class with methods", False, "No methods found for class")
            return False

    except json.JSONDecodeError as e:
        print_result("Class with methods", False, f"Failed to parse JSON: {e}")
        return False


async def test_output_format():
    """Test that output format is correct."""
    print_section("Test 10: Output Format Validation")

    test_ref = "omni.ui@ColorShade"
    print(f"Testing with: {test_ref}")

    result = await get_api_details(test_ref)

    # Check result is a dict
    if not isinstance(result, dict):
        print_result("Output format", False, f"Expected dict, got {type(result)}")
        return False

    # Check required keys
    required_keys = ["success", "result", "error"]
    missing_keys = [k for k in required_keys if k not in result]

    if missing_keys:
        print_result("Output format", False, f"Missing keys: {missing_keys}")
        return False

    # Check success is bool
    if not isinstance(result["success"], bool):
        print_result("Output format", False, f"success should be bool, got {type(result['success'])}")
        return False

    # Check result is string
    if not isinstance(result["result"], str):
        print_result("Output format", False, f"result should be string, got {type(result['result'])}")
        return False

    # If successful, result should be valid JSON
    if result["success"]:
        try:
            json.loads(result["result"])
        except json.JSONDecodeError as e:
            print_result("Output format", False, f"result is not valid JSON: {e}")
            return False

    print(f"\nOutput format:")
    print(f"  success: {result['success']} (bool)")
    print(f"  result: <string of length {len(result['result'])}>")
    print(f"  error: {result['error']}")

    print_result("Output format", True, "Output format is correct")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print(" get_api_details Function Test Suite")
    print("=" * 60)

    # Run all tests
    tests = [
        ("Single Valid API", test_single_valid_api),
        ("Multiple API References", test_multiple_api_references),
        ("None Input (List Available)", test_none_input),
        ("Invalid Format", test_invalid_format),
        ("Non-existent API", test_nonexistent_api),
        ("Empty String", test_empty_string),
        ("Empty List", test_empty_list),
        ("Mixed Valid/Invalid", test_mixed_valid_invalid),
        ("Class with Methods", test_class_with_methods),
        ("Output Format", test_output_format),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = await test_func()
            results.append((test_name, passed))
        except Exception as e:
            print_result(test_name, False, f"Exception: {e}")
            results.append((test_name, False))

    # Print summary
    print_section("Test Summary")

    passed_count = 0
    failed_count = 0

    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if passed:
            passed_count += 1
        else:
            failed_count += 1

    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

    if failed_count == 0:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{failed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
