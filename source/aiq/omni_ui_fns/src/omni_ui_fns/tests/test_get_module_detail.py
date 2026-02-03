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

"""Test the get_module_detail function."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omni_ui_fns.functions.get_module_detail import get_module_detail
from omni_ui_fns.utils import get_atlas_service


def test_atlas_service_availability():
    """Test that the Atlas service is available."""
    print(f"\n{'='*60}")
    print("TEST: Atlas Service Availability")
    print(f"{'='*60}")

    atlas_service = get_atlas_service()
    is_available = atlas_service.is_available()

    print(f"Atlas service available: {is_available}")

    if is_available:
        print("[OK] Atlas service is available")
    else:
        print("[FAIL] Atlas service is NOT available")
        print("[WARN] Tests will fail without Atlas data")


def test_get_available_modules():
    """Test get_module_detail with None to get available modules."""
    print(f"\n{'='*60}")
    print("TEST: Get Available Modules (module_names=None)")
    print(f"{'='*60}")

    atlas_service = get_atlas_service()
    if not atlas_service.is_available():
        print("[SKIP] Skipping - Atlas service not available")
        return

    try:
        result = asyncio.run(get_module_detail(None))

        print(f"Result keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert result["result"], "Result should not be empty"

        # Parse JSON result
        result_data = json.loads(result["result"])

        print(f"\n[OK] Retrieved available modules")
        print(f"Total count: {result_data.get('total_count')}")
        print(f"Available modules: {result_data.get('available_modules')[:5]}...")  # First 5

        assert "available_modules" in result_data, "Should contain available_modules"
        assert "total_count" in result_data, "Should contain total_count"
        assert isinstance(result_data["available_modules"], list), "available_modules should be a list"
        assert len(result_data["available_modules"]) > 0, "Should have at least one module"

        print(f"[OK] Available modules test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


def test_get_single_module():
    """Test get_module_detail with a single module name."""
    print(f"\n{'='*60}")
    print("TEST: Get Single Module Detail")
    print(f"{'='*60}")

    atlas_service = get_atlas_service()
    if not atlas_service.is_available():
        print("[SKIP] Skipping - Atlas service not available")
        return

    try:
        # First get available modules to pick a valid one
        available_result = asyncio.run(get_module_detail(None))
        available_data = json.loads(available_result["result"])
        available_modules = available_data["available_modules"]

        if not available_modules:
            print("[SKIP] No modules available to test")
            return

        # Test with first available module
        test_module = available_modules[0]
        print(f"Testing with module: {test_module}")

        result = asyncio.run(get_module_detail([test_module]))

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert result["result"], "Result should not be empty"

        # Parse JSON result
        result_data = json.loads(result["result"])

        print(f"\n[OK] Retrieved module detail")
        print(f"Module: {result_data.get('full_name', result_data.get('module_name'))}")
        print(f"Result keys: {list(result_data.keys())}")

        # Check for expected fields in module detail
        if "error" not in result_data:
            assert "full_name" in result_data or "module_name" in result_data, "Should have module name"
            print(f"[OK] Single module test passed!")
        else:
            print(f"[WARN] Module returned error: {result_data['error']}")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


def test_get_multiple_modules():
    """Test get_module_detail with multiple module names."""
    print(f"\n{'='*60}")
    print("TEST: Get Multiple Module Details")
    print(f"{'='*60}")

    atlas_service = get_atlas_service()
    if not atlas_service.is_available():
        print("[SKIP] Skipping - Atlas service not available")
        return

    try:
        # First get available modules to pick valid ones
        available_result = asyncio.run(get_module_detail(None))
        available_data = json.loads(available_result["result"])
        available_modules = available_data["available_modules"]

        if len(available_modules) < 2:
            print("[SKIP] Need at least 2 modules to test multiple modules")
            return

        # Test with first two available modules
        test_modules = available_modules[:2]
        print(f"Testing with modules: {test_modules}")

        result = asyncio.run(get_module_detail(test_modules))

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Multiple modules should return success if at least one succeeds
        assert result.get("success") is not None, "Should have success field"
        assert result["result"], "Result should not be empty"

        # Parse JSON result
        result_data = json.loads(result["result"])

        print(f"\n[OK] Retrieved multiple module details")
        print(f"Total requested: {result_data.get('total_requested')}")
        print(f"Successful: {result_data.get('successful')}")
        print(f"Failed: {result_data.get('failed')}")

        assert "modules" in result_data, "Should contain modules array"
        assert "total_requested" in result_data, "Should contain total_requested"
        assert result_data["total_requested"] == len(test_modules), "total_requested should match input"

        # Display each module result
        for i, module_result in enumerate(result_data["modules"], 1):
            print(f"\nModule {i}:")
            if "error" in module_result:
                print(f"  [FAIL] Error: {module_result['error']}")
            else:
                print(f"  [OK] {module_result.get('full_name', module_result.get('module_name'))}")

        print(f"\n[OK] Multiple modules test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


def test_invalid_module_name():
    """Test get_module_detail with an invalid module name."""
    print(f"\n{'='*60}")
    print("TEST: Invalid Module Name")
    print(f"{'='*60}")

    atlas_service = get_atlas_service()
    if not atlas_service.is_available():
        print("[SKIP] Skipping - Atlas service not available")
        return

    try:
        # Test with invalid module name
        invalid_module = "this_module_does_not_exist_xyz_123"
        print(f"Testing with invalid module: {invalid_module}")

        result = asyncio.run(get_module_detail([invalid_module]))

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Should fail or return error for invalid module
        if result["success"]:
            # If it somehow succeeds, check the result data
            result_data = json.loads(result["result"])
            assert "error" in result_data, "Should have error field for invalid module"
            print(f"[OK] Invalid module correctly returned error: {result_data['error']}")
        else:
            print(f"[OK] Invalid module correctly failed: {result['error']}")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


def test_empty_list():
    """Test get_module_detail with empty list."""
    print(f"\n{'='*60}")
    print("TEST: Empty List")
    print(f"{'='*60}")

    try:
        result = asyncio.run(get_module_detail([]))

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Should fail with empty list
        assert result["success"] == False, "Should fail with empty list"
        assert "empty" in result["error"].lower(), "Error should mention empty list"

        print(f"[OK] Empty list correctly rejected: {result['error']}")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


def test_string_input():
    """Test get_module_detail with string input (legacy support)."""
    print(f"\n{'='*60}")
    print("TEST: String Input (Legacy)")
    print(f"{'='*60}")

    atlas_service = get_atlas_service()
    if not atlas_service.is_available():
        print("[SKIP] Skipping - Atlas service not available")
        return

    try:
        # First get available modules to pick a valid one
        available_result = asyncio.run(get_module_detail(None))
        available_data = json.loads(available_result["result"])
        available_modules = available_data["available_modules"]

        if not available_modules:
            print("[SKIP] No modules available to test")
            return

        # Test with string input (should be converted to list internally)
        test_module = available_modules[0]
        print(f"Testing with string: '{test_module}'")

        result = asyncio.run(get_module_detail(test_module))

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Should handle string input
        if result["success"]:
            print(f"[OK] String input correctly handled")
        else:
            print(f"[FAIL] String input failed: {result['error']}")
            raise AssertionError("String input should be handled")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


def test_invalid_type():
    """Test get_module_detail with invalid type."""
    print(f"\n{'='*60}")
    print("TEST: Invalid Type")
    print(f"{'='*60}")

    try:
        # Test with invalid type (number)
        result = asyncio.run(get_module_detail(123))

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Should fail with invalid type
        assert result["success"] == False, "Should fail with invalid type"
        print(f"[OK] Invalid type correctly rejected: {result['error']}")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


def run_all_tests():
    """Run all tests manually without pytest."""
    print("\n" + "=" * 80)
    print("RUNNING GET_MODULE_DETAIL TESTS")
    print("=" * 80)

    test_count = 0
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    tests = [
        ("Atlas Service Availability", test_atlas_service_availability),
        ("Get Available Modules", test_get_available_modules),
        ("Get Single Module", test_get_single_module),
        ("Get Multiple Modules", test_get_multiple_modules),
        ("Invalid Module Name", test_invalid_module_name),
        ("Empty List", test_empty_list),
        ("String Input (Legacy)", test_string_input),
        ("Invalid Type", test_invalid_type),
    ]

    for test_name, test_func in tests:
        test_count += 1
        try:
            test_func()
            passed_count += 1
        except Exception as e:
            if "[SKIP]" in str(e) or "Skipping" in str(e):
                skipped_count += 1
            else:
                failed_count += 1
                print(f"\n[FAIL] {test_name} failed: {e}")

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {test_count}")
    print(f"[OK] Passed: {passed_count}")
    print(f"[FAIL] Failed: {failed_count}")
    print(f"[SKIP] Skipped: {skipped_count}")

    if failed_count == 0:
        print("\n[OK] ALL TESTS PASSED!")
    else:
        print(f"\n[FAIL] {failed_count} TEST(S) FAILED")

    print("=" * 80 + "\n")

    return failed_count == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
