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

"""Test the get_class_detail function."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omni_ui_fns.functions.get_class_detail import get_class_detail
from omni_ui_fns.utils import get_atlas_service


def test_atlas_service_availability():
    """Test that the OmniUI Atlas service is available."""
    print(f"\n{'='*60}")
    print("TEST: Atlas Service Availability")
    print(f"{'='*60}")

    atlas_service = get_atlas_service()
    is_available = atlas_service.is_available()

    print(f"Atlas service available: {is_available}")

    if is_available:
        print("[OK] Atlas service is available")
        # List some available classes
        classes = atlas_service.get_class_names_list()
        print(f"Total classes available: {len(classes)}")
        print(f"Sample classes: {classes[:10]}")
    else:
        print("[FAIL] Atlas service is NOT available")
        print("[WARN] Tests will be limited without Atlas data")

    return is_available


async def test_get_class_detail_none():
    """Test get_class_detail with None (should list available classes)."""
    print(f"\n{'='*60}")
    print("TEST: get_class_detail with None")
    print(f"{'='*60}")

    try:
        result = await get_class_detail(None)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        if result["success"]:
            print("[OK] Function succeeded with None input")

            # Parse the JSON result
            result_data = json.loads(result["result"])
            print(f"Available classes count: {result_data.get('total_count')}")
            print(f"Sample classes: {result_data.get('available_classes', [])[:10]}")

            assert "available_classes" in result_data, "Should contain 'available_classes' key"
            assert "total_count" in result_data, "Should contain 'total_count' key"
            assert isinstance(result_data["available_classes"], list), "available_classes should be a list"
            print("[OK] Result format is correct")

            print("\n--- Sample Output ---")
            print(json.dumps(result_data, indent=2)[:500] + "...")
        else:
            print(f"[FAIL] Function failed: {result.get('error')}")

        return result

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        raise


async def test_get_class_detail_single_string():
    """Test get_class_detail with a single class name as string."""
    print(f"\n{'='*60}")
    print("TEST: get_class_detail with Single String")
    print(f"{'='*60}")

    try:
        class_name = "Button"
        print(f"Testing with class name: '{class_name}'")

        result = await get_class_detail(class_name)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        if result["success"]:
            print(f"[OK] Function succeeded for class '{class_name}'")

            # Parse the JSON result
            result_data = json.loads(result["result"])

            # Check for expected fields
            expected_fields = ["full_name", "docstring", "methods", "properties"]
            print("\nChecking expected fields:")
            for field in expected_fields:
                has_field = field in result_data
                status = "[OK]" if has_field else "[WARN]"
                print(f"  {status} {field}: {has_field}")

            # Print class information
            print(f"\nClass Details:")
            print(f"  Full Name: {result_data.get('full_name', 'N/A')}")
            print(f"  Docstring: {result_data.get('docstring', 'N/A')[:100]}...")

            if "methods" in result_data:
                methods = result_data["methods"]
                print(f"  Methods count: {len(methods)}")
                if isinstance(methods, dict):
                    print(f"  Sample methods: {list(methods.keys())[:5]}")
                elif isinstance(methods, list):
                    sample_methods = []
                    for m in methods[:5]:
                        if isinstance(m, dict):
                            sample_methods.append(m.get("name", str(m)))
                        else:
                            sample_methods.append(str(m))
                    print(f"  Sample methods: {sample_methods}")

            if "properties" in result_data:
                properties = result_data["properties"]
                print(f"  Properties count: {len(properties)}")
                if isinstance(properties, dict):
                    print(f"  Sample properties: {list(properties.keys())[:5]}")
                elif isinstance(properties, list):
                    sample_props = []
                    for p in properties[:5]:
                        if isinstance(p, dict):
                            sample_props.append(p.get("name", str(p)))
                        else:
                            sample_props.append(str(p))
                    print(f"  Sample properties: {sample_props}")

            print("\n--- Sample Output (first 800 chars) ---")
            print(result["result"][:800] + "...")
            print("[OK] Single string test passed")

        else:
            print(f"[FAIL] Function failed: {result.get('error')}")

        return result

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        raise


async def test_get_class_detail_single_list():
    """Test get_class_detail with a single class name as list."""
    print(f"\n{'='*60}")
    print("TEST: get_class_detail with Single Item List")
    print(f"{'='*60}")

    try:
        class_names = ["Button"]
        print(f"Testing with class names: {class_names}")

        result = await get_class_detail(class_names)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        if result["success"]:
            print(f"[OK] Function succeeded for classes {class_names}")

            # Parse the JSON result - should be a single object, not array
            result_data = json.loads(result["result"])

            # For single item list, should return single object
            print(f"\nResult type: {type(result_data)}")
            assert isinstance(result_data, dict), "Single item should return dict, not array"
            print("[OK] Result is a dictionary (single object)")

            print(f"  Full Name: {result_data.get('full_name', 'N/A')}")
            print(f"  Has methods: {'methods' in result_data}")
            print(f"  Has properties: {'properties' in result_data}")

        else:
            print(f"[FAIL] Function failed: {result.get('error')}")

        return result

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        raise


async def test_get_class_detail_multiple():
    """Test get_class_detail with multiple class names."""
    print(f"\n{'='*60}")
    print("TEST: get_class_detail with Multiple Classes")
    print(f"{'='*60}")

    try:
        class_names = ["Button", "Label", "Slider"]
        print(f"Testing with class names: {class_names}")

        result = await get_class_detail(class_names)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        if result["success"]:
            print(f"[OK] Function succeeded for classes {class_names}")

            # Parse the JSON result
            result_data = json.loads(result["result"])

            # For multiple items, should return object with classes array
            print(f"\nResult structure:")
            print(f"  Has 'classes' key: {'classes' in result_data}")
            print(f"  Has 'total_requested' key: {'total_requested' in result_data}")
            print(f"  Has 'successful' key: {'successful' in result_data}")
            print(f"  Has 'failed' key: {'failed' in result_data}")

            assert "classes" in result_data, "Multiple items should have 'classes' key"
            assert "total_requested" in result_data, "Should have 'total_requested' key"
            assert "successful" in result_data, "Should have 'successful' key"
            assert "failed" in result_data, "Should have 'failed' key"

            print(f"\nSummary:")
            print(f"  Total requested: {result_data['total_requested']}")
            print(f"  Successful: {result_data['successful']}")
            print(f"  Failed: {result_data['failed']}")

            # Check each class result
            for i, class_result in enumerate(result_data["classes"]):
                class_name = class_names[i] if i < len(class_names) else "unknown"
                has_error = "error" in class_result
                status = "[FAIL]" if has_error else "[OK]"
                print(f"  {status} {class_name}: {'ERROR - ' + class_result['error'] if has_error else 'Success'}")

            print("\n--- Sample Output (first 1000 chars) ---")
            print(result["result"][:1000] + "...")
            print("[OK] Multiple classes test passed")

        else:
            print(f"[FAIL] Function failed: {result.get('error')}")

        return result

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        raise


async def test_get_class_detail_invalid_class():
    """Test get_class_detail with an invalid class name."""
    print(f"\n{'='*60}")
    print("TEST: get_class_detail with Invalid Class")
    print(f"{'='*60}")

    try:
        class_name = "NonExistentClassXYZ123"
        print(f"Testing with invalid class name: '{class_name}'")

        result = await get_class_detail(class_name)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Should fail or return error in result
        if not result["success"]:
            print(f"[OK] Function correctly failed for invalid class")
            print(f"  Error message: {result.get('error')}")
        else:
            result_data = json.loads(result["result"])
            if "error" in result_data:
                print(f"[OK] Function returned error in result")
                print(f"  Error message: {result_data.get('error')}")
            else:
                print(f"[WARN] Function succeeded for invalid class (unexpected)")

        return result

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        raise


async def test_get_class_detail_empty_string():
    """Test get_class_detail with empty string."""
    print(f"\n{'='*60}")
    print("TEST: get_class_detail with Empty String")
    print(f"{'='*60}")

    try:
        class_name = ""
        print(f"Testing with empty string")

        result = await get_class_detail(class_name)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Should fail
        if not result["success"]:
            print(f"[OK] Function correctly failed for empty string")
            print(f"  Error message: {result.get('error')}")
        else:
            print(f"[FAIL] Function should fail for empty string")

        assert not result["success"], "Should fail for empty string"

        return result

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        raise


async def test_get_class_detail_empty_list():
    """Test get_class_detail with empty list."""
    print(f"\n{'='*60}")
    print("TEST: get_class_detail with Empty List")
    print(f"{'='*60}")

    try:
        class_names = []
        print(f"Testing with empty list")

        result = await get_class_detail(class_names)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Should fail
        if not result["success"]:
            print(f"[OK] Function correctly failed for empty list")
            print(f"  Error message: {result.get('error')}")
        else:
            print(f"[FAIL] Function should fail for empty list")

        assert not result["success"], "Should fail for empty list"

        return result

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        raise


async def test_get_class_detail_mixed_valid_invalid():
    """Test get_class_detail with mix of valid and invalid class names."""
    print(f"\n{'='*60}")
    print("TEST: get_class_detail with Mixed Valid/Invalid Classes")
    print(f"{'='*60}")

    try:
        class_names = ["Button", "InvalidClassXYZ", "Label"]
        print(f"Testing with class names: {class_names}")

        result = await get_class_detail(class_names)

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        # Should succeed if at least one class is valid
        if result["success"]:
            print(f"[OK] Function succeeded (partial success allowed)")

            result_data = json.loads(result["result"])
            print(f"\nSummary:")
            print(f"  Total requested: {result_data['total_requested']}")
            print(f"  Successful: {result_data['successful']}")
            print(f"  Failed: {result_data['failed']}")

            # Check each class
            for class_result in result_data["classes"]:
                has_error = "error" in class_result
                class_name = class_result.get("class_name", class_result.get("full_name", "unknown"))
                status = "[FAIL]" if has_error else "[OK]"
                print(f"  {status} {class_name}: {'ERROR - ' + class_result['error'] if has_error else 'Success'}")

            # Should have at least one success and one failure
            assert result_data["successful"] > 0, "Should have at least one success"
            print("[OK] Partial success handled correctly")

        else:
            print(f"[WARN] Function failed entirely: {result.get('error')}")

        return result

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        raise


def run_all_tests():
    """Run all tests manually without pytest."""
    print("\n" + "=" * 80)
    print("RUNNING OMNI_UI_FNS GET_CLASS_DETAIL TESTS")
    print("=" * 80)

    # Track results
    test_results = []

    try:
        # Test 0: Atlas service availability
        is_available = test_atlas_service_availability()

        if not is_available:
            print("\n[FAIL] Atlas service not available - cannot run remaining tests")
            return

        # Test 1: None input
        print("\n--- Running Test 1 ---")
        result = asyncio.run(test_get_class_detail_none())
        test_results.append(("None input", result["success"]))

        # Test 2: Single string
        print("\n--- Running Test 2 ---")
        result = asyncio.run(test_get_class_detail_single_string())
        test_results.append(("Single string", result["success"]))

        # Test 3: Single item list
        print("\n--- Running Test 3 ---")
        result = asyncio.run(test_get_class_detail_single_list())
        test_results.append(("Single item list", result["success"]))

        # Test 4: Multiple classes
        print("\n--- Running Test 4 ---")
        result = asyncio.run(test_get_class_detail_multiple())
        test_results.append(("Multiple classes", result["success"]))

        # Test 5: Invalid class
        print("\n--- Running Test 5 ---")
        result = asyncio.run(test_get_class_detail_invalid_class())
        test_results.append(("Invalid class", not result["success"]))  # Should fail

        # Test 6: Empty string
        print("\n--- Running Test 6 ---")
        result = asyncio.run(test_get_class_detail_empty_string())
        test_results.append(("Empty string", not result["success"]))  # Should fail

        # Test 7: Empty list
        print("\n--- Running Test 7 ---")
        result = asyncio.run(test_get_class_detail_empty_list())
        test_results.append(("Empty list", not result["success"]))  # Should fail

        # Test 8: Mixed valid/invalid
        print("\n--- Running Test 8 ---")
        result = asyncio.run(test_get_class_detail_mixed_valid_invalid())
        test_results.append(("Mixed valid/invalid", result["success"]))

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for _, success in test_results if success)
        total = len(test_results)

        for test_name, success in test_results:
            status = "[OK]" if success else "[FAIL]"
            print(f"{status} {test_name}")

        print(f"\nTotal: {passed}/{total} tests passed")

        if passed == total:
            print("\n" + "=" * 80)
            print("[OK] ALL TESTS PASSED!")
            print("=" * 80 + "\n")
        else:
            print("\n" + "=" * 80)
            print(f"[FAIL] {total - passed} TEST(S) FAILED")
            print("=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[FAIL] TEST SUITE FAILED: {e}")
        print("=" * 80 + "\n")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
