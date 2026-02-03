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

"""Test the get_method_detail function."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omni_ui_fns.functions.get_method_detail import get_method_detail


class MockAtlasService:
    """Mock OmniUI Atlas service for testing."""

    def __init__(self, available=True):
        self._available = available
        self._methods_data = {
            "omni.ui.Button.__init__": {
                "name": "__init__",
                "full_name": "omni.ui.Button.__init__",
                "class_name": "omni.ui.Button",
                "parameters": [
                    {"name": "self", "type": "Button"},
                    {"name": "text", "type": "str", "default": ""},
                    {"name": "width", "type": "float", "default": "0.0"},
                    {"name": "height", "type": "float", "default": "0.0"},
                ],
                "return_type": "None",
                "docstring": "Initialize a Button widget.",
                "is_static": False,
                "is_classmethod": False,
                "is_property": False,
                "match_score": 1.0,
            },
            "omni.ui.Window.show": {
                "name": "show",
                "full_name": "omni.ui.Window.show",
                "class_name": "omni.ui.Window",
                "parameters": [{"name": "self", "type": "Window"}],
                "return_type": "None",
                "docstring": "Show the window.",
                "is_static": False,
                "is_classmethod": False,
                "is_property": False,
                "match_score": 1.0,
            },
        }

    def is_available(self):
        """Check if service is available."""
        return self._available

    def get_method_detail(self, method_name):
        """Mock get_method_detail implementation."""
        if method_name in self._methods_data:
            return self._methods_data[method_name]
        else:
            return {"error": f"Method '{method_name}' not found in OmniUI Atlas data."}

    def get_method_list(self):
        """Mock get_method_list implementation."""
        return list(self._methods_data.keys())


def test_none_input():
    """Test get_method_detail with None input (get available methods info)."""
    print(f"\n{'='*60}")
    print("TEST: None Input (Get Available Methods Info)")
    print(f"{'='*60}")

    async def run_test():
        # Mock the atlas service
        mock_service = MockAtlasService()

        with patch("omni_ui_fns.functions.get_method_detail.get_atlas_service", return_value=mock_service):
            with patch("omni_ui_fns.functions.get_method_detail.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                result = await get_method_detail(None)

                print(f"Success: {result['success']}")
                print(f"Error: {result.get('error')}")

                if result["success"]:
                    print(f"[OK] Function succeeded")

                    # Parse the JSON result
                    result_data = json.loads(result["result"])
                    print(f"Available methods count: {result_data['total_count']}")
                    print(f"Available methods: {result_data['available_methods']}")

                    # Validate structure
                    assert "available_methods" in result_data, "Should have 'available_methods' key"
                    assert "total_count" in result_data, "Should have 'total_count' key"
                    assert "usage" in result_data, "Should have 'usage' key"
                    assert result_data["total_count"] == 2, "Should have 2 methods"

                    print(f"[OK] None input test passed!")
                else:
                    print(f"[FAIL] Function failed: {result.get('error')}")
                    raise AssertionError(f"Expected success but got error: {result.get('error')}")

    asyncio.run(run_test())


def test_single_method_name():
    """Test get_method_detail with a single method name."""
    print(f"\n{'='*60}")
    print("TEST: Single Method Name")
    print(f"{'='*60}")

    async def run_test():
        # Mock the atlas service
        mock_service = MockAtlasService()

        with patch("omni_ui_fns.functions.get_method_detail.get_atlas_service", return_value=mock_service):
            with patch("omni_ui_fns.functions.get_method_detail.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                # Test with single method name as string
                result = await get_method_detail("omni.ui.Button.__init__")

                print(f"Success: {result['success']}")
                print(f"Error: {result.get('error')}")

                if result["success"]:
                    print(f"[OK] Function succeeded")

                    # Parse the JSON result
                    result_data = json.loads(result["result"])
                    print(f"Method name: {result_data['name']}")
                    print(f"Full name: {result_data['full_name']}")
                    print(f"Class name: {result_data['class_name']}")
                    print(f"Return type: {result_data['return_type']}")

                    # Validate structure
                    assert result_data["name"] == "__init__", "Method name should be '__init__'"
                    assert (
                        result_data["full_name"] == "omni.ui.Button.__init__"
                    ), "Full name should be 'omni.ui.Button.__init__'"
                    assert result_data["class_name"] == "omni.ui.Button", "Class name should be 'omni.ui.Button'"
                    assert "parameters" in result_data, "Should have 'parameters' key"
                    assert "docstring" in result_data, "Should have 'docstring' key"

                    print(f"[OK] Single method name test passed!")
                else:
                    print(f"[FAIL] Function failed: {result.get('error')}")
                    raise AssertionError(f"Expected success but got error: {result.get('error')}")

    asyncio.run(run_test())


def test_multiple_method_names():
    """Test get_method_detail with multiple method names."""
    print(f"\n{'='*60}")
    print("TEST: Multiple Method Names")
    print(f"{'='*60}")

    async def run_test():
        # Mock the atlas service
        mock_service = MockAtlasService()

        with patch("omni_ui_fns.functions.get_method_detail.get_atlas_service", return_value=mock_service):
            with patch("omni_ui_fns.functions.get_method_detail.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                # Test with multiple method names
                method_names = ["omni.ui.Button.__init__", "omni.ui.Window.show"]
                result = await get_method_detail(method_names)

                print(f"Success: {result['success']}")
                print(f"Error: {result.get('error')}")

                if result["success"]:
                    print(f"[OK] Function succeeded")

                    # Parse the JSON result
                    result_data = json.loads(result["result"])
                    print(f"Total requested: {result_data['total_requested']}")
                    print(f"Successful: {result_data['successful']}")
                    print(f"Failed: {result_data['failed']}")

                    # Validate structure
                    assert result_data["total_requested"] == 2, "Should have requested 2 methods"
                    assert result_data["successful"] == 2, "Should have 2 successful methods"
                    assert result_data["failed"] == 0, "Should have 0 failed methods"
                    assert len(result_data["methods"]) == 2, "Should have 2 methods in results"

                    # Check individual methods
                    for method in result_data["methods"]:
                        print(f"  - Method: {method['full_name']}")
                        assert "full_name" in method, "Each method should have 'full_name'"
                        assert "error" not in method, "Method should not have error"

                    print(f"[OK] Multiple method names test passed!")
                else:
                    print(f"[FAIL] Function failed: {result.get('error')}")
                    raise AssertionError(f"Expected success but got error: {result.get('error')}")

    asyncio.run(run_test())


def test_nonexistent_method():
    """Test get_method_detail with a non-existent method name."""
    print(f"\n{'='*60}")
    print("TEST: Non-existent Method Name")
    print(f"{'='*60}")

    async def run_test():
        # Mock the atlas service
        mock_service = MockAtlasService()

        with patch("omni_ui_fns.functions.get_method_detail.get_atlas_service", return_value=mock_service):
            with patch("omni_ui_fns.functions.get_method_detail.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                # Test with non-existent method name
                result = await get_method_detail(["nonexistent.method"])

                print(f"Success: {result['success']}")
                print(f"Error: {result.get('error')}")

                if not result["success"]:
                    print(f"[OK] Function correctly failed for non-existent method")
                    assert "not found" in result["error"].lower(), "Error should mention method not found"
                    print(f"[OK] Non-existent method test passed!")
                else:
                    print(f"[FAIL] Function should have failed for non-existent method")
                    raise AssertionError("Expected failure for non-existent method")

    asyncio.run(run_test())


def test_partial_success():
    """Test get_method_detail with mix of valid and invalid method names."""
    print(f"\n{'='*60}")
    print("TEST: Partial Success (Mix of Valid and Invalid Methods)")
    print(f"{'='*60}")

    async def run_test():
        # Mock the atlas service
        mock_service = MockAtlasService()

        with patch("omni_ui_fns.functions.get_method_detail.get_atlas_service", return_value=mock_service):
            with patch("omni_ui_fns.functions.get_method_detail.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                # Test with mix of valid and invalid method names
                method_names = ["omni.ui.Button.__init__", "invalid.method", "omni.ui.Window.show"]
                result = await get_method_detail(method_names)

                print(f"Success: {result['success']}")
                print(f"Error: {result.get('error')}")

                if result["success"]:
                    print(f"[OK] Function succeeded with partial results")

                    # Parse the JSON result
                    result_data = json.loads(result["result"])
                    print(f"Total requested: {result_data['total_requested']}")
                    print(f"Successful: {result_data['successful']}")
                    print(f"Failed: {result_data['failed']}")

                    # Validate structure
                    assert result_data["total_requested"] == 3, "Should have requested 3 methods"
                    assert result_data["successful"] == 2, "Should have 2 successful methods"
                    assert result_data["failed"] == 1, "Should have 1 failed method"

                    # Check that failed method has error
                    failed_methods = [m for m in result_data["methods"] if "error" in m]
                    print(f"Failed methods count: {len(failed_methods)}")
                    assert len(failed_methods) == 1, "Should have 1 failed method in results"

                    print(f"[OK] Partial success test passed!")
                else:
                    print(f"[FAIL] Function should have succeeded with partial results")
                    raise AssertionError("Expected partial success")

    asyncio.run(run_test())


def test_empty_list():
    """Test get_method_detail with empty list."""
    print(f"\n{'='*60}")
    print("TEST: Empty List")
    print(f"{'='*60}")

    async def run_test():
        # Mock the atlas service
        mock_service = MockAtlasService()

        with patch("omni_ui_fns.functions.get_method_detail.get_atlas_service", return_value=mock_service):
            with patch("omni_ui_fns.functions.get_method_detail.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                # Test with empty list
                result = await get_method_detail([])

                print(f"Success: {result['success']}")
                print(f"Error: {result.get('error')}")

                if not result["success"]:
                    print(f"[OK] Function correctly failed for empty list")
                    assert "cannot be empty" in result["error"], "Error should mention empty array"
                    print(f"[OK] Empty list test passed!")
                else:
                    print(f"[FAIL] Function should have failed for empty list")
                    raise AssertionError("Expected failure for empty list")

    asyncio.run(run_test())


def test_atlas_unavailable():
    """Test get_method_detail when Atlas service is unavailable."""
    print(f"\n{'='*60}")
    print("TEST: Atlas Service Unavailable")
    print(f"{'='*60}")

    async def run_test():
        # Mock unavailable atlas service
        mock_service = MockAtlasService(available=False)

        with patch("omni_ui_fns.functions.get_method_detail.get_atlas_service", return_value=mock_service):
            with patch("omni_ui_fns.functions.get_method_detail.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                # Test with None input
                result = await get_method_detail(None)

                print(f"Success: {result['success']}")
                print(f"Error: {result.get('error')}")

                if not result["success"]:
                    print(f"[OK] Function correctly failed when Atlas unavailable")
                    assert "not available" in result["error"].lower(), "Error should mention unavailability"
                    print(f"[OK] Atlas unavailable test passed!")
                else:
                    print(f"[FAIL] Function should have failed when Atlas unavailable")
                    raise AssertionError("Expected failure when Atlas unavailable")

    asyncio.run(run_test())


def run_all_tests():
    """Run all tests manually without pytest."""
    print("\n" + "=" * 80)
    print("RUNNING GET_METHOD_DETAIL TESTS")
    print("=" * 80)

    tests = [
        ("None Input", test_none_input),
        ("Single Method Name", test_single_method_name),
        ("Multiple Method Names", test_multiple_method_names),
        ("Non-existent Method", test_nonexistent_method),
        ("Partial Success", test_partial_success),
        ("Empty List", test_empty_list),
        ("Atlas Unavailable", test_atlas_unavailable),
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            test_func()
            passed += 1
            print(f"[OK] {test_name} completed successfully")
        except Exception as e:
            failed += 1
            error_msg = f"[FAIL] {test_name} failed: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)

    if errors:
        print("\nFailed tests:")
        for error in errors:
            print(f"  {error}")
        print()
        raise AssertionError(f"{failed} test(s) failed")
    else:
        print("\n[OK] ALL TESTS PASSED!")
        print()


if __name__ == "__main__":
    run_all_tests()
