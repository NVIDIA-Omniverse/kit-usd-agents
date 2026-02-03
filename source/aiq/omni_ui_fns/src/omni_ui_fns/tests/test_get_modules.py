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

"""Test the get_modules function."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omni_ui_fns.functions.get_modules import get_modules


async def test_get_modules():
    """Test the get_modules function."""
    print("\n" + "=" * 80)
    print("TEST: get_modules Function")
    print("=" * 80)

    try:
        # Call the function
        print("\nCalling get_modules()...")
        result = await get_modules()

        # Check result structure
        print("\n--- Result Structure ---")
        print(f"Result keys: {list(result.keys())}")

        # Verify basic structure
        assert "success" in result, "Result should have 'success' key"
        assert "result" in result, "Result should have 'result' key"
        assert "error" in result, "Result should have 'error' key"

        print(f"success: {result['success']}")
        print(f"error: {result['error']}")

        # Check if function succeeded
        if not result["success"]:
            print(f"\n[FAIL] Function returned success=False")
            print(f"Error message: {result['error']}")
            return False

        print(f"[OK] Function returned success=True")

        # Verify result is not empty
        result_json = result["result"]
        assert result_json, "Result should not be empty"
        assert len(result_json) > 0, "Result should have content"

        print(f"[OK] Result is not empty (length: {len(result_json)} chars)")

        # Parse the JSON result
        print("\n--- Parsing JSON Result ---")
        try:
            parsed_result = json.loads(result_json)
            print(f"[OK] Result is valid JSON")
        except json.JSONDecodeError as e:
            print(f"[FAIL] Result is not valid JSON: {e}")
            return False

        # Verify JSON structure
        print("\n--- JSON Structure ---")
        print(f"JSON keys: {list(parsed_result.keys())}")

        assert "module_names" in parsed_result, "JSON should have 'module_names' key"
        assert "total_count" in parsed_result, "JSON should have 'total_count' key"
        assert "description" in parsed_result, "JSON should have 'description' key"

        print(f"[OK] JSON has required keys")

        # Get module names
        module_names = parsed_result["module_names"]
        total_count = parsed_result["total_count"]
        description = parsed_result["description"]

        print(f"\nDescription: {description}")
        print(f"Total count: {total_count}")

        # Verify module_names is a list
        assert isinstance(module_names, list), "module_names should be a list"
        print(f"[OK] module_names is a list")

        # Verify count matches
        assert len(module_names) == total_count, "Length of module_names should match total_count"
        print(f"[OK] Length matches total_count ({total_count})")

        # Verify list is not empty
        assert len(module_names) > 0, "module_names should not be empty"
        print(f"[OK] Found {len(module_names)} modules")

        # Display sample module names
        print("\n--- Sample Module Names ---")
        sample_count = min(10, len(module_names))
        for i, module_name in enumerate(module_names[:sample_count], 1):
            print(f"  {i}. {module_name}")

        if len(module_names) > sample_count:
            print(f"  ... and {len(module_names) - sample_count} more modules")

        # Verify module names are strings
        print("\n--- Module Name Validation ---")
        for module_name in module_names:
            assert isinstance(module_name, str), f"Module name should be string, got {type(module_name)}"
            assert len(module_name) > 0, "Module name should not be empty"

        print(f"[OK] All {len(module_names)} module names are valid strings")

        # Display full formatted JSON (optional)
        print("\n--- Full JSON Result ---")
        print(json.dumps(parsed_result, indent=2))

        print("\n" + "=" * 80)
        print("[OK] ALL CHECKS PASSED!")
        print("=" * 80 + "\n")

        return True

    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        print("=" * 80 + "\n")
        return False

    except Exception as e:
        print(f"\n[FAIL] Test failed with exception: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        print("=" * 80 + "\n")
        return False


def run_test():
    """Run the test using asyncio."""
    print("\n" + "=" * 80)
    print("RUNNING get_modules() TESTS")
    print("=" * 80)

    try:
        success = asyncio.run(test_get_modules())

        if success:
            print("\n[OK] Test completed successfully!")
            return 0
        else:
            print("\n[FAIL] Test failed!")
            return 1

    except Exception as e:
        print(f"\n[FAIL] Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_test()
    sys.exit(exit_code)
