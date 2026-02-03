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

"""Test the get_classes function."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omni_ui_fns.functions.get_classes import get_classes


async def test_get_classes_async():
    """Test the get_classes function."""
    print(f"\n{'='*60}")
    print("TEST: get_classes Function")
    print(f"{'='*60}")

    try:
        # Call the function
        print("Calling get_classes()...")
        result = await get_classes()

        # Check result structure
        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")

        if result.get("error"):
            print(f"Error: {result.get('error')}")

        # Verify success field
        if not result.get("success"):
            print(f"[FAIL] Function returned success=False")
            if result.get("error"):
                print(f"       Error message: {result.get('error')}")
            return False

        print(f"[OK] Function returned success=True")

        # Verify result field exists
        if "result" not in result:
            print(f"[FAIL] Result dictionary missing 'result' field")
            return False

        print(f"[OK] Result field exists")

        # Verify result is a JSON string
        result_data = result.get("result", "")
        if not isinstance(result_data, str):
            print(f"[FAIL] Result is not a string (type: {type(result_data)})")
            return False

        print(f"[OK] Result is a string (length: {len(result_data)})")

        # Parse the JSON
        try:
            parsed_result = json.loads(result_data)
            print(f"[OK] Result is valid JSON")
        except json.JSONDecodeError as e:
            print(f"[FAIL] Result is not valid JSON: {e}")
            return False

        # Check the structure of parsed result
        print(f"\nParsed result keys: {parsed_result.keys()}")

        # Verify expected fields
        expected_fields = ["class_full_names", "total_count", "description"]
        for field in expected_fields:
            if field not in parsed_result:
                print(f"[FAIL] Missing expected field: {field}")
                return False
            print(f"[OK] Field '{field}' present")

        # Verify class_full_names is a list
        class_names = parsed_result.get("class_full_names", [])
        if not isinstance(class_names, list):
            print(f"[FAIL] class_full_names is not a list (type: {type(class_names)})")
            return False

        print(f"[OK] class_full_names is a list")

        # Verify total_count matches list length
        total_count = parsed_result.get("total_count", 0)
        if total_count != len(class_names):
            print(f"[FAIL] total_count ({total_count}) doesn't match list length ({len(class_names)})")
            return False

        print(f"[OK] total_count matches list length: {total_count}")

        # Check if we have any classes
        if len(class_names) == 0:
            print(f"[WARN] No classes found (empty list)")
        else:
            print(f"[OK] Found {len(class_names)} classes")

            # Print sample class names
            print(f"\nSample class names (first 10):")
            for i, class_name in enumerate(class_names[:10], 1):
                print(f"  {i}. {class_name}")

            if len(class_names) > 10:
                print(f"  ... and {len(class_names) - 10} more")

        # Print full result for inspection
        print(f"\n{'='*60}")
        print("FULL RESULT JSON:")
        print(f"{'='*60}")
        print(json.dumps(parsed_result, indent=2))
        print(f"{'='*60}\n")

        print(f"[OK] All validation checks passed!")
        return True

    except Exception as e:
        print(f"[FAIL] Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_test():
    """Run the test."""
    print("\n" + "=" * 80)
    print("RUNNING GET_CLASSES TEST")
    print("=" * 80)

    try:
        success = asyncio.run(test_get_classes_async())

        if success:
            print("\n" + "=" * 80)
            print("[OK] TEST PASSED!")
            print("=" * 80 + "\n")
            return 0
        else:
            print("\n" + "=" * 80)
            print("[FAIL] TEST FAILED!")
            print("=" * 80 + "\n")
            return 1

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[FAIL] TEST FAILED WITH EXCEPTION: {e}")
        print("=" * 80 + "\n")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_test()
    sys.exit(exit_code)
