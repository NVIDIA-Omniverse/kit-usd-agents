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

"""Test suite for get_instructions function."""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from kit_fns.functions.get_instructions import INSTRUCTION_FILES, get_instructions, list_instructions


def print_result(test_name: str, success: bool, details: str = ""):
    """Print test result with standard format."""
    status = "[OK]" if success else "[FAIL]"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")
    if not success:
        print()


async def test_list_all_instructions():
    """Test listing all available instructions (None input)."""
    test_name = "List all instructions (None)"
    try:
        result = await get_instructions(None)

        # Check basic structure
        if not result.get("success"):
            print_result(test_name, False, f"Failed: {result.get('error')}")
            return False

        if "result" not in result or not result["result"]:
            print_result(test_name, False, "Missing result field")
            return False

        if "instructions" not in result:
            print_result(test_name, False, "Missing instructions field")
            return False

        # Check that all instruction sets are listed
        instructions = result["instructions"]
        expected_count = len(INSTRUCTION_FILES)

        if len(instructions) != expected_count:
            print_result(test_name, False, f"Expected {expected_count} instructions, got {len(instructions)}")
            return False

        # Verify all expected instruction names are present
        instruction_names = {inst["name"] for inst in instructions}
        expected_names = set(INSTRUCTION_FILES.keys())

        if instruction_names != expected_names:
            print_result(
                test_name, False, f"Instruction names mismatch. Expected: {expected_names}, Got: {instruction_names}"
            )
            return False

        print_result(test_name, True, f"Found {len(instructions)} instruction sets")
        return True

    except Exception as e:
        print_result(test_name, False, f"Exception: {str(e)}")
        return False


async def test_list_empty_array():
    """Test listing instructions with empty array input."""
    test_name = "List instructions (empty array)"
    try:
        result = await get_instructions([])

        if not result.get("success"):
            print_result(test_name, False, f"Failed: {result.get('error')}")
            return False

        if "instructions" not in result:
            print_result(test_name, False, "Missing instructions field")
            return False

        print_result(test_name, True, f"Empty array returns listing")
        return True

    except Exception as e:
        print_result(test_name, False, f"Exception: {str(e)}")
        return False


async def test_get_single_instruction():
    """Test retrieving a single instruction set."""
    test_name = "Get single instruction (kit_system)"
    try:
        result = await get_instructions("kit_system")

        if not result.get("success"):
            print_result(test_name, False, f"Failed: {result.get('error')}")
            return False

        if "result" not in result or not result["result"]:
            print_result(test_name, False, "Missing result field")
            return False

        if "metadata" not in result:
            print_result(test_name, False, "Missing metadata field")
            return False

        metadata = result["metadata"]

        # Check metadata structure
        required_fields = ["name", "description", "use_cases", "filename", "content_length", "line_count"]
        for field in required_fields:
            if field not in metadata:
                print_result(test_name, False, f"Missing metadata field: {field}")
                return False

        # Check that content includes the instruction
        content = result["result"]
        if "# Kit Instruction: kit_system" not in content:
            print_result(test_name, False, "Missing instruction header in content")
            return False

        print_result(test_name, True, f"Retrieved {metadata['line_count']} lines, {metadata['content_length']} chars")
        return True

    except Exception as e:
        print_result(test_name, False, f"Exception: {str(e)}")
        return False


async def test_get_multiple_instructions():
    """Test retrieving multiple instruction sets."""
    test_name = "Get multiple instructions (kit_system, testing)"
    try:
        result = await get_instructions(["kit_system", "testing"])

        if not result.get("success"):
            print_result(test_name, False, f"Failed: {result.get('error')}")
            return False

        if "result" not in result or not result["result"]:
            print_result(test_name, False, "Missing result field")
            return False

        if "metadata" not in result:
            print_result(test_name, False, "Missing metadata field")
            return False

        metadata = result["metadata"]

        # Check that both instruction sets are included
        if "instruction_sets" not in metadata:
            print_result(test_name, False, "Missing instruction_sets in metadata")
            return False

        if set(metadata["instruction_sets"]) != {"kit_system", "testing"}:
            print_result(test_name, False, f"Expected ['kit_system', 'testing'], got {metadata['instruction_sets']}")
            return False

        # Check that content includes both instructions
        content = result["result"]
        if "# Kit Instruction: kit_system" not in content:
            print_result(test_name, False, "Missing kit_system in content")
            return False

        if "# Kit Instruction: testing" not in content:
            print_result(test_name, False, "Missing testing in content")
            return False

        print_result(
            test_name, True, f"Retrieved {metadata['total_sets']} sets, {metadata['combined_length']} total chars"
        )
        return True

    except Exception as e:
        print_result(test_name, False, f"Exception: {str(e)}")
        return False


async def test_all_instruction_sets():
    """Test retrieving each instruction set individually."""
    test_name = "Get all instruction sets individually"
    all_passed = True

    for instruction_name in INSTRUCTION_FILES.keys():
        try:
            result = await get_instructions(instruction_name)

            if not result.get("success"):
                print_result(f"  {instruction_name}", False, f"Failed: {result.get('error')}")
                all_passed = False
                continue

            metadata = result.get("metadata", {})
            lines = metadata.get("line_count", "?")
            print_result(f"  {instruction_name}", True, f"{lines} lines")

        except Exception as e:
            print_result(f"  {instruction_name}", False, f"Exception: {str(e)}")
            all_passed = False

    return all_passed


async def test_invalid_instruction_name():
    """Test error handling for invalid instruction name."""
    test_name = "Invalid instruction name (nonexistent)"
    try:
        result = await get_instructions("nonexistent")

        if result.get("success"):
            print_result(test_name, False, "Should have failed for invalid name")
            return False

        if "error" not in result:
            print_result(test_name, False, "Missing error field")
            return False

        error = result["error"]
        if "Unknown instruction set" not in error:
            print_result(test_name, False, f"Unexpected error message: {error}")
            return False

        print_result(test_name, True, "Correctly rejected invalid name")
        return True

    except Exception as e:
        print_result(test_name, False, f"Exception: {str(e)}")
        return False


async def test_invalid_instruction_in_list():
    """Test error handling for invalid instruction name in list."""
    test_name = "Invalid instruction in list (kit_system, invalid)"
    try:
        result = await get_instructions(["kit_system", "invalid"])

        if result.get("success"):
            print_result(test_name, False, "Should have failed for invalid name in list")
            return False

        if "error" not in result:
            print_result(test_name, False, "Missing error field")
            return False

        print_result(test_name, True, "Correctly rejected invalid name in list")
        return True

    except Exception as e:
        print_result(test_name, False, f"Exception: {str(e)}")
        return False


async def test_invalid_type():
    """Test error handling for invalid input type."""
    test_name = "Invalid type (integer)"
    try:
        result = await get_instructions(123)

        if result.get("success"):
            print_result(test_name, False, "Should have failed for invalid type")
            return False

        if "error" not in result:
            print_result(test_name, False, "Missing error field")
            return False

        print_result(test_name, True, "Correctly rejected invalid type")
        return True

    except Exception as e:
        print_result(test_name, False, f"Exception: {str(e)}")
        return False


async def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("TESTING get_instructions function")
    print("=" * 80)
    print()

    tests = [
        (
            "List Operations",
            [
                test_list_all_instructions,
                test_list_empty_array,
            ],
        ),
        (
            "Single Instruction Retrieval",
            [
                test_get_single_instruction,
            ],
        ),
        (
            "Multiple Instructions Retrieval",
            [
                test_get_multiple_instructions,
            ],
        ),
        (
            "All Instruction Sets",
            [
                test_all_instruction_sets,
            ],
        ),
        (
            "Error Handling",
            [
                test_invalid_instruction_name,
                test_invalid_instruction_in_list,
                test_invalid_type,
            ],
        ),
    ]

    total_tests = 0
    passed_tests = 0

    for category, test_funcs in tests:
        print(f"\n{category}:")
        print("-" * 40)

        for test_func in test_funcs:
            total_tests += 1
            if await test_func():
                passed_tests += 1

    print()
    print("=" * 80)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("[OK] All tests passed!")
        print("=" * 80)
        return 0
    else:
        print(f"[FAIL] {total_tests - passed_tests} test(s) failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
