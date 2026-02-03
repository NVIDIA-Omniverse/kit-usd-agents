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

"""Test the get_instructions function and list_instructions functionality."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from omni_ui_fns.functions.get_instructions import INSTRUCTION_FILES, get_instructions, list_instructions


def test_instruction_files_defined():
    """Test that instruction files are properly defined."""
    print(f"\n{'='*60}")
    print("TEST: Instruction Files Configuration")
    print(f"{'='*60}")

    print(f"Number of instruction files defined: {len(INSTRUCTION_FILES)}")
    assert len(INSTRUCTION_FILES) > 0, "Should have at least one instruction file defined"

    for name, info in INSTRUCTION_FILES.items():
        print(f"\nInstruction: {name}")
        print(f"  Filename: {info['filename']}")
        print(f"  Description: {info['description'][:50]}...")
        print(f"  Use cases: {len(info['use_cases'])} cases")

        # Validate structure
        assert "filename" in info, f"Instruction {name} should have 'filename'"
        assert "description" in info, f"Instruction {name} should have 'description'"
        assert "use_cases" in info, f"Instruction {name} should have 'use_cases'"
        assert len(info["use_cases"]) > 0, f"Instruction {name} should have at least one use case"

    print(f"\n[OK] All instruction files properly configured")


def test_instruction_files_exist():
    """Test that all defined instruction files actually exist on disk."""
    print(f"\n{'='*60}")
    print("TEST: Instruction Files Existence")
    print(f"{'='*60}")

    # Get the path to the instructions directory
    test_file_path = Path(__file__)
    functions_dir = test_file_path.parent.parent / "functions"
    instructions_dir = functions_dir.parent / "data" / "instructions"

    print(f"Instructions directory: {instructions_dir}")
    print(f"Directory exists: {instructions_dir.exists()}")

    if not instructions_dir.exists():
        print(f"[FAIL] Instructions directory not found: {instructions_dir}")
        pytest.fail(f"Instructions directory not found: {instructions_dir}")

    missing_files = []
    for name, info in INSTRUCTION_FILES.items():
        file_path = instructions_dir / info["filename"]
        exists = file_path.exists()
        status = "[OK]" if exists else "[FAIL]"
        print(f"{status} {name}: {info['filename']} - {file_path}")

        if not exists:
            missing_files.append(name)

    if missing_files:
        print(f"\n[FAIL] Missing instruction files: {', '.join(missing_files)}")
        pytest.fail(f"Missing instruction files: {', '.join(missing_files)}")
    else:
        print(f"\n[OK] All instruction files found on disk")


@pytest.mark.asyncio
async def test_list_instructions():
    """Test the list_instructions function."""
    print(f"\n{'='*60}")
    print("TEST: List Instructions Function")
    print(f"{'='*60}")

    try:
        result = await list_instructions()

        print(f"Result keys: {result.keys()}")
        print(f"Success: {result.get('success')}")

        assert result["success"] == True, "list_instructions should succeed"
        assert "result" in result, "Should have 'result' key"
        assert "instructions" in result, "Should have 'instructions' key"

        # Validate result format
        result_text = result["result"]
        print(f"\nResult text length: {len(result_text)}")
        assert len(result_text) > 0, "Result text should not be empty"
        assert "# Available OmniUI Instructions" in result_text, "Should have header"
        assert "Total instructions available:" in result_text, "Should have summary"

        # Validate instructions list
        instructions = result["instructions"]
        print(f"Number of instructions: {len(instructions)}")
        assert len(instructions) == len(INSTRUCTION_FILES), "Should list all defined instructions"

        # Print formatted output
        print(f"\n{'='*60}")
        print("LIST INSTRUCTIONS OUTPUT:")
        print(f"{'='*60}")
        print(result_text)
        print(f"{'='*60}\n")

        # Validate each instruction in list
        for inst in instructions:
            assert "name" in inst, "Each instruction should have 'name'"
            assert "description" in inst, "Each instruction should have 'description'"
            assert "use_cases" in inst, "Each instruction should have 'use_cases'"
            print(f"[OK] Instruction '{inst['name']}' properly formatted")

        print(f"\n[OK] list_instructions test passed!")

    except Exception as e:
        print(f"[FAIL] list_instructions test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_get_instructions_with_valid_names():
    """Test get_instructions with all valid instruction names."""
    print(f"\n{'='*60}")
    print("TEST: Get Instructions with Valid Names")
    print(f"{'='*60}")

    for name in INSTRUCTION_FILES.keys():
        print(f"\n--- Testing instruction: {name} ---")

        try:
            result = await get_instructions(name=name)

            print(f"Success: {result.get('success')}")
            assert result["success"] == True, f"get_instructions should succeed for '{name}'"
            assert "result" in result, f"Should have 'result' key for '{name}'"
            assert "metadata" in result, f"Should have 'metadata' key for '{name}'"

            # Validate result content
            result_text = result["result"]
            assert len(result_text) > 0, f"Result should not be empty for '{name}'"
            assert f"# OmniUI Instruction: {name}" in result_text, f"Should have header for '{name}'"
            assert "## Description" in result_text, "Should have description section"
            assert "## Use Cases" in result_text, "Should have use cases section"

            # Validate metadata
            metadata = result["metadata"]
            print(f"Metadata keys: {list(metadata.keys())}")
            assert "name" in metadata, "Metadata should have 'name'"
            assert "description" in metadata, "Metadata should have 'description'"
            assert "use_cases" in metadata, "Metadata should have 'use_cases'"
            assert "filename" in metadata, "Metadata should have 'filename'"
            assert "content_length" in metadata, "Metadata should have 'content_length'"
            assert "line_count" in metadata, "Metadata should have 'line_count'"

            print(f"  Content length: {metadata['content_length']}")
            print(f"  Line count: {metadata['line_count']}")
            print(f"  Filename: {metadata['filename']}")

            assert metadata["content_length"] > 0, f"Content should not be empty for '{name}'"
            assert metadata["line_count"] > 0, f"Should have at least one line for '{name}'"

            # Print first 500 chars of result
            print(f"\n  Result preview (first 500 chars):")
            print(f"  {result_text[:500]}...")

            print(f"[OK] Instruction '{name}' retrieved successfully")

        except Exception as e:
            print(f"[FAIL] Failed to get instruction '{name}': {e}")
            raise

    print(f"\n[OK] All valid instructions retrieved successfully!")


@pytest.mark.asyncio
async def test_get_instructions_invalid_name():
    """Test get_instructions with an invalid instruction name."""
    print(f"\n{'='*60}")
    print("TEST: Get Instructions with Invalid Name")
    print(f"{'='*60}")

    invalid_name = "nonexistent_instruction"
    print(f"Testing with invalid name: '{invalid_name}'")

    try:
        result = await get_instructions(name=invalid_name)

        print(f"Result keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == False, "Should fail for invalid name"
        assert "error" in result, "Should have 'error' key"
        assert result["result"] is None, "Result should be None for invalid name"

        error_msg = result["error"]
        assert "Unknown instruction name" in error_msg, "Error should mention unknown instruction"
        assert invalid_name in error_msg, "Error should include the invalid name"
        assert "Available instructions:" in error_msg, "Error should list available instructions"

        print(f"\nError message: {error_msg}")
        print(f"[OK] Invalid name properly rejected with helpful error message")

    except Exception as e:
        print(f"[FAIL] Invalid name test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_get_instructions_output_format():
    """Test that get_instructions output format is correct and complete."""
    print(f"\n{'='*60}")
    print("TEST: Get Instructions Output Format")
    print(f"{'='*60}")

    # Test with one instruction
    test_name = "agent_system"
    print(f"Testing output format for: {test_name}")

    try:
        result = await get_instructions(name=test_name)

        assert result["success"] == True, "Should succeed"

        result_text = result["result"]
        metadata = result["metadata"]

        # Check header format
        assert result_text.startswith(f"# OmniUI Instruction: {test_name}"), "Should start with header"

        # Check sections exist
        sections = ["## Description", "## Use Cases", "---"]
        for section in sections:
            assert section in result_text, f"Should contain section: {section}"
            print(f"[OK] Found section: {section}")

        # Check use cases are listed
        use_cases = INSTRUCTION_FILES[test_name]["use_cases"]
        for use_case in use_cases:
            # Use cases should appear in the formatted output
            assert use_case in result_text, f"Use case should be in output: {use_case}"

        # Check description is included
        description = INSTRUCTION_FILES[test_name]["description"]
        assert description in result_text, "Description should be in output"

        # Print full output
        print(f"\n{'='*60}")
        print("FULL OUTPUT:")
        print(f"{'='*60}")
        print(result_text[:1000])  # Print first 1000 chars
        print(f"\n... (total length: {len(result_text)} chars)")
        print(f"{'='*60}\n")

        print(f"[OK] Output format is correct and complete")

    except Exception as e:
        print(f"[FAIL] Output format test failed: {e}")
        raise


def run_all_tests():
    """Run all tests manually without pytest."""
    print("\n" + "=" * 80)
    print("RUNNING OMNI_UI_FNS GET_INSTRUCTIONS TESTS")
    print("=" * 80)

    try:
        # Test 1: Instruction files defined
        test_instruction_files_defined()

        # Test 2: Instruction files exist
        test_instruction_files_exist()

        # Test 3: List instructions
        print(f"\n{'='*60}")
        print("Running list_instructions test...")
        print(f"{'='*60}")
        asyncio.run(test_list_instructions())

        # Test 4: Get instructions with valid names
        print(f"\n{'='*60}")
        print("Running get_instructions with valid names test...")
        print(f"{'='*60}")
        asyncio.run(test_get_instructions_with_valid_names())

        # Test 5: Get instructions with invalid name
        print(f"\n{'='*60}")
        print("Running get_instructions with invalid name test...")
        print(f"{'='*60}")
        asyncio.run(test_get_instructions_invalid_name())

        # Test 6: Output format
        print(f"\n{'='*60}")
        print("Running output format test...")
        print(f"{'='*60}")
        asyncio.run(test_get_instructions_output_format())

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[FAIL] TESTS FAILED: {e}")
        print("=" * 80 + "\n")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
