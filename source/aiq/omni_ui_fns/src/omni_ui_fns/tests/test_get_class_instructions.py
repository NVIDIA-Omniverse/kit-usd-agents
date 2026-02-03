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

"""Test the get_class_instructions function and class instruction retrieval."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from omni_ui_fns.functions.get_class_instructions import (
    CATEGORIES,
    find_class_file,
    get_class_instructions,
    get_classes_directory,
    list_class_categories,
    list_classes_in_category,
    normalize_class_name,
)


def test_classes_directory_exists():
    """Test that the classes directory exists."""
    print(f"\n{'='*60}")
    print("TEST: Classes Directory Existence")
    print(f"{'='*60}")

    classes_dir = get_classes_directory()
    print(f"Classes directory path: {classes_dir}")
    print(f"Path exists: {classes_dir.exists()}")

    if classes_dir.exists():
        print(f"[OK] Classes directory found at: {classes_dir}")
        # List categories (subdirectories)
        categories = [d.name for d in classes_dir.iterdir() if d.is_dir()]
        print(f"Categories found: {', '.join(categories)}")

        # Check for some expected files
        for category in ["widgets", "shapes", "containers"]:
            category_dir = classes_dir / category
            if category_dir.exists():
                files = list(category_dir.glob("*.md"))
                print(f"  {category}: {len(files)} files")
    else:
        print(f"[FAIL] Classes directory NOT found at: {classes_dir}")
        pytest.fail("Classes directory not found")

    assert classes_dir.exists(), "Classes directory must exist"


def test_normalize_class_name():
    """Test class name normalization."""
    print(f"\n{'='*60}")
    print("TEST: Class Name Normalization")
    print(f"{'='*60}")

    test_cases = [
        ("Button", ("Button", None)),
        ("omni.ui.Button", ("Button", None)),
        ("scene.Line", ("Line", "scene")),
        ("omni.ui.scene.Line", ("Line", "scene")),
        ("TreeView", ("TreeView", None)),
    ]

    all_passed = True
    for input_name, expected in test_cases:
        result = normalize_class_name(input_name)
        passed = result == expected
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} normalize_class_name('{input_name}') -> {result}")
        if not passed:
            print(f"     Expected: {expected}")
            all_passed = False
        assert result == expected, f"Failed for {input_name}"

    if all_passed:
        print(f"[OK] All normalization tests passed")


def test_find_class_file():
    """Test finding class files."""
    print(f"\n{'='*60}")
    print("TEST: Find Class File")
    print(f"{'='*60}")

    classes_dir = get_classes_directory()

    # Test cases: (class_name, should_find, expected_category)
    test_cases = [
        ("Button", True, "widgets"),
        ("omni.ui.Button", True, "widgets"),
        ("TreeView", True, "widgets"),
        ("Circle", True, "shapes"),
        ("scene.Line", True, "scene"),
        ("omni.ui.scene.Line", True, "scene"),
        ("VStack", True, "containers"),
        ("NonExistentClass", False, None),
    ]

    all_passed = True
    for class_name, should_find, expected_category in test_cases:
        result = find_class_file(class_name, classes_dir)

        if should_find:
            if result:
                file_path, category = result
                passed = category == expected_category and file_path.exists()
                status = "[OK]" if passed else "[FAIL]"
                print(f"{status} Found '{class_name}' in category '{category}' at {file_path.name}")
                if not passed:
                    print(f"     Expected category: {expected_category}, got: {category}")
                    all_passed = False
                assert result is not None, f"Should find {class_name}"
                assert category == expected_category, f"Wrong category for {class_name}"
            else:
                print(f"[FAIL] Could not find '{class_name}' (expected in {expected_category})")
                all_passed = False
                assert False, f"Should find {class_name}"
        else:
            passed = result is None
            status = "[OK]" if passed else "[FAIL]"
            print(f"{status} '{class_name}' not found (as expected)")
            if not passed:
                all_passed = False
            assert result is None, f"Should not find {class_name}"

    if all_passed:
        print(f"[OK] All file finding tests passed")


@pytest.mark.asyncio
async def test_list_class_categories():
    """Test listing all class categories."""
    print(f"\n{'='*60}")
    print("TEST: List Class Categories")
    print(f"{'='*60}")

    result = await list_class_categories()

    print(f"Success: {result.get('success')}")
    print(f"Result keys: {result.keys()}")

    assert result["success"] == True, "Should succeed"
    assert "result" in result, "Should have result field"
    assert "categories" in result, "Should have categories field"

    # Check that all categories are listed
    categories_info = result["categories"]
    print(f"[OK] Found {len(categories_info)} categories")

    for cat in categories_info:
        print(f"  - {cat['name']}: {cat['class_count']} classes")

    assert len(categories_info) == len(CATEGORIES), "Should list all categories"

    # Print formatted result
    print(f"\n{'='*60}")
    print("FORMATTED RESULT:")
    print(f"{'='*60}")
    print(result["result"])
    print(f"{'='*60}\n")


@pytest.mark.asyncio
async def test_list_classes_in_category():
    """Test listing classes in a specific category."""
    print(f"\n{'='*60}")
    print("TEST: List Classes in Category")
    print(f"{'='*60}")

    # Test valid category
    result = await list_classes_in_category("widgets")

    print(f"Success: {result.get('success')}")
    print(f"Category: {result.get('category')}")
    print(f"Class count: {len(result.get('classes', []))}")

    assert result["success"] == True, "Should succeed for valid category"
    assert result["category"] == "widgets", "Should return widgets category"
    assert len(result["classes"]) > 0, "Should have classes"

    print(f"[OK] Found {len(result['classes'])} classes in widgets category")
    print(f"Classes: {', '.join(result['classes'][:5])}...")

    # Test invalid category
    result_invalid = await list_classes_in_category("invalid_category")
    print(f"\n[OK] Invalid category handled: success={result_invalid['success']}")
    assert result_invalid["success"] == False, "Should fail for invalid category"
    assert "error" in result_invalid, "Should have error message"


@pytest.mark.asyncio
async def test_get_single_class_instruction():
    """Test retrieving a single class instruction."""
    print(f"\n{'='*60}")
    print("TEST: Get Single Class Instruction")
    print(f"{'='*60}")

    # Test with a known class
    class_name = "Button"
    print(f"Retrieving instructions for: {class_name}")

    result = await get_class_instructions([class_name])

    print(f"Success: {result.get('success')}")
    print(f"Result keys: {result.keys()}")

    assert result["success"] == True, f"Should succeed. Error: {result.get('error')}"
    assert "result" in result, "Should have result field"
    assert "metadata" in result, "Should have metadata field"

    content = result["result"]
    metadata = result["metadata"]

    print(f"[OK] Retrieved {len(content)} characters")
    print(f"Metadata:")
    print(f"  - Class name: {metadata.get('class_name')}")
    print(f"  - Normalized name: {metadata.get('normalized_name')}")
    print(f"  - Category: {metadata.get('category')}")
    print(f"  - Line count: {metadata.get('line_count')}")

    # Verify content is not empty
    assert len(content) > 0, "Content should not be empty"
    assert metadata["category"] == "widgets", "Button should be in widgets category"

    # Print first 500 characters of content
    print(f"\n{'='*60}")
    print("CONTENT PREVIEW (first 500 chars):")
    print(f"{'='*60}")
    print(content[:500])
    print(f"{'='*60}\n")


@pytest.mark.asyncio
async def test_get_multiple_class_instructions():
    """Test retrieving multiple class instructions."""
    print(f"\n{'='*60}")
    print("TEST: Get Multiple Class Instructions")
    print(f"{'='*60}")

    # Test with multiple classes
    class_names = ["Button", "Label", "TreeView"]
    print(f"Retrieving instructions for: {', '.join(class_names)}")

    result = await get_class_instructions(class_names)

    print(f"Success: {result.get('success')}")
    print(f"Result keys: {result.keys()}")

    assert result["success"] == True, f"Should succeed. Error: {result.get('error')}"
    assert "result" in result, "Should have result field"
    assert "metadata" in result, "Should have metadata field"

    content = result["result"]
    metadata = result["metadata"]

    print(f"[OK] Retrieved {len(content)} characters")
    print(f"Metadata:")
    print(f"  - Total requested: {metadata.get('total_requested')}")
    print(f"  - Successful: {metadata.get('successful')}")
    print(f"  - Failed: {metadata.get('failed')}")

    assert metadata["total_requested"] == 3, "Should request 3 classes"
    assert metadata["successful"] == 3, "Should retrieve all 3 classes"
    assert metadata["failed"] == 0, "Should have no failures"

    # Check that content contains headers for each class
    for class_name in class_names:
        assert f"# Class: {class_name}" in content, f"Should contain header for {class_name}"

    # Print classes info
    print(f"\nClasses retrieved:")
    for cls_meta in metadata["classes"]:
        print(f"  - {cls_meta['class_name']}: {cls_meta['line_count']} lines from {cls_meta['category']}")


@pytest.mark.asyncio
async def test_get_scene_class_instruction():
    """Test retrieving scene class instructions."""
    print(f"\n{'='*60}")
    print("TEST: Get Scene Class Instruction")
    print(f"{'='*60}")

    # Test different ways to reference scene classes
    test_cases = [
        "scene.Line",
        "omni.ui.scene.Line",
        "Line",  # Should find scene.Line
    ]

    for class_name in test_cases:
        print(f"\nTrying: {class_name}")
        result = await get_class_instructions([class_name])

        print(f"  Success: {result.get('success')}")

        if result["success"]:
            metadata = result["metadata"]
            print(f"  [OK] Found in category: {metadata.get('category')}")
            print(f"  Normalized name: {metadata.get('normalized_name')}")

            # At least one of these should find the scene.Line class
            if metadata.get("category") == "scene":
                print(f"  [OK] Successfully retrieved scene class")
                break
        else:
            print(f"  [FAIL] Error: {result.get('error')}")


@pytest.mark.asyncio
async def test_invalid_class_name():
    """Test handling of invalid class names."""
    print(f"\n{'='*60}")
    print("TEST: Invalid Class Name Handling")
    print(f"{'='*60}")

    # Test with non-existent class
    result = await get_class_instructions(["NonExistentClass"])

    print(f"Success: {result.get('success')}")
    print(f"Error: {result.get('error', 'None')[:200]}...")

    assert result["success"] == False, "Should fail for non-existent class"
    assert "error" in result, "Should have error message"
    assert "not found" in result["error"].lower(), "Error should mention class not found"

    print(f"[OK] Invalid class handled correctly")


@pytest.mark.asyncio
async def test_empty_input():
    """Test handling of empty input."""
    print(f"\n{'='*60}")
    print("TEST: Empty Input Handling")
    print(f"{'='*60}")

    # Test with empty list
    result = await get_class_instructions([])

    print(f"Success: {result.get('success')}")
    print(f"Error: {result.get('error')}")

    assert result["success"] == False, "Should fail for empty list"
    assert "error" in result, "Should have error message"
    assert "empty" in result["error"].lower(), "Error should mention empty input"

    print(f"[OK] Empty input handled correctly")


@pytest.mark.asyncio
async def test_mixed_valid_invalid_classes():
    """Test handling of mixed valid and invalid class names."""
    print(f"\n{'='*60}")
    print("TEST: Mixed Valid/Invalid Classes")
    print(f"{'='*60}")

    # Mix of valid and invalid classes
    class_names = ["Button", "NonExistent1", "Label", "NonExistent2"]
    print(f"Retrieving instructions for: {', '.join(class_names)}")

    result = await get_class_instructions(class_names)

    print(f"Success: {result.get('success')}")
    print(f"Result keys: {result.keys()}")

    # Should succeed with partial results
    assert result["success"] == True, "Should succeed even with some failures"

    metadata = result["metadata"]
    print(f"Metadata:")
    print(f"  - Total requested: {metadata.get('total_requested')}")
    print(f"  - Successful: {metadata.get('successful')}")
    print(f"  - Failed: {metadata.get('failed')}")

    assert metadata["total_requested"] == 4, "Should request 4 classes"
    assert metadata["successful"] == 2, "Should retrieve 2 valid classes"
    assert metadata["failed"] == 2, "Should have 2 failures"

    # Check errors
    errors = metadata.get("errors", [])
    print(f"Errors: {errors}")
    assert len(errors) == 2, "Should have 2 error messages"

    print(f"[OK] Mixed input handled correctly")


def run_all_tests():
    """Run all tests manually without pytest."""
    print("\n" + "=" * 80)
    print("RUNNING OMNI_UI_FNS CLASS INSTRUCTIONS TESTS")
    print("=" * 80)

    try:
        # Test 1: Directory exists
        test_classes_directory_exists()

        # Test 2: Normalize class names
        test_normalize_class_name()

        # Test 3: Find class files
        test_find_class_file()

        # Test 4: List categories
        print(f"\n{'='*60}")
        print("Running list categories test...")
        print(f"{'='*60}")
        asyncio.run(test_list_class_categories())

        # Test 5: List classes in category
        print(f"\n{'='*60}")
        print("Running list classes in category test...")
        print(f"{'='*60}")
        asyncio.run(test_list_classes_in_category())

        # Test 6: Get single class
        print(f"\n{'='*60}")
        print("Running single class retrieval test...")
        print(f"{'='*60}")
        asyncio.run(test_get_single_class_instruction())

        # Test 7: Get multiple classes
        print(f"\n{'='*60}")
        print("Running multiple classes retrieval test...")
        print(f"{'='*60}")
        asyncio.run(test_get_multiple_class_instructions())

        # Test 8: Get scene class
        print(f"\n{'='*60}")
        print("Running scene class retrieval test...")
        print(f"{'='*60}")
        asyncio.run(test_get_scene_class_instruction())

        # Test 9: Invalid class
        print(f"\n{'='*60}")
        print("Running invalid class test...")
        print(f"{'='*60}")
        asyncio.run(test_invalid_class_name())

        # Test 10: Empty input
        print(f"\n{'='*60}")
        print("Running empty input test...")
        print(f"{'='*60}")
        asyncio.run(test_empty_input())

        # Test 11: Mixed valid/invalid
        print(f"\n{'='*60}")
        print("Running mixed valid/invalid test...")
        print(f"{'='*60}")
        asyncio.run(test_mixed_valid_invalid_classes())

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"[FAIL] TESTS FAILED: {e}")
        print("=" * 80 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()
