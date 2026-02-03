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

"""Test the get_style_docs function."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omni_ui_fns.functions.get_style_docs import get_style_docs


def test_data_directory_exists():
    """Test that the styles data directory exists."""
    print(f"\n{'='*60}")
    print("TEST: Styles Data Directory Existence")
    print(f"{'='*60}")

    data_dir = Path(__file__).parent.parent / "data" / "styles"
    print(f"Data directory: {data_dir}")
    print(f"Path exists: {data_dir.exists()}")

    if data_dir.exists():
        print(f"[OK] Styles data directory found")
        # List files in the directory
        files = sorted(data_dir.glob("*.md"))
        print(f"Markdown files found ({len(files)}):")
        for f in files:
            print(f"  - {f.name}")
        assert len(files) > 0, "Should have at least one markdown file"
    else:
        print(f"[FAIL] Styles data directory NOT found at: {data_dir}")


async def test_get_style_docs_none():
    """Test getting combined style documentation (sections=None)."""
    print(f"\n{'='*60}")
    print("TEST: Get Combined Style Documentation (None)")
    print(f"{'='*60}")

    try:
        result = await get_style_docs(sections=None)

        print(f"Result keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert result["error"] is None, "Error should be None on success"
        assert len(result["result"]) > 0, "Should return non-empty result"

        # Parse the JSON result
        result_data = json.loads(result["result"])

        print(f"\nResult data keys: {list(result_data.keys())}")
        print(f"Type: {result_data.get('type')}")
        print(f"Content length: {result_data.get('size')}")
        print(f"Sections included: {len(result_data.get('sections_included', []))}")

        # Validate structure
        assert result_data["type"] == "combined", "Should be 'combined' type"
        assert "content" in result_data, "Should have content"
        assert len(result_data["content"]) > 0, "Content should not be empty"
        assert "metadata" in result_data, "Should have metadata"
        assert "available_sections" in result_data["metadata"], "Metadata should have available_sections"

        # Check content preview
        content_preview = result_data["content"][:200]
        print(f"\nContent preview (first 200 chars):")
        print(f"{content_preview}...")

        print(f"\n[OK] Combined style documentation test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


async def test_get_style_docs_single_section_list():
    """Test getting a single section using list format."""
    print(f"\n{'='*60}")
    print("TEST: Get Single Section (List Format)")
    print(f"{'='*60}")

    try:
        section = "buttons"
        result = await get_style_docs(sections=[section])

        print(f"Requesting section: {section}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert result["error"] is None, "Error should be None on success"

        # Parse the JSON result
        result_data = json.loads(result["result"])

        print(f"\nResult data keys: {list(result_data.keys())}")
        print(f"Section: {result_data.get('section')}")
        print(f"File: {result_data.get('file')}")
        print(f"Content size: {result_data.get('size')}")

        # Validate structure
        assert result_data["section"] == section, f"Should return section '{section}'"
        assert "content" in result_data, "Should have content"
        assert len(result_data["content"]) > 0, "Content should not be empty"
        assert result_data["file"] == f"{section}.md", "File should match section name"
        assert "metadata" in result_data, "Should have metadata"

        # Check content preview
        content_preview = result_data["content"][:200]
        print(f"\nContent preview (first 200 chars):")
        print(f"{content_preview}...")

        print(f"\n[OK] Single section test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


async def test_get_style_docs_single_section_string():
    """Test getting a single section using string format (legacy)."""
    print(f"\n{'='*60}")
    print("TEST: Get Single Section (String Format - Legacy)")
    print(f"{'='*60}")

    try:
        section = "widgets"
        result = await get_style_docs(sections=section)

        print(f"Requesting section: {section}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert result["error"] is None, "Error should be None on success"

        # Parse the JSON result
        result_data = json.loads(result["result"])

        print(f"\nResult data keys: {list(result_data.keys())}")
        print(f"Section: {result_data.get('section')}")
        print(f"Content size: {result_data.get('size')}")

        # Validate structure
        assert result_data["section"] == section, f"Should return section '{section}'"
        assert "content" in result_data, "Should have content"
        assert len(result_data["content"]) > 0, "Content should not be empty"

        print(f"\n[OK] String format test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


async def test_get_style_docs_multiple_sections():
    """Test getting multiple sections."""
    print(f"\n{'='*60}")
    print("TEST: Get Multiple Sections")
    print(f"{'='*60}")

    try:
        sections = ["buttons", "widgets", "styling"]
        result = await get_style_docs(sections=sections)

        print(f"Requesting sections: {sections}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert result["error"] is None, "Error should be None on success"

        # Parse the JSON result
        result_data = json.loads(result["result"])

        print(f"\nResult data keys: {list(result_data.keys())}")
        print(f"Sections returned: {[k for k in result_data.keys() if k != 'metadata']}")

        # Validate structure
        for section in sections:
            assert section in result_data, f"Should have section '{section}'"
            section_data = result_data[section]
            print(f"\n  Section '{section}':")
            print(f"    File: {section_data.get('file')}")
            print(f"    Size: {section_data.get('size')}")
            assert "content" in section_data, f"Section '{section}' should have content"
            assert len(section_data["content"]) > 0, f"Section '{section}' content should not be empty"

        assert "metadata" in result_data, "Should have metadata"

        print(f"\n[OK] Multiple sections test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


async def test_get_style_docs_invalid_section():
    """Test handling of invalid section name."""
    print(f"\n{'='*60}")
    print("TEST: Invalid Section Name")
    print(f"{'='*60}")

    try:
        invalid_section = "nonexistent_section"
        result = await get_style_docs(sections=[invalid_section])

        print(f"Requesting invalid section: {invalid_section}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == False, "Function should fail with invalid section"
        assert result["error"] is not None, "Should return error message"
        assert "Unknown section" in result["error"], "Error should mention unknown section"

        print(f"Error message: {result['error']}")
        print(f"\n[OK] Invalid section handling test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


async def test_get_style_docs_empty_list():
    """Test handling of empty sections list."""
    print(f"\n{'='*60}")
    print("TEST: Empty Sections List")
    print(f"{'='*60}")

    try:
        result = await get_style_docs(sections=[])

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == False, "Function should fail with empty list"
        assert result["error"] is not None, "Should return error message"
        assert "cannot be empty" in result["error"], "Error should mention empty array"

        print(f"Error message: {result['error']}")
        print(f"\n[OK] Empty list handling test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


async def test_get_style_docs_invalid_type():
    """Test handling of invalid input type."""
    print(f"\n{'='*60}")
    print("TEST: Invalid Input Type")
    print(f"{'='*60}")

    try:
        result = await get_style_docs(sections=123)  # Invalid type

        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == False, "Function should fail with invalid type"
        assert result["error"] is not None, "Should return error message"

        print(f"Error message: {result['error']}")
        print(f"\n[OK] Invalid type handling test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


async def test_get_style_docs_all_sections():
    """Test that all available sections can be retrieved."""
    print(f"\n{'='*60}")
    print("TEST: All Available Sections")
    print(f"{'='*60}")

    try:
        # First get metadata to see available sections
        result = await get_style_docs(sections=None)
        result_data = json.loads(result["result"])
        available_sections = result_data["metadata"]["available_sections"]

        print(f"Available sections: {available_sections}")

        # Test each section individually
        for section in available_sections:
            print(f"\n  Testing section: {section}")
            result = await get_style_docs(sections=[section])

            assert result["success"] == True, f"Section '{section}' should succeed"
            result_data = json.loads(result["result"])
            assert result_data["section"] == section, f"Should return section '{section}'"
            assert len(result_data["content"]) > 0, f"Section '{section}' content should not be empty"
            print(f"    [OK] Size: {result_data['size']} bytes")

        print(f"\n[OK] All sections test passed!")

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise


def run_all_tests():
    """Run all tests manually without pytest."""
    print("\n" + "=" * 80)
    print("RUNNING OMNI_UI_FNS GET_STYLE_DOCS TESTS")
    print("=" * 80)

    try:
        # Test 1: Data directory
        test_data_directory_exists()

        # Test 2: Combined docs (None)
        print(f"\nRunning async test: Combined documentation...")
        asyncio.run(test_get_style_docs_none())

        # Test 3: Single section (list format)
        print(f"\nRunning async test: Single section (list)...")
        asyncio.run(test_get_style_docs_single_section_list())

        # Test 4: Single section (string format)
        print(f"\nRunning async test: Single section (string)...")
        asyncio.run(test_get_style_docs_single_section_string())

        # Test 5: Multiple sections
        print(f"\nRunning async test: Multiple sections...")
        asyncio.run(test_get_style_docs_multiple_sections())

        # Test 6: Invalid section
        print(f"\nRunning async test: Invalid section...")
        asyncio.run(test_get_style_docs_invalid_section())

        # Test 7: Empty list
        print(f"\nRunning async test: Empty list...")
        asyncio.run(test_get_style_docs_empty_list())

        # Test 8: Invalid type
        print(f"\nRunning async test: Invalid type...")
        asyncio.run(test_get_style_docs_invalid_type())

        # Test 9: All available sections
        print(f"\nRunning async test: All available sections...")
        asyncio.run(test_get_style_docs_all_sections())

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
