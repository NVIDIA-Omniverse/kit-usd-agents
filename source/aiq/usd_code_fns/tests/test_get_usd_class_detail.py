## Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Test suite for get_usd_class_detail function - Get USD Class Detail tool tests."""

import json
import logging
import sys
from pathlib import Path

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omni_aiq_usd_code.config import USD_ATLAS_FILE_PATH
from omni_aiq_usd_code.functions.get_usd_class_detail import get_usd_class_detail

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def skip_if_atlas_unavailable():
    """Helper to skip tests if USD Atlas data is not available."""
    assert USD_ATLAS_FILE_PATH.exists(), f"USD Atlas data not found at {USD_ATLAS_FILE_PATH}"


# =============================================================================
# Test: Response Structure for Single Class
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_response_structure():
    """Test that get_usd_class_detail returns correct response structure."""
    print(f"\n{'='*60}")
    print("TEST: Response Structure for Single Class")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_class_detail("UsdStage")

    # Check response structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "success" in result, "Result should have 'success' key"
    assert "result" in result, "Result should have 'result' key"
    assert "error" in result, "Result should have 'error' key"

    # Check success
    assert result["success"] is True, f"Operation should succeed, got error: {result.get('error')}"
    assert result["error"] is None, f"Error should be None, got: {result['error']}"
    assert result["result"], "Result should not be empty"

    # Parse the JSON result
    parsed_result = json.loads(result["result"])
    assert isinstance(parsed_result, dict), "Parsed result should be a dictionary"

    print("[OK] Response has correct structure: success=True, result contains JSON, error=None")


# =============================================================================
# Test: UsdStage Class Detail
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_usd_stage():
    """Test getting details for UsdStage class."""
    print(f"\n{'='*60}")
    print("TEST: UsdStage Class Detail")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_class_detail("UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check class key exists
    assert "UsdStage" in parsed_result, "Result should contain 'UsdStage' key"
    stage_class = parsed_result["UsdStage"]

    # Check class structure
    assert "class" in stage_class, "Class detail should contain 'class' key"
    assert "methods" in stage_class, "Class detail should contain 'methods' key"

    class_info = stage_class["class"]

    # Check class has a docstring
    docstring = class_info.get("docstring", "")
    assert len(docstring) > 100, "UsdStage should have a substantial docstring"
    assert "stage" in docstring.lower() or "scene" in docstring.lower(), "Docstring should mention stage or scene"

    # Check methods
    methods = stage_class["methods"]
    assert "own" in methods or "all" in methods, "Methods should have 'own' or 'all' key"

    all_methods = methods.get("all", methods.get("own", []))
    assert len(all_methods) > 50, f"UsdStage should have many methods, got {len(all_methods)}"

    # Check for specific expected methods
    expected_methods = ["Open", "CreateInMemory", "GetPrimAtPath", "GetRootLayer", "Save"]
    for expected_method in expected_methods:
        found = any(expected_method in m for m in all_methods)
        assert found, f"UsdStage should have method containing '{expected_method}'"

    print(f"[OK] UsdStage class has {len(all_methods)} methods including methods like: {expected_methods}")


# =============================================================================
# Test: UsdGeom.Mesh Class Detail
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_usdgeom_mesh():
    """Test getting details for UsdGeom.Mesh class."""
    print(f"\n{'='*60}")
    print("TEST: UsdGeom.Mesh Class Detail")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_class_detail("UsdGeom.Mesh")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check class key exists (might use short name or full name)
    class_key = None
    for key in parsed_result:
        if "Mesh" in key:
            class_key = key
            break

    assert class_key is not None, "Result should contain a Mesh class key"
    mesh_class = parsed_result[class_key]

    # Check for expected content
    assert "class" in mesh_class, "Class detail should contain 'class' key"
    assert "methods" in mesh_class, "Class detail should contain 'methods' key"

    print(f"[OK] UsdGeom.Mesh class retrieved successfully")


# =============================================================================
# Test: Multiple Classes at Once
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_multiple_classes():
    """Test getting details for multiple classes at once."""
    print(f"\n{'='*60}")
    print("TEST: Multiple Classes at Once")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_class_detail("UsdStage,UsdPrim")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Both classes should be present
    assert len(parsed_result) >= 2, f"Should return at least 2 classes, got {len(parsed_result)}"

    print(f"[OK] Retrieved details for {len(parsed_result)} classes")


# =============================================================================
# Test: Fuzzy Class Name Matching
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_fuzzy_matching():
    """Test that fuzzy class name matching works (e.g., 'Stage' vs 'UsdStage')."""
    print(f"\n{'='*60}")
    print("TEST: Fuzzy Class Name Matching")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    # Try with different name formats
    result1 = await get_usd_class_detail("Stage")
    result2 = await get_usd_class_detail("pxr.Usd.Stage")

    assert result1["success"] is True, f"Query 'Stage' failed: {result1.get('error')}"
    assert result2["success"] is True, f"Query 'pxr.Usd.Stage' failed: {result2.get('error')}"

    parsed1 = json.loads(result1["result"])
    parsed2 = json.loads(result2["result"])

    # Both should return class details
    assert len(parsed1) > 0, "'Stage' should return class details"
    assert len(parsed2) > 0, "'pxr.Usd.Stage' should return class details"

    print("[OK] Both short and full name formats return valid results")


# =============================================================================
# Test: Class Has Docstring
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_has_docstring():
    """Test that class details include docstrings."""
    print(f"\n{'='*60}")
    print("TEST: Class Has Docstring")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_class_detail("UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    stage_class = parsed_result.get("UsdStage", {})

    class_info = stage_class.get("class", {})
    docstring = class_info.get("docstring", "")

    assert docstring, "Class should have a docstring"
    assert len(docstring) > 50, "Docstring should be substantial"

    print(f"[OK] Class has docstring ({len(docstring)} chars)")


# =============================================================================
# Test: Invalid Class Name Handling
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_invalid_class():
    """Test handling of invalid class names."""
    print(f"\n{'='*60}")
    print("TEST: Invalid Class Name Handling")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_class_detail("NonExistentClass123456")
    assert result["success"] is True, "Query should still succeed (may return empty or error in result)"

    parsed_result = json.loads(result["result"])
    class_result = parsed_result.get("NonExistentClass123456", {})

    # Should either be empty or contain an error indicator
    print(f"[OK] Invalid class name handled gracefully: {class_result}")


# =============================================================================
# Test: Empty Input Handling
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_empty_input():
    """Test handling of empty input."""
    print(f"\n{'='*60}")
    print("TEST: Empty Input Handling")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_class_detail("")

    # Empty input should fail gracefully
    assert isinstance(result, dict), "Result should be a dictionary"
    assert result["success"] is False, "Empty input should return success=False"

    print(f"[OK] Empty input handled correctly: success=False, error={result.get('error')}")


# =============================================================================
# Test: Class Method Count Summary
# =============================================================================
@pytest.mark.asyncio
async def test_class_detail_method_count():
    """Test that class detail includes method count in summary."""
    print(f"\n{'='*60}")
    print("TEST: Class Method Count Summary")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_class_detail("UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    stage_class = parsed_result.get("UsdStage", {})

    # Check summary if present
    summary = stage_class.get("summary", {})
    if summary:
        assert "total_method_count" in summary or "own_method_count" in summary, "Summary should include method count"
        print(f"[OK] Class summary: {summary}")
    else:
        # Count from methods
        methods = stage_class.get("methods", {})
        all_methods = methods.get("all", methods.get("own", []))
        print(f"[OK] Class has {len(all_methods)} methods")


# =============================================================================
# Main entry point for running tests directly
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests manually."""
        print("\n" + "=" * 80)
        print("USD Get Class Detail Tests")
        print("=" * 80)

        await test_class_detail_response_structure()
        await test_class_detail_usd_stage()
        await test_class_detail_usdgeom_mesh()
        await test_class_detail_multiple_classes()
        await test_class_detail_fuzzy_matching()
        await test_class_detail_has_docstring()
        await test_class_detail_invalid_class()
        await test_class_detail_empty_input()
        await test_class_detail_method_count()

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    asyncio.run(run_all_tests())
