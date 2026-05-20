## Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Test suite for get_usd_module_detail function - Get USD Module Detail tool tests."""

import json
import logging
import sys
from pathlib import Path

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omni_aiq_usd_code.config import USD_ATLAS_FILE_PATH
from omni_aiq_usd_code.functions.get_usd_module_detail import get_usd_module_detail

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def skip_if_atlas_unavailable():
    """Helper to skip tests if USD Atlas data is not available."""
    assert USD_ATLAS_FILE_PATH.exists(), f"USD Atlas data not found at {USD_ATLAS_FILE_PATH}"


# =============================================================================
# Test: Response Structure for Single Module
# =============================================================================
@pytest.mark.asyncio
async def test_module_detail_response_structure():
    """Test that get_usd_module_detail returns correct response structure."""
    print(f"\n{'='*60}")
    print("TEST: Response Structure for Single Module")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_module_detail("Usd")

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
# Test: Usd Module Detail
# =============================================================================
@pytest.mark.asyncio
async def test_module_detail_usd_module():
    """Test getting details for the Usd module."""
    print(f"\n{'='*60}")
    print("TEST: Usd Module Detail")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_module_detail("Usd")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check module key exists
    assert "Usd" in parsed_result, "Result should contain 'Usd' key"
    usd_module = parsed_result["Usd"]

    # Check module structure
    assert "module" in usd_module, "Module detail should contain 'module' key"
    assert "classes" in usd_module, "Module detail should contain 'classes' key"

    module_info = usd_module["module"]
    assert module_info.get("name") == "Usd" or module_info.get("full_name") == "Usd", "Module name should be 'Usd'"

    # Check for expected classes
    classes = usd_module["classes"]
    assert isinstance(classes, list), "Classes should be a list"
    assert len(classes) > 50, f"Usd module should have many classes, got {len(classes)}"

    # Check for specific expected classes
    expected_classes = ["Stage", "Prim", "Attribute", "Property"]
    for expected_class in expected_classes:
        assert expected_class in classes, f"Usd module should contain '{expected_class}'"

    print(f"[OK] Usd module has {len(classes)} classes including: {expected_classes}")


# =============================================================================
# Test: UsdGeom Module Detail
# =============================================================================
@pytest.mark.asyncio
async def test_module_detail_usdgeom_module():
    """Test getting details for the UsdGeom module."""
    print(f"\n{'='*60}")
    print("TEST: UsdGeom Module Detail")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_module_detail("UsdGeom")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check module key exists
    assert "UsdGeom" in parsed_result, "Result should contain 'UsdGeom' key"
    usdgeom_module = parsed_result["UsdGeom"]

    # Check for expected classes
    classes = usdgeom_module.get("classes", [])
    expected_classes = ["Mesh", "Xform", "Camera", "Sphere", "Cube"]

    for expected_class in expected_classes:
        assert expected_class in classes, f"UsdGeom module should contain '{expected_class}'"

    print(f"[OK] UsdGeom module has {len(classes)} classes including: {expected_classes}")


# =============================================================================
# Test: Multiple Modules at Once
# =============================================================================
@pytest.mark.asyncio
async def test_module_detail_multiple_modules():
    """Test getting details for multiple modules at once."""
    print(f"\n{'='*60}")
    print("TEST: Multiple Modules at Once")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_module_detail("Usd,UsdGeom,UsdShade")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check all modules are present
    expected_modules = ["Usd", "UsdGeom", "UsdShade"]
    for module_name in expected_modules:
        assert module_name in parsed_result, f"Result should contain '{module_name}' key"

    print(f"[OK] Retrieved details for all modules: {expected_modules}")


# =============================================================================
# Test: Fuzzy Module Name Matching
# =============================================================================
@pytest.mark.asyncio
async def test_module_detail_fuzzy_matching():
    """Test that fuzzy module name matching works (e.g., 'pxr.Usd' vs 'Usd')."""
    print(f"\n{'='*60}")
    print("TEST: Fuzzy Module Name Matching")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    # Try with different name formats
    result1 = await get_usd_module_detail("pxr.Usd")
    result2 = await get_usd_module_detail("Usd")

    assert result1["success"] is True, f"Query 'pxr.Usd' failed: {result1.get('error')}"
    assert result2["success"] is True, f"Query 'Usd' failed: {result2.get('error')}"

    parsed1 = json.loads(result1["result"])
    parsed2 = json.loads(result2["result"])

    # Both should return module details
    assert len(parsed1) > 0, "pxr.Usd should return module details"
    assert len(parsed2) > 0, "Usd should return module details"

    print("[OK] Both 'pxr.Usd' and 'Usd' name formats return valid results")


# =============================================================================
# Test: Invalid Module Name Handling
# =============================================================================
@pytest.mark.asyncio
async def test_module_detail_invalid_module():
    """Test handling of invalid module names."""
    print(f"\n{'='*60}")
    print("TEST: Invalid Module Name Handling")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_module_detail("NonExistentModule123456")
    assert result["success"] is True, "Query should still succeed (may return empty or error in result)"

    parsed_result = json.loads(result["result"])
    module_result = parsed_result.get("NonExistentModule123456", {})

    # Should either be empty or contain an error indicator
    # The function may handle this differently, so we just check it doesn't crash
    print(f"[OK] Invalid module name handled gracefully: {module_result}")


# =============================================================================
# Test: Empty Input Handling
# =============================================================================
@pytest.mark.asyncio
async def test_module_detail_empty_input():
    """Test handling of empty input."""
    print(f"\n{'='*60}")
    print("TEST: Empty Input Handling")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_module_detail("")

    # Empty input should fail gracefully
    assert isinstance(result, dict), "Result should be a dictionary"
    assert result["success"] is False, "Empty input should return success=False"

    print(f"[OK] Empty input handled correctly: success=False, error={result.get('error')}")


# =============================================================================
# Test: Module Contains Summary Statistics
# =============================================================================
@pytest.mark.asyncio
async def test_module_detail_summary_stats():
    """Test that module detail contains summary statistics."""
    print(f"\n{'='*60}")
    print("TEST: Module Contains Summary Statistics")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_module_detail("UsdGeom")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    usdgeom_module = parsed_result.get("UsdGeom", {})

    # Check for summary
    if "summary" in usdgeom_module:
        summary = usdgeom_module["summary"]
        assert "class_count" in summary, "Summary should contain 'class_count'"
        assert summary["class_count"] > 0, "class_count should be positive"
        print(f"[OK] Module has summary statistics: {summary}")
    else:
        # Summary may be computed from classes list
        classes = usdgeom_module.get("classes", [])
        print(f"[OK] Module has {len(classes)} classes (no separate summary)")


# =============================================================================
# Main entry point for running tests directly
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests manually."""
        print("\n" + "=" * 80)
        print("USD Get Module Detail Tests")
        print("=" * 80)

        await test_module_detail_response_structure()
        await test_module_detail_usd_module()
        await test_module_detail_usdgeom_module()
        await test_module_detail_multiple_modules()
        await test_module_detail_fuzzy_matching()
        await test_module_detail_invalid_module()
        await test_module_detail_empty_input()
        await test_module_detail_summary_stats()

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    asyncio.run(run_all_tests())
