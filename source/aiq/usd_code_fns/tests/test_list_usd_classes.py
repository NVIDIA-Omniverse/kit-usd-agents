## Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Test suite for get_usd_classes function - List USD Classes tool tests."""

import json
import logging
import sys
from pathlib import Path

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omni_aiq_usd_code.config import USD_ATLAS_FILE_PATH
from omni_aiq_usd_code.functions.get_usd_classes import get_usd_classes

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Expected well-known USD classes that should always exist
EXPECTED_CORE_CLASSES = [
    "Usd.Stage",
    "Usd.Prim",
    "Usd.Attribute",
    "UsdGeom.Mesh",
    "UsdGeom.Xform",
    "UsdShade.Material",
    "UsdLux.DomeLight",
    "Sdf.Path",
    "Gf.Vec3d",
]


def skip_if_atlas_unavailable():
    """Helper to skip tests if USD Atlas data is not available."""
    assert USD_ATLAS_FILE_PATH.exists(), f"USD Atlas data not found at {USD_ATLAS_FILE_PATH}"


# =============================================================================
# Test: Response Structure
# =============================================================================
@pytest.mark.asyncio
async def test_list_classes_response_structure():
    """Test that list_usd_classes returns correct response structure."""
    print(f"\n{'='*60}")
    print("TEST: Response Structure")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_classes()

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
# Test: Contains Expected Core Classes
# =============================================================================
@pytest.mark.asyncio
async def test_list_classes_contains_core_classes():
    """Test that the result contains expected core USD classes."""
    print(f"\n{'='*60}")
    print("TEST: Contains Expected Core Classes")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_classes()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check that class_full_names list exists
    assert "class_full_names" in parsed_result, "Result should contain 'class_full_names' key"
    class_full_names = parsed_result["class_full_names"]
    assert isinstance(class_full_names, list), "class_full_names should be a list"

    # Check for expected core classes
    missing_classes = []
    for expected_class in EXPECTED_CORE_CLASSES:
        if expected_class not in class_full_names:
            missing_classes.append(expected_class)

    assert len(missing_classes) == 0, f"Missing expected core classes: {missing_classes}"

    print(f"[OK] Found all expected core classes: {EXPECTED_CORE_CLASSES}")
    print(f"[OK] Total classes in result: {len(class_full_names)}")


# =============================================================================
# Test: Total Count is Reasonable
# =============================================================================
@pytest.mark.asyncio
async def test_list_classes_total_count():
    """Test that the total class count is reasonable."""
    print(f"\n{'='*60}")
    print("TEST: Total Count is Reasonable")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_classes()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check total_count exists
    assert "total_count" in parsed_result, "Result should contain 'total_count'"
    total_count = parsed_result["total_count"]

    # Based on MCP output, we expect around 870 classes
    assert total_count >= 500, f"Expected at least 500 classes, got {total_count}"
    assert total_count <= 2000, f"Expected at most 2000 classes, got {total_count}"

    # Also check that the list length matches total_count
    class_full_names = parsed_result["class_full_names"]
    assert (
        len(class_full_names) == total_count
    ), f"List length {len(class_full_names)} should match total_count {total_count}"

    print(f"[OK] Total class count: {total_count} (within expected range 500-2000)")


# =============================================================================
# Test: Classes are Sorted
# =============================================================================
@pytest.mark.asyncio
async def test_list_classes_are_sorted():
    """Test that class names are sorted alphabetically."""
    print(f"\n{'='*60}")
    print("TEST: Classes are Sorted")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_classes()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    class_full_names = parsed_result["class_full_names"]

    # Check if list is sorted
    sorted_names = sorted(class_full_names)
    assert class_full_names == sorted_names, "Class names should be sorted alphabetically"

    print(f"[OK] All {len(class_full_names)} class names are sorted alphabetically")


# =============================================================================
# Test: Class Names Follow Naming Convention
# =============================================================================
@pytest.mark.asyncio
async def test_list_classes_naming_convention():
    """Test that class names follow expected naming convention (Module.ClassName)."""
    print(f"\n{'='*60}")
    print("TEST: Class Names Follow Naming Convention")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_classes()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    class_full_names = parsed_result["class_full_names"]

    # Check naming convention: most classes should have Module.ClassName format
    properly_formatted = 0
    for class_name in class_full_names:
        if "." in class_name:
            properly_formatted += 1

    # At least 90% should follow the convention
    percentage = (properly_formatted / len(class_full_names)) * 100
    assert percentage >= 90, f"Expected at least 90% classes with Module.ClassName format, got {percentage:.1f}%"

    print(f"[OK] {percentage:.1f}% of classes follow Module.ClassName naming convention")


# =============================================================================
# Test: Specific UsdGeom Classes Present
# =============================================================================
@pytest.mark.asyncio
async def test_list_classes_usdgeom_classes():
    """Test that specific UsdGeom classes are present."""
    print(f"\n{'='*60}")
    print("TEST: Specific UsdGeom Classes Present")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_classes()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    class_full_names = parsed_result["class_full_names"]

    # Expected UsdGeom classes
    expected_usdgeom_classes = [
        "UsdGeom.Mesh",
        "UsdGeom.Xform",
        "UsdGeom.Camera",
        "UsdGeom.Sphere",
        "UsdGeom.Cube",
        "UsdGeom.Cone",
        "UsdGeom.Cylinder",
        "UsdGeom.BBoxCache",
        "UsdGeom.Points",
    ]

    missing = [c for c in expected_usdgeom_classes if c not in class_full_names]
    assert len(missing) == 0, f"Missing UsdGeom classes: {missing}"

    print(f"[OK] All expected UsdGeom classes found: {expected_usdgeom_classes}")


# =============================================================================
# Test: Specific UsdShade Classes Present
# =============================================================================
@pytest.mark.asyncio
async def test_list_classes_usdshade_classes():
    """Test that specific UsdShade classes are present."""
    print(f"\n{'='*60}")
    print("TEST: Specific UsdShade Classes Present")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_classes()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    class_full_names = parsed_result["class_full_names"]

    # Expected UsdShade classes
    expected_usdshade_classes = [
        "UsdShade.Material",
        "UsdShade.Shader",
        "UsdShade.Input",
        "UsdShade.Output",
        "UsdShade.NodeGraph",
        "UsdShade.ConnectableAPI",
    ]

    missing = [c for c in expected_usdshade_classes if c not in class_full_names]
    assert len(missing) == 0, f"Missing UsdShade classes: {missing}"

    print(f"[OK] All expected UsdShade classes found: {expected_usdshade_classes}")


# =============================================================================
# Main entry point for running tests directly
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests manually."""
        print("\n" + "=" * 80)
        print("USD List Classes Tests")
        print("=" * 80)

        await test_list_classes_response_structure()
        await test_list_classes_contains_core_classes()
        await test_list_classes_total_count()
        await test_list_classes_are_sorted()
        await test_list_classes_naming_convention()
        await test_list_classes_usdgeom_classes()
        await test_list_classes_usdshade_classes()

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    asyncio.run(run_all_tests())
