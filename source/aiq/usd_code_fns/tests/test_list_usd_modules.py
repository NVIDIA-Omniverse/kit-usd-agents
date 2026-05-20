## Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Test suite for get_usd_modules function - List USD Modules tool tests."""

import json
import logging
import sys
from pathlib import Path

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omni_aiq_usd_code.config import USD_ATLAS_FILE_PATH
from omni_aiq_usd_code.functions.get_usd_modules import get_usd_modules

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Expected well-known USD modules that should always exist
EXPECTED_CORE_MODULES = ["Usd", "UsdGeom", "UsdShade", "UsdLux", "Sdf", "Gf", "Vt", "Ar"]


def skip_if_atlas_unavailable():
    """Helper to skip tests if USD Atlas data is not available."""
    assert USD_ATLAS_FILE_PATH.exists(), f"USD Atlas data not found at {USD_ATLAS_FILE_PATH}"


# =============================================================================
# Test: USD Atlas Data Exists
# =============================================================================
def test_usd_atlas_data_exists():
    """Test that the USD Atlas data file exists."""
    print(f"\n{'='*60}")
    print("TEST: USD Atlas Data Exists")
    print(f"{'='*60}")

    print(f"Checking USD Atlas path: {USD_ATLAS_FILE_PATH}")

    assert USD_ATLAS_FILE_PATH.exists(), f"USD Atlas data not found at {USD_ATLAS_FILE_PATH}"
    print(f"[OK] USD Atlas data exists at {USD_ATLAS_FILE_PATH}")


# =============================================================================
# Test: Response Structure
# =============================================================================
@pytest.mark.asyncio
async def test_list_modules_response_structure():
    """Test that list_usd_modules returns correct response structure."""
    print(f"\n{'='*60}")
    print("TEST: Response Structure")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_modules()

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
# Test: Contains Expected Core Modules
# =============================================================================
@pytest.mark.asyncio
async def test_list_modules_contains_core_modules():
    """Test that the result contains expected core USD modules."""
    print(f"\n{'='*60}")
    print("TEST: Contains Expected Core Modules")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_modules()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check that modules list exists
    assert "modules" in parsed_result, "Result should contain 'modules' key"
    modules = parsed_result["modules"]
    assert isinstance(modules, list), "Modules should be a list"

    # Get all module names
    module_names = [m.get("name", "") for m in modules]

    # Check for expected core modules
    missing_modules = []
    for expected_module in EXPECTED_CORE_MODULES:
        if expected_module not in module_names:
            missing_modules.append(expected_module)

    assert len(missing_modules) == 0, f"Missing expected core modules: {missing_modules}"

    print(f"[OK] Found all expected core modules: {EXPECTED_CORE_MODULES}")
    print(f"[OK] Total modules in result: {len(modules)}")


# =============================================================================
# Test: Module Structure Validation
# =============================================================================
@pytest.mark.asyncio
async def test_list_modules_module_structure():
    """Test that each module has the expected structure."""
    print(f"\n{'='*60}")
    print("TEST: Module Structure Validation")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_modules()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    modules = parsed_result["modules"]

    # Expected fields in each module
    expected_fields = ["name", "full_name", "file_path", "class_count", "function_count"]

    # Check first few modules for structure
    for module in modules[:10]:
        for field in expected_fields:
            assert field in module, f"Module should have '{field}' field, got: {list(module.keys())}"

        # Type checks
        assert isinstance(module["name"], str), "Module name should be a string"
        assert isinstance(module["class_count"], int), "class_count should be an integer"
        assert isinstance(module["function_count"], int), "function_count should be an integer"

    print(f"[OK] All modules have expected fields: {expected_fields}")


# =============================================================================
# Test: Total Count is Reasonable
# =============================================================================
@pytest.mark.asyncio
async def test_list_modules_total_count():
    """Test that the total module count is reasonable (not empty, not too few)."""
    print(f"\n{'='*60}")
    print("TEST: Total Count is Reasonable")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_modules()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check total_count exists
    assert "total_count" in parsed_result, "Result should contain 'total_count'"
    total_count = parsed_result["total_count"]

    # Based on MCP output, we expect around 90+ modules
    assert total_count >= 50, f"Expected at least 50 modules, got {total_count}"
    assert total_count <= 500, f"Expected at most 500 modules, got {total_count}"

    print(f"[OK] Total module count: {total_count} (within expected range 50-500)")


# =============================================================================
# Test: Summary Statistics Present
# =============================================================================
@pytest.mark.asyncio
async def test_list_modules_summary_statistics():
    """Test that summary statistics are included in the response."""
    print(f"\n{'='*60}")
    print("TEST: Summary Statistics Present")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_modules()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check summary exists
    assert "summary" in parsed_result, "Result should contain 'summary'"
    summary = parsed_result["summary"]

    # Check summary fields
    expected_summary_fields = ["total_modules", "total_classes"]
    for field in expected_summary_fields:
        assert field in summary, f"Summary should contain '{field}'"

    # Validate summary values
    assert summary["total_modules"] > 0, "total_modules should be positive"
    assert summary["total_classes"] > 0, "total_classes should be positive"

    print(f"[OK] Summary statistics present: {summary}")


# =============================================================================
# Test: UsdGeom Module Has Expected Classes
# =============================================================================
@pytest.mark.asyncio
async def test_list_modules_usdgeom_has_classes():
    """Test that UsdGeom module has expected classes like Mesh, Xform, etc."""
    print(f"\n{'='*60}")
    print("TEST: UsdGeom Module Has Expected Classes")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_modules()
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    modules = parsed_result["modules"]

    # Find UsdGeom module
    usdgeom_module = None
    for module in modules:
        if module.get("name") == "UsdGeom" or module.get("full_name") == "UsdGeom":
            usdgeom_module = module
            break

    assert usdgeom_module is not None, "UsdGeom module should be present"

    # Check it has classes
    class_count = usdgeom_module.get("class_count", 0)
    assert class_count > 20, f"UsdGeom should have many classes, got {class_count}"

    # Check for expected class names if available
    class_names = usdgeom_module.get("class_names", [])
    expected_classes = ["Mesh", "Xform", "Camera", "Sphere"]

    if class_names:
        for expected_class in expected_classes:
            assert expected_class in class_names, f"UsdGeom should contain {expected_class}"
        print(f"[OK] UsdGeom contains expected classes: {expected_classes}")
    else:
        print(f"[OK] UsdGeom has {class_count} classes (class names not included in summary)")


# =============================================================================
# Main entry point for running tests directly
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests manually."""
        print("\n" + "=" * 80)
        print("USD List Modules Tests")
        print("=" * 80)

        test_usd_atlas_data_exists()
        await test_list_modules_response_structure()
        await test_list_modules_contains_core_modules()
        await test_list_modules_module_structure()
        await test_list_modules_total_count()
        await test_list_modules_summary_statistics()
        await test_list_modules_usdgeom_has_classes()

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    asyncio.run(run_all_tests())
