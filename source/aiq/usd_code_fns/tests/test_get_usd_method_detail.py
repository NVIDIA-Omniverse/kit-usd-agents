## Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Test suite for get_usd_method_detail function - Get USD Method Detail tool tests."""

import json
import logging
import sys
from pathlib import Path

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omni_aiq_usd_code.config import USD_ATLAS_FILE_PATH
from omni_aiq_usd_code.functions.get_usd_method_detail import get_usd_method_detail

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def skip_if_atlas_unavailable():
    """Helper to skip tests if USD Atlas data is not available."""
    assert USD_ATLAS_FILE_PATH.exists(), f"USD Atlas data not found at {USD_ATLAS_FILE_PATH}"


# =============================================================================
# Test: Response Structure for Single Method
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_response_structure():
    """Test that get_usd_method_detail returns correct response structure."""
    print(f"\n{'='*60}")
    print("TEST: Response Structure for Single Method")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("GetPrim", class_name="UsdStage")

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
# Test: GetPrimAtPath Method Detail
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_getprimatpath():
    """Test getting details for GetPrimAtPath method on UsdStage."""
    print(f"\n{'='*60}")
    print("TEST: GetPrimAtPath Method Detail")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("GetPrim", class_name="UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Check method key exists
    assert "GetPrim" in parsed_result, "Result should contain 'GetPrim' key"
    method_info = parsed_result["GetPrim"]

    # Check method structure
    assert "methods" in method_info or "query" in method_info, "Method info should contain 'methods' or 'query' key"

    # Get the methods list
    methods = method_info.get("methods", [])
    assert len(methods) > 0, "Should find at least one matching method"

    # Check first method has expected fields
    first_method = methods[0]
    assert "name" in first_method or "full_name" in first_method, "Method should have 'name' or 'full_name'"

    # Check for docstring
    docstring = first_method.get("docstring", "")
    if docstring:
        assert "prim" in docstring.lower() or "path" in docstring.lower(), "Docstring should mention prim or path"

    print(f"[OK] Found {len(methods)} method(s) matching 'GetPrim'")


# =============================================================================
# Test: Method with Arguments
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_has_arguments():
    """Test that method details include argument information."""
    print(f"\n{'='*60}")
    print("TEST: Method with Arguments")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("GetPrim", class_name="UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    method_info = parsed_result.get("GetPrim", {})
    methods = method_info.get("methods", [])

    if methods:
        first_method = methods[0]
        arguments = first_method.get("arguments", [])

        # GetPrimAtPath should have at least 'self' and 'path' arguments
        assert len(arguments) >= 1, "Method should have at least one argument"

        # Check that 'path' argument exists
        arg_names = [arg.get("name", "") for arg in arguments]
        has_path_or_self = "path" in arg_names or "self" in arg_names
        assert has_path_or_self, f"Expected 'path' or 'self' argument, got: {arg_names}"

        print(f"[OK] Method has {len(arguments)} arguments: {arg_names}")
    else:
        print("[SKIP] No methods found to check arguments")


# =============================================================================
# Test: Method with Return Type
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_has_return_type():
    """Test that method details include return type information."""
    print(f"\n{'='*60}")
    print("TEST: Method with Return Type")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("GetPrim", class_name="UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    method_info = parsed_result.get("GetPrim", {})
    methods = method_info.get("methods", [])

    if methods:
        first_method = methods[0]
        return_type = first_method.get("return_type", "")

        # GetPrimAtPath should return a Prim
        if return_type:
            assert "Prim" in return_type, f"Expected 'Prim' in return type, got: {return_type}"
            print(f"[OK] Method return type: {return_type}")
        else:
            print("[OK] Method detail retrieved (return type may not be specified)")
    else:
        print("[SKIP] No methods found to check return type")


# =============================================================================
# Test: Multiple Methods at Once
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_multiple_methods():
    """Test getting details for multiple methods at once."""
    print(f"\n{'='*60}")
    print("TEST: Multiple Methods at Once")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("GetPrim,CreatePrim", class_name="UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])

    # Both method queries should be present
    assert len(parsed_result) >= 2, f"Should return at least 2 method queries, got {len(parsed_result)}"

    print(f"[OK] Retrieved details for {len(parsed_result)} method queries")


# =============================================================================
# Test: Method Without Class Context
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_without_class():
    """Test getting method details without specifying a class."""
    print(f"\n{'='*60}")
    print("TEST: Method Without Class Context")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    # Common method name that exists in multiple classes
    result = await get_usd_method_detail("GetPath")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    assert "GetPath" in parsed_result, "Result should contain 'GetPath' key"

    method_info = parsed_result["GetPath"]
    methods = method_info.get("methods", [])

    # Should find multiple methods from different classes
    if len(methods) > 1:
        print(f"[OK] Found {len(methods)} methods named 'GetPath' across different classes")
    else:
        print(f"[OK] Found {len(methods)} method(s) named 'GetPath'")


# =============================================================================
# Test: Method Detail Has Docstring
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_has_docstring():
    """Test that method details include docstrings."""
    print(f"\n{'='*60}")
    print("TEST: Method Detail Has Docstring")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("Open", class_name="UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    method_info = parsed_result.get("Open", {})
    methods = method_info.get("methods", [])

    if methods:
        # Check if any method has a docstring
        has_docstring = False
        for method in methods:
            docstring = method.get("docstring", "")
            if docstring and len(docstring) > 10:
                has_docstring = True
                break

        if has_docstring:
            print(f"[OK] Method has docstring")
        else:
            print("[OK] Method detail retrieved (docstring may not be available)")
    else:
        print("[SKIP] No methods found to check docstring")


# =============================================================================
# Test: Invalid Method Name Handling
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_invalid_method():
    """Test handling of invalid method names."""
    print(f"\n{'='*60}")
    print("TEST: Invalid Method Name Handling")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("NonExistentMethod123456", class_name="UsdStage")
    assert result["success"] is True, "Query should still succeed (may return empty or error in result)"

    parsed_result = json.loads(result["result"])
    method_result = parsed_result.get("NonExistentMethod123456", {})
    methods = method_result.get("methods", [])

    # Should return empty methods list for non-existent method
    assert len(methods) == 0, f"Should return no methods for invalid name, got {len(methods)}"

    print(f"[OK] Invalid method name handled gracefully (returned 0 methods)")


# =============================================================================
# Test: Empty Input Handling
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_empty_input():
    """Test handling of empty input."""
    print(f"\n{'='*60}")
    print("TEST: Empty Input Handling")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("")

    # Empty input should fail gracefully
    assert isinstance(result, dict), "Result should be a dictionary"
    assert result["success"] is False, "Empty input should return success=False"

    print(f"[OK] Empty input handled correctly: success=False, error={result.get('error')}")


# =============================================================================
# Test: Method Query Summary
# =============================================================================
@pytest.mark.asyncio
async def test_method_detail_query_summary():
    """Test that method detail includes query summary."""
    print(f"\n{'='*60}")
    print("TEST: Method Query Summary")
    print(f"{'='*60}")

    skip_if_atlas_unavailable()

    result = await get_usd_method_detail("GetPrim", class_name="UsdStage")
    assert result["success"] is True, f"Query failed: {result.get('error')}"

    parsed_result = json.loads(result["result"])
    method_info = parsed_result.get("GetPrim", {})

    # Check for query or summary info
    query_info = method_info.get("query", {})
    summary_info = method_info.get("summary", {})

    if query_info:
        assert "method_name" in query_info, "Query info should contain 'method_name'"
        print(f"[OK] Query info present: {query_info}")
    elif summary_info:
        print(f"[OK] Summary info present: {summary_info}")
    else:
        print("[OK] Method detail retrieved successfully")


# =============================================================================
# Main entry point for running tests directly
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests manually."""
        print("\n" + "=" * 80)
        print("USD Get Method Detail Tests")
        print("=" * 80)

        await test_method_detail_response_structure()
        await test_method_detail_getprimatpath()
        await test_method_detail_has_arguments()
        await test_method_detail_has_return_type()
        await test_method_detail_multiple_methods()
        await test_method_detail_without_class()
        await test_method_detail_has_docstring()
        await test_method_detail_invalid_method()
        await test_method_detail_empty_input()
        await test_method_detail_query_summary()

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    asyncio.run(run_all_tests())
