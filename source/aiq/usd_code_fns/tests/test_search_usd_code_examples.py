## Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Test suite for search_usd_code_examples tool - USD code example search and validation."""

import ast
import logging
import os
import re
import sys
from pathlib import Path

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omni_aiq_usd_code.config import FAISS_CODE_INDEX_PATH
from omni_aiq_usd_code.functions.get_usd_code_example import get_usd_code_example

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_code_from_result(result_text: str) -> list[str]:
    """Extract Python code blocks from the formatted result text.

    Args:
        result_text: The formatted result string from get_usd_code_example

    Returns:
        List of code strings extracted from code blocks
    """
    code_blocks = []

    # Match code between ```...``` blocks (with or without language tag)
    # Pattern 1: Code:\n```\n...\n```
    pattern1 = r"Code:\n```\n(.*?)\n```"
    matches1 = re.findall(pattern1, result_text, re.DOTALL)
    code_blocks.extend(matches1)

    # Pattern 2: ```python\n...\n```
    pattern2 = r"```python\n(.*?)\n```"
    matches2 = re.findall(pattern2, result_text, re.DOTALL)
    code_blocks.extend(matches2)

    # Pattern 3: ```\n...\n``` (generic code blocks)
    pattern3 = r"```\n(.*?)\n```"
    matches3 = re.findall(pattern3, result_text, re.DOTALL)
    # Only add if not already captured by other patterns
    for match in matches3:
        if match not in code_blocks:
            code_blocks.append(match)

    # Filter out non-code blocks (metadata like "Question:" or "Code:" labels without actual code)
    valid_code_blocks = []
    for code in code_blocks:
        stripped_code = code.strip()
        # Strip any embedded markdown fences from the code content itself
        # (some FAISS entries have ```python ... ``` inside their page_content)
        if stripped_code.startswith("```"):
            first_newline = stripped_code.find("\n")
            if first_newline != -1:
                stripped_code = stripped_code[first_newline + 1 :]
            if stripped_code.endswith("```"):
                stripped_code = stripped_code[:-3]
            stripped_code = stripped_code.strip()
        # Skip if it's just metadata labels or empty
        if (
            stripped_code
            and not stripped_code.startswith("Question:")
            and not stripped_code.startswith("Code:")
            and not (stripped_code.count("\n") == 0 and ":" in stripped_code and len(stripped_code) < 100)
        ):
            valid_code_blocks.append(stripped_code)

    return valid_code_blocks


# =============================================================================
# Test: FAISS Index Existence
# =============================================================================
def test_faiss_index_exists():
    """Test that the FAISS index file exists."""
    print(f"\n{'='*60}")
    print("TEST: FAISS Index Exists")
    print(f"{'='*60}")

    print(f"Checking FAISS index path: {FAISS_CODE_INDEX_PATH}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS index not found at {FAISS_CODE_INDEX_PATH}"
    print(f"[OK] FAISS index exists at {FAISS_CODE_INDEX_PATH}")


# =============================================================================
# Test: Response Structure
# =============================================================================
@pytest.mark.asyncio
async def test_response_structure():
    """Test that the response has the expected structure."""
    print(f"\n{'='*60}")
    print("TEST: Response Structure")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    result = await get_usd_code_example("How to create a USD stage?", rerank_k=3, enable_rerank=False)

    # Check response structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "success" in result, "Result should have 'success' key"
    assert "result" in result, "Result should have 'result' key"
    assert "error" in result, "Result should have 'error' key"

    print("[OK] Response has correct structure: success, result, error")


# =============================================================================
# Test: Successful Query
# =============================================================================
@pytest.mark.asyncio
async def test_successful_query():
    """Test a successful query returns valid results."""
    print(f"\n{'='*60}")
    print("TEST: Successful Query")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    result = await get_usd_code_example("How to create a USD stage?", rerank_k=3, enable_rerank=False)

    assert result["success"] is True, f"Query should succeed, got error: {result.get('error')}"
    assert result["result"], "Result should not be empty"
    assert result["error"] is None, f"Error should be None, got: {result['error']}"

    print(f"[OK] Query succeeded with result length: {len(result['result'])} chars")


# =============================================================================
# Test: Empty Query Handling
# =============================================================================
@pytest.mark.asyncio
async def test_empty_query():
    """Test handling of empty queries."""
    print(f"\n{'='*60}")
    print("TEST: Empty Query Handling")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    result = await get_usd_code_example("", rerank_k=3, enable_rerank=False)

    # Empty query might still succeed but return no relevant results
    # The function should handle it gracefully without crashing
    assert isinstance(result, dict), "Result should be a dictionary even for empty query"
    print(f"[OK] Empty query handled gracefully, success={result['success']}")


# =============================================================================
# Test: Different rerank_k Values
# =============================================================================
@pytest.mark.asyncio
async def test_rerank_k_values():
    """Test different rerank_k parameter values."""
    print(f"\n{'='*60}")
    print("TEST: Different rerank_k Values")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    test_values = [1, 3, 5, 10]
    query = "How to create a mesh?"

    for rerank_k in test_values:
        result = await get_usd_code_example(query, rerank_k=rerank_k, enable_rerank=False)
        assert isinstance(result, dict), f"Result should be dict for rerank_k={rerank_k}"
        print(f"  [OK] rerank_k={rerank_k}: success={result['success']}")

    print("[OK] All rerank_k values handled correctly")


# =============================================================================
# Test: Rerank Toggle
# =============================================================================
@pytest.mark.asyncio
async def test_enable_rerank_toggle():
    """Test with reranking enabled vs disabled."""
    print(f"\n{'='*60}")
    print("TEST: Rerank Toggle")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    query = "How to set prim attributes?"

    # Test with reranking disabled
    result_no_rerank = await get_usd_code_example(query, rerank_k=3, enable_rerank=False)
    assert isinstance(result_no_rerank, dict), "Result should be dict with rerank disabled"
    print(f"  [OK] Rerank disabled: success={result_no_rerank['success']}")

    # Test with reranking enabled (may fail if API key not available, which is OK)
    try:
        result_with_rerank = await get_usd_code_example(query, rerank_k=3, enable_rerank=True)
        assert isinstance(result_with_rerank, dict), "Result should be dict with rerank enabled"
        print(f"  [OK] Rerank enabled: success={result_with_rerank['success']}")
    except Exception as e:
        print(f"  [SKIP] Rerank enabled test skipped (API may not be available): {e}")

    print("[OK] Rerank toggle test completed")


# =============================================================================
# Test: Code Examples Contain Expected Imports
# =============================================================================
@pytest.mark.asyncio
async def test_code_examples_contain_expected_imports():
    """Test that code examples for USD stage creation contain expected imports."""
    print(f"\n{'='*60}")
    print("TEST: Code Examples Contain Expected Imports")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    # Get API key from environment for embedding config
    api_key = os.getenv("NVIDIA_API_KEY", "")
    embedding_config = (
        {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }
        if api_key
        else None
    )

    result = await get_usd_code_example(
        "How to create a USD stage?",
        rerank_k=5,
        enable_rerank=False,
        embedding_config=embedding_config,
    )

    assert result["success"] is True, f"Query should succeed, got error: {result.get('error')}"

    result_text = result["result"]

    # Code examples for USD stage should contain pxr imports
    expected_patterns = ["from pxr", "import Usd", "Usd.Stage"]
    found_patterns = [p for p in expected_patterns if p in result_text]

    assert len(found_patterns) > 0, (
        f"Result should contain USD import patterns like {expected_patterns}, " f"got: {result_text[:300]}..."
    )

    print(f"[OK] Found expected patterns in code: {found_patterns}")


# =============================================================================
# Test: Code Examples Contain Stage Creation Methods
# =============================================================================
@pytest.mark.asyncio
async def test_code_examples_contain_stage_creation():
    """Test that code examples show how to create a USD stage."""
    print(f"\n{'='*60}")
    print("TEST: Code Examples Contain Stage Creation Methods")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    api_key = os.getenv("NVIDIA_API_KEY", "")
    embedding_config = (
        {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }
        if api_key
        else None
    )

    result = await get_usd_code_example(
        "How to create a USD stage?",
        rerank_k=5,
        enable_rerank=False,
        embedding_config=embedding_config,
    )

    assert result["success"] is True, f"Query should succeed"

    result_text = result["result"]

    # Should contain stage creation methods
    creation_methods = ["CreateInMemory", "CreateNew", "Stage.Open"]
    found_methods = [m for m in creation_methods if m in result_text]

    assert len(found_methods) > 0, f"Result should contain stage creation methods like {creation_methods}"

    print(f"[OK] Found stage creation methods: {found_methods}")


# =============================================================================
# Test: Code Examples for Mesh Contain Geometry
# =============================================================================
@pytest.mark.asyncio
async def test_code_examples_mesh_contains_geometry():
    """Test that mesh code examples contain geometry-related code."""
    print(f"\n{'='*60}")
    print("TEST: Code Examples for Mesh Contain Geometry")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    api_key = os.getenv("NVIDIA_API_KEY", "")
    embedding_config = (
        {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }
        if api_key
        else None
    )

    result = await get_usd_code_example(
        "How to create a mesh in USD?",
        rerank_k=5,
        enable_rerank=False,
        embedding_config=embedding_config,
    )

    assert result["success"] is True, f"Query should succeed"

    result_text = result["result"]

    # Should contain mesh-related terms
    mesh_terms = ["Mesh", "UsdGeom", "points", "vertices", "faces", "faceVertexCounts"]
    found_terms = [t for t in mesh_terms if t.lower() in result_text.lower()]

    assert len(found_terms) >= 2, (
        f"Result should contain mesh-related terms like {mesh_terms}, " f"found: {found_terms}"
    )

    print(f"[OK] Found mesh-related terms: {found_terms}")


# =============================================================================
# Test: Python Code Validation
# =============================================================================
@pytest.mark.asyncio
async def test_get_code_examples_returns_valid_python():
    """Test that get_usd_code_example returns valid Python code from the FAISS database."""
    print(f"\n{'='*60}")
    print("TEST: Python Code Validation")
    print(f"{'='*60}")

    assert FAISS_CODE_INDEX_PATH.exists(), f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}"
    # Test queries - use common USD operations that should be in the database
    test_queries = [
        "How to create a USD stage?",
        "How to create a mesh?",
        "How to set prim attributes?",
    ]

    all_passed = True
    total_code_blocks = 0

    # Get API key from environment for embedding config
    api_key = os.getenv("NVIDIA_API_KEY", "")
    embedding_config = (
        {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }
        if api_key
        else None
    )

    for query in test_queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing query: {query}")
        logger.info(f"{'='*80}")

        # Call the actual function to retrieve real code from FAISS
        result = await get_usd_code_example(
            query,
            rerank_k=3,
            enable_rerank=False,
            embedding_config=embedding_config,
        )

        # Check if the function succeeded
        if not result["success"]:
            logger.warning(f"Query '{query}' failed: {result.get('error', 'Unknown error')}")
            continue

        if not result["result"] or "No relevant code examples found" in result["result"]:
            logger.warning(f"No results found for query: {query}")
            continue

        # Extract code blocks from the result
        code_blocks = extract_code_from_result(result["result"])

        if not code_blocks:
            logger.warning(f"No code blocks extracted for query: {query}")
            continue

        logger.info(f"Found {len(code_blocks)} code block(s) for query: {query}")

        # Validate each code block is valid Python
        for idx, code in enumerate(code_blocks):
            total_code_blocks += 1
            code_preview = code[:100].replace("\n", " ") + ("..." if len(code) > 100 else "")

            try:
                # Try to parse the code as Python AST
                ast.parse(code)
                logger.info(f"✓ Code block {idx + 1} is valid Python: {code_preview}")
            except SyntaxError as e:
                all_passed = False
                logger.error(f"✗ Code block {idx + 1} contains invalid Python syntax!")
                logger.error(f"  Error: {e}")
                logger.error(f"  Code preview: {code_preview}")
                pytest.fail(
                    f"Query '{query}' returned invalid Python in code block {idx + 1}:\n"
                    f"Error: {e}\n"
                    f"Code:\n{code}"
                )

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("Validation Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total queries tested: {len(test_queries)}")
    logger.info(f"Total code blocks validated: {total_code_blocks}")
    logger.info(f"All code blocks valid: {all_passed}")
    logger.info(f"{'='*80}\n")

    # Ensure we tested at least some code
    assert total_code_blocks > 0, (
        f"No code blocks were tested. This indicates:\n"
        f"1. FAISS database is not properly configured\n"
        f"2. No matching examples found for test queries\n"
        f"3. Code extraction pattern needs adjustment"
    )

    print(f"[OK] Python validation test passed! Validated {total_code_blocks} code blocks.")


# =============================================================================
# Main entry point for running tests directly
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests manually."""
        print("\n" + "=" * 80)
        print("search_usd_code_examples Tests")
        print("=" * 80)

        # Run FAISS check first
        test_faiss_index_exists()

        # Run async tests
        await test_response_structure()
        await test_successful_query()
        await test_empty_query()
        await test_rerank_k_values()
        await test_enable_rerank_toggle()
        await test_code_examples_contain_expected_imports()
        await test_code_examples_contain_stage_creation()
        await test_code_examples_mesh_contains_geometry()
        await test_get_code_examples_returns_valid_python()

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    asyncio.run(run_all_tests())
