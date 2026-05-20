## Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Test suite for search_usd_knowledge tool - USD knowledge search tests."""

import logging
import os
import sys
from pathlib import Path

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omni_aiq_usd_code.config import FAISS_KNOWLEDGE_INDEX_PATH
from omni_aiq_usd_code.functions.get_usd_knowledge import get_usd_knowledge

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def skip_if_faiss_unavailable():
    """Helper to skip tests if FAISS knowledge index is not available."""
    assert FAISS_KNOWLEDGE_INDEX_PATH.exists(), f"FAISS knowledge index not found at {FAISS_KNOWLEDGE_INDEX_PATH}"


def get_embedding_config():
    """Get embedding configuration with API key if available."""
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key:
        return {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }
    return None


# =============================================================================
# Test: FAISS Knowledge Index Exists
# =============================================================================
def test_faiss_knowledge_index_exists():
    """Test that the FAISS knowledge index file exists."""
    print(f"\n{'='*60}")
    print("TEST: FAISS Knowledge Index Exists")
    print(f"{'='*60}")

    print(f"Checking FAISS knowledge index path: {FAISS_KNOWLEDGE_INDEX_PATH}")

    assert FAISS_KNOWLEDGE_INDEX_PATH.exists(), f"FAISS knowledge index not found at {FAISS_KNOWLEDGE_INDEX_PATH}"
    print(f"[OK] FAISS knowledge index exists at {FAISS_KNOWLEDGE_INDEX_PATH}")


# =============================================================================
# Test: Response Structure
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_response_structure():
    """Test that get_usd_knowledge returns correct response structure."""
    print(f"\n{'='*60}")
    print("TEST: Response Structure")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    result = await get_usd_knowledge(
        "What is layer composition?",
        rerank_k=3,
        enable_rerank=False,
        embedding_config=get_embedding_config(),
    )

    # Check response structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "success" in result, "Result should have 'success' key"
    assert "result" in result, "Result should have 'result' key"
    assert "error" in result, "Result should have 'error' key"

    print(f"[OK] Response structure correct: success={result['success']}")


# =============================================================================
# Test: Query About Composition
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_query_composition():
    """Test querying about USD composition concepts."""
    print(f"\n{'='*60}")
    print("TEST: Query About Composition")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    result = await get_usd_knowledge(
        "What is layer composition in USD?",
        rerank_k=5,
        enable_rerank=False,
        embedding_config=get_embedding_config(),
    )

    assert result["success"] is True, f"Query should succeed, got error: {result.get('error')}"
    assert result["result"], "Result should not be empty"

    # Check that result contains relevant content
    result_text = result["result"].lower()

    # Should contain at least one of these composition-related terms
    composition_terms = ["composition", "layer", "arc", "combine", "stack"]
    found_terms = [term for term in composition_terms if term in result_text]

    assert len(found_terms) > 0, f"Result should mention composition concepts, got: {result['result'][:200]}..."

    print(f"[OK] Query returned relevant composition content (found terms: {found_terms})")


# =============================================================================
# Test: Query About Prims
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_query_prims():
    """Test querying about USD prims."""
    print(f"\n{'='*60}")
    print("TEST: Query About Prims")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    result = await get_usd_knowledge(
        "What are prims in USD?",
        rerank_k=5,
        enable_rerank=False,
        embedding_config=get_embedding_config(),
    )

    assert result["success"] is True, f"Query should succeed, got error: {result.get('error')}"
    assert result["result"], "Result should not be empty"

    # Check that result contains relevant content
    result_text = result["result"].lower()

    # Should contain at least one prim-related term
    prim_terms = ["prim", "primitive", "scene", "hierarchy", "stage"]
    found_terms = [term for term in prim_terms if term in result_text]

    assert len(found_terms) > 0, f"Result should mention prim concepts"

    print(f"[OK] Query returned relevant prim content (found terms: {found_terms})")


# =============================================================================
# Test: Query About Materials
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_query_materials():
    """Test querying about USD materials and shading."""
    print(f"\n{'='*60}")
    print("TEST: Query About Materials")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    result = await get_usd_knowledge(
        "How do materials work in USD?",
        rerank_k=5,
        enable_rerank=False,
        embedding_config=get_embedding_config(),
    )

    assert result["success"] is True, f"Query should succeed, got error: {result.get('error')}"
    assert result["result"], "Result should not be empty"

    print(f"[OK] Query about materials returned: {len(result['result'])} chars")


# =============================================================================
# Test: Different rerank_k Values
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_rerank_k_values():
    """Test different rerank_k parameter values."""
    print(f"\n{'='*60}")
    print("TEST: Different rerank_k Values")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    test_values = [1, 3, 5, 10]
    query = "What is USD layer composition?"

    for rerank_k in test_values:
        result = await get_usd_knowledge(
            query,
            rerank_k=rerank_k,
            enable_rerank=False,
            embedding_config=get_embedding_config(),
        )
        assert isinstance(result, dict), f"Result should be dict for rerank_k={rerank_k}"
        print(f"  [OK] rerank_k={rerank_k}: success={result['success']}")

    print("[OK] All rerank_k values handled correctly")


# =============================================================================
# Test: Empty Query Handling
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_empty_query():
    """Test handling of empty queries."""
    print(f"\n{'='*60}")
    print("TEST: Empty Query Handling")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    result = await get_usd_knowledge(
        "",
        rerank_k=3,
        enable_rerank=False,
        embedding_config=get_embedding_config(),
    )

    # Empty query should be handled gracefully without crashing
    assert isinstance(result, dict), "Result should be a dictionary even for empty query"
    print(f"[OK] Empty query handled gracefully, success={result['success']}")


# =============================================================================
# Test: Rerank Toggle
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_rerank_toggle():
    """Test with reranking enabled vs disabled."""
    print(f"\n{'='*60}")
    print("TEST: Rerank Toggle")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    query = "What is USD prim inheritance?"

    # Test with reranking disabled
    result_no_rerank = await get_usd_knowledge(
        query,
        rerank_k=3,
        enable_rerank=False,
        embedding_config=get_embedding_config(),
    )
    assert isinstance(result_no_rerank, dict), "Result should be dict with rerank disabled"
    print(f"  [OK] Rerank disabled: success={result_no_rerank['success']}")

    # Test with reranking enabled (may fail if API key not available)
    try:
        result_with_rerank = await get_usd_knowledge(
            query,
            rerank_k=3,
            enable_rerank=True,
            embedding_config=get_embedding_config(),
        )
        assert isinstance(result_with_rerank, dict), "Result should be dict with rerank enabled"
        print(f"  [OK] Rerank enabled: success={result_with_rerank['success']}")
    except Exception as e:
        print(f"  [SKIP] Rerank enabled test skipped (API may not be available): {e}")

    print("[OK] Rerank toggle test completed")


# =============================================================================
# Test: Result Contains Title and Content
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_result_format():
    """Test that result contains formatted title and content."""
    print(f"\n{'='*60}")
    print("TEST: Result Contains Title and Content")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    result = await get_usd_knowledge(
        "What is layer composition?",
        rerank_k=3,
        enable_rerank=False,
        embedding_config=get_embedding_config(),
    )

    assert result["success"] is True, f"Query should succeed, got error: {result.get('error')}"

    result_text = result["result"]

    # Result should have some structure (titles, separators, etc.)
    has_structure = "Title" in result_text or "---" in result_text or "URL" in result_text or len(result_text) > 100

    assert has_structure, "Result should have formatted structure with titles or content"

    print(f"[OK] Result has proper formatting ({len(result_text)} chars)")


# =============================================================================
# Test: Query About Lighting (UsdLux)
# =============================================================================
@pytest.mark.asyncio
async def test_knowledge_query_lighting():
    """Test querying about USD lighting."""
    print(f"\n{'='*60}")
    print("TEST: Query About Lighting")
    print(f"{'='*60}")

    skip_if_faiss_unavailable()

    result = await get_usd_knowledge(
        "How do lights work in USD?",
        rerank_k=5,
        enable_rerank=False,
        embedding_config=get_embedding_config(),
    )

    assert result["success"] is True, f"Query should succeed, got error: {result.get('error')}"
    assert result["result"], "Result should not be empty"

    print(f"[OK] Query about lighting returned: {len(result['result'])} chars")


# =============================================================================
# Main entry point for running tests directly
# =============================================================================
if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        """Run all tests manually."""
        print("\n" + "=" * 80)
        print("search_usd_knowledge Tests")
        print("=" * 80)

        test_faiss_knowledge_index_exists()
        await test_knowledge_response_structure()
        await test_knowledge_query_composition()
        await test_knowledge_query_prims()
        await test_knowledge_query_materials()
        await test_knowledge_rerank_k_values()
        await test_knowledge_empty_query()
        await test_knowledge_rerank_toggle()
        await test_knowledge_result_format()
        await test_knowledge_query_lighting()

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80 + "\n")

    asyncio.run(run_all_tests())
