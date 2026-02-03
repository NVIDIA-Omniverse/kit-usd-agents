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

"""Test the get_window_examples function and UI window examples retrieval."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from omni_ui_fns.functions.get_window_examples import get_window_examples
from omni_ui_fns.services.ui_window_examples_retrieval import (
    UIWindowExamplesRetriever,
    create_ui_window_examples_retriever,
    get_ui_window_examples,
)


def get_faiss_index_path():
    """Get the default FAISS index path for UI window examples."""
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / "data"
    faiss_index_path = data_dir / "ui_window_examples_faiss"
    return faiss_index_path


def test_faiss_index_exists():
    """Test that the FAISS index exists for UI window examples."""
    print(f"\n{'='*60}")
    print("TEST: UI Window Examples FAISS Index Existence")
    print(f"{'='*60}")

    faiss_index_path = get_faiss_index_path()
    print(f"FAISS Index Path: {faiss_index_path}")
    print(f"Path exists: {faiss_index_path.exists()}")

    if faiss_index_path.exists():
        print(f"[OK] FAISS index found at: {faiss_index_path}")
        # List files in the directory
        if faiss_index_path.is_dir():
            files = list(faiss_index_path.glob("*"))
            print(f"Files in index directory:")
            for f in files:
                print(f"  - {f.name}")

            # Check for required files
            index_file = faiss_index_path / "index.faiss"
            pkl_file = faiss_index_path / "index.pkl"

            if index_file.exists():
                print(f"[OK] index.faiss found")
            else:
                print(f"[FAIL] index.faiss NOT found")

            if pkl_file.exists():
                print(f"[OK] index.pkl found")
            else:
                print(f"[FAIL] index.pkl NOT found")
    else:
        print(f"[FAIL] FAISS index NOT found at: {faiss_index_path}")
        print(f"[WARN] Tests will not be able to retrieve actual examples")


def test_retriever_initialization():
    """Test that the UI window examples retriever can be initialized."""
    print(f"\n{'='*60}")
    print("TEST: UI Window Examples Retriever Initialization")
    print(f"{'='*60}")

    faiss_index_path = get_faiss_index_path()
    if not faiss_index_path.exists():
        pytest.skip("FAISS index not found, skipping retriever test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        retriever = create_ui_window_examples_retriever(
            faiss_index_path=str(faiss_index_path),
            api_key=api_key,
            top_k=5,
        )

        print(f"[OK] UI Window Examples Retriever initialized successfully")
        print(f"   - Vector DB: {retriever.vectordb is not None}")
        print(f"   - Retriever: {retriever.retriever is not None}")
        print(f"   - Top K: {retriever.top_k}")

        assert retriever.vectordb is not None, "Vector database should be loaded"
        assert retriever.retriever is not None, "Retriever should be initialized"

    except Exception as e:
        print(f"[FAIL] Failed to initialize retriever: {e}")
        raise


def test_retriever_search():
    """Test that retriever can search and return documents with correct metadata."""
    print(f"\n{'='*60}")
    print("TEST: UI Window Examples Retriever Search")
    print(f"{'='*60}")

    faiss_index_path = get_faiss_index_path()
    if not faiss_index_path.exists():
        pytest.skip("FAISS index not found, skipping search test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        retriever = create_ui_window_examples_retriever(
            faiss_index_path=str(faiss_index_path),
            api_key=api_key,
            top_k=3,
        )

        # Perform a search
        query = "create a collapsible frame"
        print(f"Search query: '{query}'")
        docs = retriever.search(query, top_k=3)

        print(f"[OK] Retrieved {len(docs)} documents")

        assert len(docs) > 0, "Should retrieve at least one document"

        # Check metadata of first document
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"Page content (first 100 chars): {doc.page_content[:100]}...")
            print(f"Metadata keys: {list(doc.metadata.keys())}")

            # Expected metadata for window examples
            expected_keys = ["description", "code", "file_path", "function_name", "class_name", "line_number"]

            for key in expected_keys:
                has_key = key in doc.metadata
                value = doc.metadata.get(key, "MISSING")
                status = "[OK]" if has_key else "[FAIL]"
                if isinstance(value, str):
                    print(f"  {status} {key}: {str(value)[:80]}...")
                else:
                    print(f"  {status} {key}: {value}")

            # Verify the first document in detail
            if i == 1:
                assert "code" in doc.metadata, "Document should have 'code' in metadata"
                assert "description" in doc.metadata, "Document should have 'description' in metadata"
                assert "file_path" in doc.metadata, "Document should have 'file_path' in metadata"
                assert "function_name" in doc.metadata, "Document should have 'function_name' in metadata"
                assert "class_name" in doc.metadata, "Document should have 'class_name' in metadata"

                code = doc.metadata.get("code", "")
                description = doc.metadata.get("description", "")
                print(f"\n  Code length: {len(code)}")
                print(f"  Description length: {len(description)}")
                assert len(code) > 0, "Code should not be empty"
                assert len(description) > 0, "Description should not be empty"

    except Exception as e:
        print(f"[FAIL] Search test failed: {e}")
        raise


def test_structured_results():
    """Test that get_ui_window_examples returns properly structured results."""
    print(f"\n{'='*60}")
    print("TEST: Structured Results Format")
    print(f"{'='*60}")

    faiss_index_path = get_faiss_index_path()
    if not faiss_index_path.exists():
        pytest.skip("FAISS index not found, skipping structured results test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        retriever = create_ui_window_examples_retriever(
            faiss_index_path=str(faiss_index_path),
            api_key=api_key,
            top_k=3,
        )

        # Get structured results
        query = "create a window with a tree view"
        print(f"Query: '{query}'")

        results = get_ui_window_examples(
            user_query=query,
            retriever=retriever,
            top_k=3,
            format_type="structured",
        )

        print(f"[OK] Retrieved {len(results)} structured results")

        assert len(results) > 0, "Should return at least one result"
        assert isinstance(results, list), "Structured results should be a list"

        # Check first result structure
        first_result = results[0]
        print(f"\nFirst result structure:")
        for key, value in first_result.items():
            if isinstance(value, str) and len(value) > 80:
                print(f"  {key}: {value[:80]}...")
            else:
                print(f"  {key}: {value}")

        # Validate required fields
        required_fields = ["rank", "description", "code", "file_path", "function_name", "class_name", "line_number"]
        for field in required_fields:
            assert field in first_result, f"Result should have '{field}' field"
            print(f"[OK] Field '{field}' present")

        # Validate data types
        assert isinstance(first_result["rank"], int), "rank should be an integer"
        assert isinstance(first_result["description"], str), "description should be a string"
        assert isinstance(first_result["code"], str), "code should be a string"
        assert isinstance(first_result["file_path"], str), "file_path should be a string"
        assert len(first_result["code"]) > 0, "code should not be empty"

    except Exception as e:
        print(f"[FAIL] Structured results test failed: {e}")
        raise


def test_formatted_results():
    """Test that get_ui_window_examples returns properly formatted string results."""
    print(f"\n{'='*60}")
    print("TEST: Formatted String Results")
    print(f"{'='*60}")

    faiss_index_path = get_faiss_index_path()
    if not faiss_index_path.exists():
        pytest.skip("FAISS index not found, skipping formatted results test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        retriever = create_ui_window_examples_retriever(
            faiss_index_path=str(faiss_index_path),
            api_key=api_key,
            top_k=3,
        )

        # Get formatted results
        query = "create a settings panel"
        print(f"Query: '{query}'")

        results = get_ui_window_examples(
            user_query=query,
            retriever=retriever,
            top_k=3,
            format_type="formatted",
        )

        print(f"[OK] Retrieved formatted results ({len(results)} chars)")

        assert isinstance(results, str), "Formatted results should be a string"
        assert len(results) > 0, "Formatted results should not be empty"

        # Check for expected format elements
        assert "UI Window Examples Search Results" in results, "Should contain search results header"
        assert "Match" in results, "Should contain match headers"
        assert "**File:**" in results, "Should contain file information"
        assert "**Function:**" in results, "Should contain function information"
        assert "**Description:**" in results, "Should contain description"
        assert "**Code:**" in results, "Should contain code section"
        assert "```python" in results, "Should contain Python code blocks"

        print(f"\n{'='*60}")
        print("FORMATTED OUTPUT PREVIEW (first 500 chars):")
        print(f"{'='*60}")
        print(results[:500])
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"[FAIL] Formatted results test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_get_window_examples_async_structured():
    """Test the full get_window_examples function with structured format."""
    print(f"\n{'='*60}")
    print("TEST: Full get_window_examples Function (Structured)")
    print(f"{'='*60}")

    faiss_index_path = get_faiss_index_path()
    if not faiss_index_path.exists():
        pytest.skip("FAISS index not found, skipping function test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        # Create embedding config
        embedding_config = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }

        # Test with structured format
        query = "create a dockable window"
        print(f"Query: '{query}'")
        print(f"Format: structured")

        result = await get_window_examples(
            request=query,
            top_k=3,
            format_type="structured",
            embedding_config=embedding_config,
            faiss_index_path=str(faiss_index_path),
        )

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")
        print(f"Count: {result.get('count')}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert "result" in result, "Should have 'result' key"
        assert len(result["result"]) > 0, "Should return non-empty result"

        # Print sample output
        output = result["result"]
        print(f"\n{'='*60}")
        print("FUNCTION OUTPUT PREVIEW (first 500 chars):")
        print(f"{'='*60}")
        print(str(output)[:500])
        print(f"{'='*60}\n")

        # Validate structured output format
        assert "Found" in output, "Should mention number of examples found"
        assert "### Example" in output, "Should contain example headers"
        assert "**File:**" in output, "Should contain file information"
        assert "```python" in output, "Should contain code blocks"

        print(f"[OK] Structured format test passed!")

    except Exception as e:
        print(f"[FAIL] Structured format test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_get_window_examples_async_formatted():
    """Test the full get_window_examples function with formatted format."""
    print(f"\n{'='*60}")
    print("TEST: Full get_window_examples Function (Formatted)")
    print(f"{'='*60}")

    faiss_index_path = get_faiss_index_path()
    if not faiss_index_path.exists():
        pytest.skip("FAISS index not found, skipping function test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        # Create embedding config
        embedding_config = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }

        # Test with formatted format (default)
        query = "create a property panel with sliders"
        print(f"Query: '{query}'")
        print(f"Format: formatted")

        result = await get_window_examples(
            request=query,
            top_k=5,
            format_type="formatted",
            embedding_config=embedding_config,
            faiss_index_path=str(faiss_index_path),
        )

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert "result" in result, "Should have 'result' key"
        assert len(result["result"]) > 0, "Should return non-empty result"

        # Print the formatted output
        output = result["result"]
        print(f"\n{'='*60}")
        print("FUNCTION OUTPUT:")
        print(f"{'='*60}")
        print(output)
        print(f"{'='*60}\n")

        # Validate formatted output
        assert "UI Window Examples Search Results" in output, "Should contain search results header"
        assert "Match" in output, "Should contain match headers"
        assert "```python" in output, "Should contain code blocks"

        print(f"[OK] Formatted format test passed!")

    except Exception as e:
        print(f"[FAIL] Formatted format test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_get_window_examples_different_queries():
    """Test get_window_examples with different query types."""
    print(f"\n{'='*60}")
    print("TEST: Different Query Types")
    print(f"{'='*60}")

    faiss_index_path = get_faiss_index_path()
    if not faiss_index_path.exists():
        pytest.skip("FAISS index not found, skipping query test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        # Create embedding config
        embedding_config = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }

        # Test different query types
        queries = [
            "create a collapsible group",
            "add a checkbox widget",
            "make a scrollable container",
            "create a menu bar",
        ]

        for query in queries:
            print(f"\nTesting query: '{query}'")

            result = await get_window_examples(
                request=query,
                top_k=2,
                format_type="formatted",
                embedding_config=embedding_config,
                faiss_index_path=str(faiss_index_path),
            )

            if result["success"]:
                print(f"[OK] Query succeeded - returned {len(result['result'])} chars")
            else:
                print(f"[FAIL] Query failed: {result.get('error')}")
                assert False, f"Query should succeed for '{query}'"

        print(f"\n[OK] All query types test passed!")

    except Exception as e:
        print(f"[FAIL] Different queries test failed: {e}")
        raise


def run_all_tests():
    """Run all tests manually without pytest."""
    print("\n" + "=" * 80)
    print("RUNNING OMNI_UI_FNS WINDOW EXAMPLES TESTS")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        print("\n[WARNING] NVIDIA_API_KEY not set in environment")
        print("   Some tests may fail without proper authentication\n")
    else:
        print(f"\n[OK] NVIDIA_API_KEY is set (length: {len(api_key)})\n")

    try:
        # Test 1: FAISS index existence
        test_faiss_index_exists()

        # Test 2: Retriever initialization
        test_retriever_initialization()

        # Test 3: Retriever search
        test_retriever_search()

        # Test 4: Structured results
        test_structured_results()

        # Test 5: Formatted results
        test_formatted_results()

        # Test 6: Full async function (structured)
        print(f"\n{'='*60}")
        print("Running async function test (structured)...")
        print(f"{'='*60}")
        asyncio.run(test_get_window_examples_async_structured())

        # Test 7: Full async function (formatted)
        print(f"\n{'='*60}")
        print("Running async function test (formatted)...")
        print(f"{'='*60}")
        asyncio.run(test_get_window_examples_async_formatted())

        # Test 8: Different queries
        print(f"\n{'='*60}")
        print("Running different queries test...")
        print(f"{'='*60}")
        asyncio.run(test_get_window_examples_different_queries())

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
