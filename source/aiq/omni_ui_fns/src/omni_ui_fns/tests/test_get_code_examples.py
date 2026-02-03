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

"""Test the get_code_examples function and retrieval formatting."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from omni_ui_fns.config import FAISS_CODE_INDEX_PATH
from omni_ui_fns.functions.get_code_examples import get_code_examples
from omni_ui_fns.services.retrieval import Retriever, get_rag_context_omni_ui_code


def test_faiss_index_exists():
    """Test that the FAISS index exists."""
    print(f"\n{'='*60}")
    print("TEST: FAISS Index Existence")
    print(f"{'='*60}")
    print(f"FAISS_CODE_INDEX_PATH: {FAISS_CODE_INDEX_PATH}")
    print(f"Path exists: {FAISS_CODE_INDEX_PATH.exists()}")

    if FAISS_CODE_INDEX_PATH.exists():
        print(f"[OK] FAISS index found at: {FAISS_CODE_INDEX_PATH}")
        # List files in the directory
        if FAISS_CODE_INDEX_PATH.is_dir():
            files = list(FAISS_CODE_INDEX_PATH.glob("*"))
            print(f"Files in index directory:")
            for f in files:
                print(f"  - {f.name}")
    else:
        print(f"[FAIL] FAISS index NOT found at: {FAISS_CODE_INDEX_PATH}")
        print(f"[WARN]  Tests will not be able to retrieve actual examples")


def test_retriever_initialization():
    """Test that the retriever can be initialized."""
    print(f"\n{'='*60}")
    print("TEST: Retriever Initialization")
    print(f"{'='*60}")

    if not FAISS_CODE_INDEX_PATH.exists():
        pytest.skip("FAISS index not found, skipping retriever test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        # Create embedding config
        embedding_config = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }

        retriever = Retriever(
            embedding_config=embedding_config,
            load_path=str(FAISS_CODE_INDEX_PATH),
            top_k=5,
        )

        print(f"[OK] Retriever initialized successfully")
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
    print("TEST: Retriever Search and Metadata")
    print(f"{'='*60}")

    if not FAISS_CODE_INDEX_PATH.exists():
        pytest.skip("FAISS index not found, skipping search test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        # Create embedding config
        embedding_config = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }

        retriever = Retriever(
            embedding_config=embedding_config,
            load_path=str(FAISS_CODE_INDEX_PATH),
            top_k=3,
        )

        # Perform a search
        query = "create a search field"
        print(f"Search query: '{query}'")
        docs = retriever.search(query, top_k=3)

        print(f"[OK] Retrieved {len(docs)} documents")

        assert len(docs) > 0, "Should retrieve at least one document"

        # Check metadata of first document
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"Page content (first 100 chars): {doc.page_content[:100]}...")
            print(f"Metadata keys: {list(doc.metadata.keys())}")

            # Check which schema this document uses
            # Schema 1: file_name, method_name, source_code
            # Schema 2: description, code, file_path, function_name, class_name

            is_schema1 = "method_name" in doc.metadata and "source_code" in doc.metadata
            is_schema2 = "function_name" in doc.metadata or "code" in doc.metadata

            print(
                f"  Schema detected: {'Schema 1 (original)' if is_schema1 else 'Schema 2 (window examples)' if is_schema2 else 'Unknown'}"
            )

            if is_schema1:
                expected_keys = ["file_name", "file_path", "source_code"]
            elif is_schema2:
                expected_keys = ["description", "code", "file_path", "function_name", "class_name"]
            else:
                expected_keys = list(doc.metadata.keys())

            for key in expected_keys:
                has_key = key in doc.metadata
                value = doc.metadata.get(key, "MISSING")
                status = "[OK]" if has_key else "[FAIL]"
                print(f"  {status} {key}: {str(value)[:50]}...")

            # Verify the values based on schema
            if i == 1:  # Check first document in detail
                if is_schema1:
                    assert "source_code" in doc.metadata, "Schema 1 document should have 'source_code' in metadata"
                    source_code = doc.metadata.get("source_code", "")
                    print(f"\n  Source code length: {len(source_code)}")
                    assert len(source_code) > 0, "Source code should not be empty"
                elif is_schema2:
                    assert "code" in doc.metadata, "Schema 2 document should have 'code' in metadata"
                    code = doc.metadata.get("code", "")
                    description = doc.page_content
                    print(f"\n  Code length: {len(code)}")
                    print(f"  Description length: {len(description)}")
                    assert len(code) > 0, "Code should not be empty"

    except Exception as e:
        print(f"[FAIL] Search test failed: {e}")
        raise


def test_rag_context_formatting():
    """Test that get_rag_context_omni_ui_code formats output correctly."""
    print(f"\n{'='*60}")
    print("TEST: RAG Context Formatting")
    print(f"{'='*60}")

    if not FAISS_CODE_INDEX_PATH.exists():
        pytest.skip("FAISS index not found, skipping formatting test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        # Create embedding config
        embedding_config = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }

        retriever = Retriever(
            embedding_config=embedding_config,
            load_path=str(FAISS_CODE_INDEX_PATH),
            top_k=3,
        )

        # Get RAG context
        query = "create a button with styling"
        print(f"Query: '{query}'")

        rag_context = get_rag_context_omni_ui_code(
            user_query=query,
            retriever=retriever,
            rag_top_k=3,
            rerank_k=3,
            reranker=None,  # No reranking for this test
        )

        print(f"[OK] Retrieved RAG context ({len(rag_context)} chars)")
        print(f"\n{'='*60}")
        print("RAG CONTEXT OUTPUT:")
        print(f"{'='*60}")
        print(rag_context)
        print(f"{'='*60}\n")

        # Validate the format
        assert len(rag_context) > 0, "RAG context should not be empty"

        # Check for expected format elements
        assert "### Example" in rag_context, "Should contain example headers"
        assert "File:" in rag_context, "Should contain file information"
        assert "Path:" in rag_context, "Should contain path information"
        assert "Method:" in rag_context, "Should contain method information"
        assert "```python" in rag_context, "Should contain code blocks"

        # Check that we don't have "unknown" appearing everywhere
        unknown_count = rag_context.count("unknown")
        print(f"'unknown' appears {unknown_count} times in output")

        # Should have minimal unknowns (some might be legitimate)
        if unknown_count > 3:
            print(f"[WARN]  WARNING: Many 'unknown' values detected ({unknown_count})")

        # Check that Method: line doesn't just say "unknown"
        lines = rag_context.split("\n")
        method_lines = [line for line in lines if line.startswith("Method:")]
        print(f"\nMethod lines found:")
        for line in method_lines:
            print(f"  {line}")
            # Method should not be just "Method: unknown"
            assert line.strip() != "Method: unknown", "Method should have a proper name"

    except Exception as e:
        print(f"[FAIL] Formatting test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_get_code_examples_async():
    """Test the full get_code_examples function."""
    print(f"\n{'='*60}")
    print("TEST: Full get_code_examples Function")
    print(f"{'='*60}")

    if not FAISS_CODE_INDEX_PATH.exists():
        pytest.skip("FAISS index not found, skipping function test")

    try:
        # Get API key from environment
        api_key = os.getenv("NVIDIA_API_KEY", "")

        # Create configs
        embedding_config = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "endpoint": None,
            "api_key": api_key,
        }

        # Test the function
        query = "create a search field widget"
        print(f"Query: '{query}'")

        result = await get_code_examples(
            request=query,
            rerank_k=3,
            enable_rerank=False,  # Disable reranking for faster test
            embedding_config=embedding_config,
            reranking_config=None,
        )

        print(f"\nResult keys: {result.keys()}")
        print(f"Success: {result.get('success')}")
        print(f"Error: {result.get('error')}")
        print(f"Result length: {len(result.get('result', ''))}")

        assert result["success"] == True, f"Function should succeed. Error: {result.get('error')}"
        assert len(result["result"]) > 0, "Should return non-empty result"

        # Print the formatted output
        print(f"\n{'='*60}")
        print("FUNCTION OUTPUT:")
        print(f"{'='*60}")
        print(result["result"])
        print(f"{'='*60}\n")

        # Validate output format
        output = result["result"]
        assert "### Example" in output, "Should contain example headers"
        assert "Method:" in output, "Should contain method information"
        assert "```python" in output, "Should contain code blocks"

        # Check method lines
        lines = output.split("\n")
        method_lines = [line for line in lines if line.startswith("Method:")]
        print(f"Method lines in output:")
        for line in method_lines:
            print(f"  {line}")
            assert line.strip() != "Method: unknown", "Method should not be 'unknown'"

        print(f"[OK] Full function test passed!")

    except Exception as e:
        print(f"[FAIL] Full function test failed: {e}")
        raise


def run_all_tests():
    """Run all tests manually without pytest."""
    print("\n" + "=" * 80)
    print("RUNNING OMNI_UI_FNS CODE EXAMPLES TESTS")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        print("\n[WARNING] NVIDIA_API_KEY not set in environment")
        print("   Some tests may fail without proper authentication\n")
    else:
        print(f"\n[OK] NVIDIA_API_KEY is set (length: {len(api_key)})\n")

    try:
        # Test 1: FAISS index
        test_faiss_index_exists()

        # Test 2: Retriever init
        test_retriever_initialization()

        # Test 3: Retriever search
        test_retriever_search()

        # Test 4: RAG context formatting
        test_rag_context_formatting()

        # Test 5: Full async function
        print(f"\n{'='*60}")
        print("Running async function test...")
        print(f"{'='*60}")
        asyncio.run(test_get_code_examples_async())

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
