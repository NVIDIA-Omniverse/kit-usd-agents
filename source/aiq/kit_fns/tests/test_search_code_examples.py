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

"""Test suite for search_code_examples function."""

import ast
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from kit_fns.functions.search_code_examples import get_code_search_service, search_code_examples
from kit_fns.services.code_search_service import CodeSearchService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_valid_query():
    """Test with a valid query."""
    # Mock the code search service
    with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True
        mock_instance.search_code_examples.return_value = [
            {
                "id": "example1",
                "title": "Sample Method",
                "description": "A sample method for testing",
                "file_path": "/path/to/file.py",
                "extension_id": "test.extension",
                "line_start": 10,
                "line_end": 20,
                "code": "def sample_method():\n    pass",
                "tags": ["test", "sample"],
                "relevance_score": 0.95,
            }
        ]
        mock_service.return_value = mock_instance

        # Mock telemetry
        with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
            with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                result = await search_code_examples("test query", rerank_k=5)

                assert result["success"]
                assert "Sample Method" in result["result"]


@pytest.mark.asyncio
async def test_empty_query():
    """Test with an empty query."""
    # Mock telemetry
    with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
        with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
            mock_telemetry.capture_call = AsyncMock()

            result = await search_code_examples("", rerank_k=5)

            assert not result["success"]
            assert "cannot be empty" in result["error"]


@pytest.mark.asyncio
async def test_whitespace_query():
    """Test with a whitespace-only query."""
    # Mock telemetry
    with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
        with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
            mock_telemetry.capture_call = AsyncMock()

            result = await search_code_examples("   ", rerank_k=5)

            assert not result["success"]
            assert "cannot be empty" in result["error"]


@pytest.mark.asyncio
async def test_invalid_rerank_k():
    """Test with invalid rerank_k parameter."""
    # Mock telemetry
    with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
        with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
            mock_telemetry.capture_call = AsyncMock()

            result = await search_code_examples("test query", rerank_k=0)

            assert not result["success"]
            assert "must be positive" in result["error"]


@pytest.mark.asyncio
async def test_negative_rerank_k():
    """Test with negative rerank_k parameter."""
    # Mock telemetry
    with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
        with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
            mock_telemetry.capture_call = AsyncMock()

            result = await search_code_examples("test query", rerank_k=-5)

            assert not result["success"]
            assert "must be positive" in result["error"]


@pytest.mark.asyncio
async def test_service_unavailable():
    """Test when code search service is unavailable."""
    # Mock the code search service as unavailable
    with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
        mock_instance = MagicMock()
        mock_instance.is_available.return_value = False
        mock_service.return_value = mock_instance

        # Mock telemetry
        with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
            with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                result = await search_code_examples("test query", rerank_k=5)

                assert not result["success"]
                assert "not available" in result["error"]


@pytest.mark.asyncio
async def test_no_results_found():
    """Test when no code examples are found."""
    # Mock the code search service with empty results
    with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True
        mock_instance.search_code_examples.return_value = []
        mock_service.return_value = mock_instance

        # Mock telemetry
        with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
            with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                result = await search_code_examples("nonexistent query", rerank_k=5)

                assert result["success"]
                assert "No code examples found" in result["result"]


@pytest.mark.asyncio
async def test_multiple_results():
    """Test with multiple search results."""
    # Mock the code search service with multiple results
    with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True
        mock_instance.search_code_examples.return_value = [
            {
                "id": f"example{i}",
                "title": f"Method {i}",
                "description": f"Description for method {i}",
                "file_path": f"/path/to/file{i}.py",
                "extension_id": "test.extension",
                "line_start": i * 10,
                "line_end": i * 10 + 10,
                "code": f"def method_{i}():\n    pass",
                "tags": ["test"],
                "relevance_score": 0.9 - (i * 0.1),
            }
            for i in range(1, 4)
        ]
        mock_service.return_value = mock_instance

        # Mock telemetry
        with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
            with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                result = await search_code_examples("test query", rerank_k=10)

                assert result["success"]
                assert "Found 3 relevant examples" in result["result"]
                assert "Method 1" in result["result"]
                assert "Method 2" in result["result"]
                assert "Method 3" in result["result"]


@pytest.mark.asyncio
async def test_result_formatting():
    """Test that results are properly formatted."""
    # Mock the code search service
    with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True
        mock_instance.search_code_examples.return_value = [
            {
                "id": "test_example",
                "title": "Test Method",
                "description": "Test description",
                "file_path": "/test/path.py",
                "extension_id": "test.ext",
                "line_start": 5,
                "line_end": 15,
                "code": "def test():\n    return True",
                "tags": ["test", "example"],
                "relevance_score": 0.88,
            }
        ]
        mock_service.return_value = mock_instance

        # Mock telemetry
        with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
            with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                result = await search_code_examples("test", rerank_k=5)

                # Check formatting elements
                assert result["success"]
                assert "Test Method" in result["result"]
                assert "File:**" in result["result"]
                assert "Extension:**" in result["result"]
                assert "Lines:**" in result["result"]
                assert "Relevance Score:**" in result["result"]
                assert "Description:**" in result["result"]
                assert "Code:**" in result["result"]
                assert "```python" in result["result"]
                assert "Tags:**" in result["result"]
                assert "test, example" in result["result"]


@pytest.mark.asyncio
async def test_exception_handling():
    """Test exception handling in search."""
    # Mock the code search service to raise an exception
    with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
        mock_instance = MagicMock()
        mock_instance.is_available.return_value = True
        mock_instance.search_code_examples.side_effect = Exception("Test exception")
        mock_service.return_value = mock_instance

        # Mock telemetry
        with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
            with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                result = await search_code_examples("test query", rerank_k=5)

                assert not result["success"]
                assert "Error searching code examples" in result["error"]


@pytest.mark.asyncio
async def test_python_code_validation():
    """Test that search_code_examples returns valid Python code from the FAISS database."""
    # Test queries - use common Kit operations that should be in the database
    test_queries = [
        "How to create an extension?",
        "How to use omni.kit.app?",
        "How to create a window?",
    ]

    all_passed = True
    total_code_blocks = 0

    for query in test_queries:
        logger.info(f"Testing Python validation for query: {query}")

        # Call the actual function without mocking to get real FAISS data
        result = await search_code_examples(query, rerank_k=3, enable_rerank=False)

        # Check if the function succeeded
        if not result["success"]:
            error = result.get("error", "Unknown error")
            # Skip if FAISS data is not available (expected in some environments)
            if "not available" in error:
                logger.info(f"Code search data not available, skipping validation test")
                pytest.skip("FAISS data not available")
                return
            continue

        if not result["result"] or "No code examples found" in result["result"]:
            continue

        # Extract code blocks from the result
        pattern = r"```python\n(.*?)\n```"
        code_blocks = re.findall(pattern, result["result"], re.DOTALL)

        if not code_blocks:
            continue

        # Validate each code block is valid Python
        for idx, code in enumerate(code_blocks):
            total_code_blocks += 1
            try:
                # Try to parse the code as Python AST
                ast.parse(code)
                logger.info(f"✓ Code block {idx + 1} from '{query}' is valid Python")
            except SyntaxError as e:
                all_passed = False
                logger.error(f"✗ Code block {idx + 1} from '{query}' has invalid syntax: {e}")
                pytest.fail(f"Code block {idx + 1} from '{query}' has invalid Python syntax: {e}")

    # Ensure we tested at least some code if service is available
    if total_code_blocks == 0:
        logger.warning("No code blocks found to validate - service may be unavailable")
        pytest.skip("No code blocks found to validate")

    assert all_passed, f"Some code blocks had invalid Python syntax"
    logger.info(f"✅ All {total_code_blocks} code blocks contain valid Python")
