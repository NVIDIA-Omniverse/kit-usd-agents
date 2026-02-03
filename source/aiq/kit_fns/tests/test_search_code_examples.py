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

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from kit_fns.functions.search_code_examples import get_code_search_service, search_code_examples
from kit_fns.services.code_search_service import CodeSearchService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestSearchCodeExamples:
    """Test cases for search_code_examples function."""

    def __init__(self):
        """Initialize test suite."""
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "[OK]"
        else:
            self.failed_tests += 1
            status = "[FAIL]"

        log_message = f"{status} {test_name}"
        if message:
            log_message += f": {message}"

        logger.info(log_message)
        self.test_results.append({"name": test_name, "passed": passed, "message": message})

    async def test_valid_query(self):
        """Test with a valid query."""
        test_name = "test_valid_query"
        try:
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
                with patch(
                    "kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock
                ):
                    with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                        mock_telemetry.capture_call = AsyncMock()

                        result = await search_code_examples("test query", top_k=5)

                        if result["success"] and "Sample Method" in result["result"]:
                            self.log_test(test_name, True, "Valid query returned expected results")
                        else:
                            self.log_test(test_name, False, f"Unexpected result: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_empty_query(self):
        """Test with an empty query."""
        test_name = "test_empty_query"
        try:
            # Mock telemetry
            with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
                with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                    mock_telemetry.capture_call = AsyncMock()

                    result = await search_code_examples("", top_k=5)

                    if not result["success"] and "cannot be empty" in result["error"]:
                        self.log_test(test_name, True, "Empty query correctly rejected")
                    else:
                        self.log_test(test_name, False, f"Expected validation error, got: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_whitespace_query(self):
        """Test with a whitespace-only query."""
        test_name = "test_whitespace_query"
        try:
            # Mock telemetry
            with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
                with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                    mock_telemetry.capture_call = AsyncMock()

                    result = await search_code_examples("   ", top_k=5)

                    if not result["success"] and "cannot be empty" in result["error"]:
                        self.log_test(test_name, True, "Whitespace query correctly rejected")
                    else:
                        self.log_test(test_name, False, f"Expected validation error, got: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_invalid_top_k(self):
        """Test with invalid top_k parameter."""
        test_name = "test_invalid_top_k"
        try:
            # Mock telemetry
            with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
                with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                    mock_telemetry.capture_call = AsyncMock()

                    result = await search_code_examples("test query", top_k=0)

                    if not result["success"] and "must be positive" in result["error"]:
                        self.log_test(test_name, True, "Invalid top_k correctly rejected")
                    else:
                        self.log_test(test_name, False, f"Expected validation error, got: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_negative_top_k(self):
        """Test with negative top_k parameter."""
        test_name = "test_negative_top_k"
        try:
            # Mock telemetry
            with patch("kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock):
                with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                    mock_telemetry.capture_call = AsyncMock()

                    result = await search_code_examples("test query", top_k=-5)

                    if not result["success"] and "must be positive" in result["error"]:
                        self.log_test(test_name, True, "Negative top_k correctly rejected")
                    else:
                        self.log_test(test_name, False, f"Expected validation error, got: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_service_unavailable(self):
        """Test when code search service is unavailable."""
        test_name = "test_service_unavailable"
        try:
            # Mock the code search service as unavailable
            with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_service.return_value = mock_instance

                # Mock telemetry
                with patch(
                    "kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock
                ):
                    with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                        mock_telemetry.capture_call = AsyncMock()

                        result = await search_code_examples("test query", top_k=5)

                        if not result["success"] and "not available" in result["error"]:
                            self.log_test(test_name, True, "Service unavailable correctly handled")
                        else:
                            self.log_test(test_name, False, f"Expected unavailable error, got: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_no_results_found(self):
        """Test when no code examples are found."""
        test_name = "test_no_results_found"
        try:
            # Mock the code search service with empty results
            with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.search_code_examples.return_value = []
                mock_service.return_value = mock_instance

                # Mock telemetry
                with patch(
                    "kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock
                ):
                    with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                        mock_telemetry.capture_call = AsyncMock()

                        result = await search_code_examples("nonexistent query", top_k=5)

                        if result["success"] and "No code examples found" in result["result"]:
                            self.log_test(test_name, True, "No results case handled correctly")
                        else:
                            self.log_test(test_name, False, f"Expected no results message, got: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_multiple_results(self):
        """Test with multiple search results."""
        test_name = "test_multiple_results"
        try:
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
                with patch(
                    "kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock
                ):
                    with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                        mock_telemetry.capture_call = AsyncMock()

                        result = await search_code_examples("test query", top_k=10)

                        if (
                            result["success"]
                            and "Found 3 relevant examples" in result["result"]
                            and "Method 1" in result["result"]
                            and "Method 2" in result["result"]
                            and "Method 3" in result["result"]
                        ):
                            self.log_test(test_name, True, "Multiple results formatted correctly")
                        else:
                            self.log_test(test_name, False, f"Expected 3 formatted results, got: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_result_formatting(self):
        """Test that results are properly formatted."""
        test_name = "test_result_formatting"
        try:
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
                with patch(
                    "kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock
                ):
                    with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                        mock_telemetry.capture_call = AsyncMock()

                        result = await search_code_examples("test", top_k=5)

                        # Check formatting elements
                        checks = [
                            result["success"],
                            "Test Method" in result["result"],
                            "File:**" in result["result"],
                            "Extension:**" in result["result"],
                            "Lines:**" in result["result"],
                            "Relevance Score:**" in result["result"],
                            "Description:**" in result["result"],
                            "Code:**" in result["result"],
                            "```python" in result["result"],
                            "Tags:**" in result["result"],
                            "test, example" in result["result"],
                        ]

                        if all(checks):
                            self.log_test(test_name, True, "Result formatting includes all required elements")
                        else:
                            self.log_test(
                                test_name,
                                False,
                                f"Missing formatting elements. Passed checks: {sum(checks)}/{len(checks)}",
                            )
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def test_exception_handling(self):
        """Test exception handling in search."""
        test_name = "test_exception_handling"
        try:
            # Mock the code search service to raise an exception
            with patch("kit_fns.functions.search_code_examples.get_code_search_service") as mock_service:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.search_code_examples.side_effect = Exception("Test exception")
                mock_service.return_value = mock_instance

                # Mock telemetry
                with patch(
                    "kit_fns.functions.search_code_examples.ensure_telemetry_initialized", new_callable=AsyncMock
                ):
                    with patch("kit_fns.functions.search_code_examples.telemetry") as mock_telemetry:
                        mock_telemetry.capture_call = AsyncMock()

                        result = await search_code_examples("test query", top_k=5)

                        if not result["success"] and "Error searching code examples" in result["error"]:
                            self.log_test(test_name, True, "Exception handled gracefully")
                        else:
                            self.log_test(test_name, False, f"Expected error handling, got: {result}")
        except Exception as e:
            self.log_test(test_name, False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all test cases."""
        logger.info("=" * 80)
        logger.info("Starting test suite for search_code_examples")
        logger.info("=" * 80)

        # Run all test methods
        await self.test_valid_query()
        await self.test_empty_query()
        await self.test_whitespace_query()
        await self.test_invalid_top_k()
        await self.test_negative_top_k()
        await self.test_service_unavailable()
        await self.test_no_results_found()
        await self.test_multiple_results()
        await self.test_result_formatting()
        await self.test_exception_handling()

        # Print summary
        logger.info("=" * 80)
        logger.info("Test Summary")
        logger.info("=" * 80)
        logger.info(f"Total tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        logger.info(f"Success rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        logger.info("=" * 80)

        # Return exit code
        return 0 if self.failed_tests == 0 else 1


async def main():
    """Main test execution."""
    test_suite = TestSearchCodeExamples()
    exit_code = await test_suite.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
