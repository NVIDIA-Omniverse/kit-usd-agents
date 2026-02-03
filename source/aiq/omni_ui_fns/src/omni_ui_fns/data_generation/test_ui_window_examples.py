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

"""Test script for UI window examples FAISS database and retrieval."""

import logging
import os
import sys

# Add the parent directory to the path to allow importing from omni_ui_fns
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from omni_ui_fns.services.ui_window_examples_retrieval import (
    create_ui_window_examples_retriever,
    get_ui_window_examples,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ui_window_examples_search():
    """Test the UI window examples search functionality."""
    logger.info("Testing UI Window Examples Search")

    # Get API key from environment
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        logger.warning("No NVIDIA_API_KEY found, search may not work")

    # Create retriever
    try:
        retriever = create_ui_window_examples_retriever(api_key=api_key, top_k=5)
        logger.info("Successfully created UI Window Examples Retriever")
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        return

    # Test queries
    test_queries = [
        "create window with buttons",
        "animation curve simplification",
        "dialog window with text",
        "UI with sliders and controls",
        "build user interface",
    ]

    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing query: '{query}'")
        logger.info("=" * 60)

        try:
            # Test structured results
            structured_results = get_ui_window_examples(query, retriever, top_k=3, format_type="structured")

            logger.info(f"Found {len(structured_results)} structured results")

            for i, result in enumerate(structured_results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Function: {result['class_name']}.{result['function_name']}()")
                print(f"File: {result['file_path']}")
                print(f"Description: {result['description'][:100]}...")
                print(f"Code preview: {result['code'][:150].replace(chr(10), ' ')}...")

            # Test formatted results
            formatted_results = get_ui_window_examples(query, retriever, top_k=2, format_type="formatted")

            logger.info("\nFormatted results preview:")
            print(formatted_results[:500] + "..." if len(formatted_results) > 500 else formatted_results)

        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")

        print("\n" + "=" * 60 + "\n")


def main():
    """Main test function."""
    # Check if FAISS database exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    faiss_path = os.path.join(data_dir, "ui_window_examples_faiss")

    if not os.path.exists(faiss_path):
        logger.error(f"FAISS database not found at: {faiss_path}")
        logger.info("Please run build_faiss_database.py first to create the database")
        return

    logger.info(f"Found FAISS database at: {faiss_path}")

    # Run tests
    test_ui_window_examples_search()


if __name__ == "__main__":
    main()
