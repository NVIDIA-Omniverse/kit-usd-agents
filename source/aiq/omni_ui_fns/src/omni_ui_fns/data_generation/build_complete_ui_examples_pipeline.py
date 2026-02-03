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

"""Complete pipeline to build UI window examples database and test retrieval."""

import logging
import os
import sys
from typing import Optional

# Add the parent directory to the path to allow importing from omni_ui_fns
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from omni_ui_fns.data_generation.build_embedding_vectors import build_embedding_vectors
from omni_ui_fns.data_generation.build_faiss_database import build_faiss_database
from omni_ui_fns.services.ui_window_examples_retrieval import (
    create_ui_window_examples_retriever,
    get_ui_window_examples,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_file_paths():
    """Setup and return all required file paths."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")

    paths = {
        "input_json": os.path.join(data_dir, "ui_functions_with_descriptions.json"),
        "vector_json": os.path.join(data_dir, "ui_functions_with_descriptions_vector.json"),
        "faiss_db": os.path.join(data_dir, "ui_window_examples_faiss"),
        "data_dir": data_dir,
    }

    return paths


def check_prerequisites(paths: dict) -> bool:
    """Check if all prerequisite files exist."""
    if not os.path.exists(paths["input_json"]):
        logger.error(f"Input JSON file not found: {paths['input_json']}")
        return False

    logger.info("Prerequisites check passed")
    return True


def step_1_build_embeddings(paths: dict, api_key: str, force_rebuild: bool = False) -> bool:
    """Step 1: Build embedding vectors if they don't exist."""
    if os.path.exists(paths["vector_json"]) and not force_rebuild:
        logger.info("Embedding vectors file already exists, skipping Step 1")
        return True

    logger.info("Step 1: Building embedding vectors...")
    try:
        build_embedding_vectors(
            input_file_path=paths["input_json"], output_file_path=paths["vector_json"], api_key=api_key
        )
        logger.info("Step 1 completed successfully")
        return True
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        return False


def step_2_build_faiss_database(paths: dict, api_key: str, force_rebuild: bool = False) -> bool:
    """Step 2: Build FAISS database from embedding vectors."""
    if os.path.exists(paths["faiss_db"]) and not force_rebuild:
        logger.info("FAISS database already exists, skipping Step 2")
        return True

    if not os.path.exists(paths["vector_json"]):
        logger.error("Vector JSON file not found for Step 2")
        return False

    logger.info("Step 2: Building FAISS database...")
    try:
        build_faiss_database(
            vector_json_path=paths["vector_json"], faiss_output_path=paths["faiss_db"], api_key=api_key
        )
        logger.info("Step 2 completed successfully")
        return True
    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        return False


def step_3_test_retrieval(paths: dict, api_key: str) -> bool:
    """Step 3: Test the retrieval system."""
    if not os.path.exists(paths["faiss_db"]):
        logger.error("FAISS database not found for Step 3")
        return False

    logger.info("Step 3: Testing retrieval system...")
    try:
        # Create retriever
        retriever = create_ui_window_examples_retriever(faiss_index_path=paths["faiss_db"], api_key=api_key, top_k=3)

        # Test with a sample query
        test_query = "create window with buttons and controls"
        logger.info(f"Testing with query: '{test_query}'")

        # Get structured results
        results = get_ui_window_examples(test_query, retriever, top_k=3, format_type="structured")

        logger.info(f"Retrieved {len(results)} results")

        # Display sample results
        for i, result in enumerate(results[:2], 1):
            logger.info(f"Result {i}:")
            logger.info(f"  Function: {result['class_name']}.{result['function_name']}()")
            logger.info(f"  File: {result['file_path']}")
            logger.info(f"  Description: {result['description'][:100]}...")

        logger.info("Step 3 completed successfully")
        return True

    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        return False


def run_pipeline(
    api_key: Optional[str] = None,
    force_rebuild_embeddings: bool = False,
    force_rebuild_faiss: bool = False,
    skip_test: bool = False,
) -> bool:
    """Run the complete UI examples pipeline.

    Args:
        api_key: NVIDIA API key (will use environment variable if not provided)
        force_rebuild_embeddings: Force rebuild of embedding vectors
        force_rebuild_faiss: Force rebuild of FAISS database
        skip_test: Skip the retrieval test

    Returns:
        True if pipeline completed successfully, False otherwise
    """
    logger.info("Starting UI Window Examples Pipeline")
    logger.info("=" * 60)

    # Setup
    if api_key is None:
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            logger.warning("No NVIDIA API key provided or found in environment")

    paths = setup_file_paths()
    logger.info(f"Data directory: {paths['data_dir']}")

    # Check prerequisites
    if not check_prerequisites(paths):
        return False

    # Step 1: Build embeddings
    if not step_1_build_embeddings(paths, api_key, force_rebuild_embeddings):
        return False

    # Step 2: Build FAISS database
    if not step_2_build_faiss_database(paths, api_key, force_rebuild_faiss):
        return False

    # Step 3: Test retrieval
    if not skip_test:
        if not step_3_test_retrieval(paths, api_key):
            return False

    logger.info("=" * 60)
    logger.info("UI Window Examples Pipeline completed successfully!")
    logger.info(f"FAISS database location: {paths['faiss_db']}")
    logger.info("You can now use UIWindowExamplesRetriever to search for UI examples")

    return True


def main():
    """Main function with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Build UI Window Examples Pipeline")
    parser.add_argument("--api-key", help="NVIDIA API key")
    parser.add_argument("--force-embeddings", action="store_true", help="Force rebuild embedding vectors")
    parser.add_argument("--force-faiss", action="store_true", help="Force rebuild FAISS database")
    parser.add_argument("--skip-test", action="store_true", help="Skip retrieval test")

    args = parser.parse_args()

    success = run_pipeline(
        api_key=args.api_key,
        force_rebuild_embeddings=args.force_embeddings,
        force_rebuild_faiss=args.force_faiss,
        skip_test=args.skip_test,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
