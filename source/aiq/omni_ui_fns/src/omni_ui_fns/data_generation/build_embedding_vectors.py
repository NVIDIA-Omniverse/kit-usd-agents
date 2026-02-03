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

"""Script to build embedding vectors for UI function descriptions."""

import json
import logging
import os
import sys
from typing import Any, Dict, List

# Add the parent directory to the path to allow importing from omni_ui_fns
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from omni_ui_fns.services.retrieval import create_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_embedding_vectors(
    input_file_path: str, output_file_path: str, endpoint_url: str = None, api_key: str = ""
) -> None:
    """Build embedding vectors for descriptions in UI functions JSON file.

    Args:
        input_file_path: Path to the input JSON file
        output_file_path: Path to save the output JSON file with vectors
        endpoint_url: Optional custom endpoint URL for embeddings
        api_key: API key for authentication
    """
    logger.info(f"Starting embedding vector creation for {input_file_path}")

    # Create embedder
    try:
        embedder = create_embeddings(endpoint_url, api_key)
        logger.info("Successfully created embedder")
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        raise

    # Load input JSON file
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries from {input_file_path}")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        raise

    # Process each entry and add embedding vectors
    processed_entries = []
    total_entries = len(data)

    for i, entry in enumerate(data, 1):
        try:
            # Get the description
            description = entry.get("description", "")

            if not description:
                logger.warning(f"Entry {i} has no description, skipping embedding")
                # Add empty vector field
                entry["embedding_vector"] = []
                processed_entries.append(entry)
                continue

            # Create embedding for the description
            embedding = embedder.embed_query(description)

            # Add the embedding vector to the entry
            entry["embedding_vector"] = embedding
            processed_entries.append(entry)

            # Log progress every 100 entries
            if i % 3 == 0:
                logger.info(f"Processed {i}/{total_entries} entries ({i/total_entries*100:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to process entry {i}: {e}")
            # Add empty vector on failure
            entry["embedding_vector"] = []
            processed_entries.append(entry)
            continue

    # Save the output JSON file
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(processed_entries, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved {len(processed_entries)} entries with embeddings to {output_file_path}")

    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        raise


def main():
    """Main function to run the embedding vector creation."""
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")

    input_file = os.path.join(data_dir, "ui_functions_with_descriptions.json")
    output_file = os.path.join(data_dir, "ui_functions_with_descriptions_vector.json")

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return

    # Get API key from environment variable
    api_key = os.getenv("NVIDIA_API_KEY", "")

    if not api_key:
        logger.warning("No NVIDIA_API_KEY found in environment variables")

    # Build embedding vectors
    try:
        build_embedding_vectors(input_file_path=input_file, output_file_path=output_file, api_key=api_key)
        logger.info("Embedding vector creation completed successfully!")

    except Exception as e:
        logger.error(f"Embedding vector creation failed: {e}")
        raise


if __name__ == "__main__":
    main()
