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

"""Script to build FAISS database from UI functions with embedding vectors."""

import json
import logging
import os
import sys
from typing import List

# Add the parent directory to the path to allow importing from omni_ui_fns
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from omni_ui_fns.services.retrieval import create_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_faiss_database(
    vector_json_path: str, faiss_output_path: str, endpoint_url: str = None, api_key: str = ""
) -> None:
    """Build FAISS database from UI functions with embedding vectors.

    Args:
        vector_json_path: Path to the JSON file with embedding vectors
        faiss_output_path: Path to save the FAISS database
        endpoint_url: Optional custom endpoint URL for embeddings
        api_key: API key for authentication
    """
    logger.info(f"Starting FAISS database creation from {vector_json_path}")

    # Create embedder (needed for FAISS initialization)
    try:
        embedder = create_embeddings(endpoint_url, api_key)
        logger.info("Successfully created embedder")
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        raise

    # Load the vector JSON file
    try:
        with open(vector_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries from {vector_json_path}")
    except Exception as e:
        logger.error(f"Failed to load vector JSON file: {e}")
        raise

    # Prepare documents for FAISS
    documents = []
    embeddings = []

    for i, entry in enumerate(data, 1):
        try:
            # Extract required fields
            description = entry.get("description", "")
            function_body = entry.get("function_body", "")
            file_path = entry.get("file_path", "")
            function_name = entry.get("function_name", "unknown")
            class_name = entry.get("class_name", "unknown")
            line_number = entry.get("line_number", 0)
            embedding_vector = entry.get("embedding_vector", [])

            # Skip entries without embeddings
            if not embedding_vector:
                logger.warning(f"Entry {i} has no embedding vector, skipping")
                continue

            # Create document with description as page_content and metadata
            doc = Document(
                page_content=description,
                metadata={
                    "description": description,
                    "code": function_body,
                    "file_path": file_path,
                    "function_name": function_name,
                    "class_name": class_name,
                    "line_number": line_number,
                    "entry_index": i - 1,
                },
            )

            documents.append(doc)
            embeddings.append(embedding_vector)

            # Log progress every 500 entries
            if i % 500 == 0:
                logger.info(f"Processed {i}/{len(data)} entries ({i/len(data)*100:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to process entry {i}: {e}")
            continue

    logger.info(f"Prepared {len(documents)} documents for FAISS database")

    # Create FAISS database
    try:
        # Create FAISS index from documents and embeddings
        vectorstore = FAISS.from_embeddings(
            text_embeddings=[(doc.page_content, emb) for doc, emb in zip(documents, embeddings)],
            embedding=embedder,
            metadatas=[doc.metadata for doc in documents],
        )

        # Save the FAISS database
        os.makedirs(os.path.dirname(faiss_output_path), exist_ok=True)
        vectorstore.save_local(faiss_output_path)

        logger.info(f"Successfully saved FAISS database to {faiss_output_path}")
        logger.info(f"Database contains {len(documents)} entries")

    except Exception as e:
        logger.error(f"Failed to create/save FAISS database: {e}")
        raise


def main():
    """Main function to run the FAISS database creation."""
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")

    input_file = os.path.join(data_dir, "ui_functions_with_descriptions_vector.json")
    output_dir = os.path.join(data_dir, "ui_window_examples_faiss")

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return

    # Get API key from environment variable
    api_key = os.getenv("NVIDIA_API_KEY", "")

    if not api_key:
        logger.warning("No NVIDIA_API_KEY found in environment variables")

    # Build FAISS database
    try:
        build_faiss_database(vector_json_path=input_file, faiss_output_path=output_dir, api_key=api_key)
        logger.info("FAISS database creation completed successfully!")

    except Exception as e:
        logger.error(f"FAISS database creation failed: {e}")
        raise


if __name__ == "__main__":
    main()
