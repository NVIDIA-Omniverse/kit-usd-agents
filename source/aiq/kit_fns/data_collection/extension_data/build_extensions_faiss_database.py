#!/usr/bin/env python3
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

"""
Build FAISS database from extension embeddings.
Creates a searchable vector database for finding relevant extensions.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import embeddings creator
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    NVIDIA_EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("Error: langchain_nvidia_ai_endpoints not available.")
    print("Please run: poetry install")
    NVIDIA_EMBEDDINGS_AVAILABLE = False

# Configuration
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
INPUT_DIR = Path("extension_detail")
EMBEDDINGS_FILE = INPUT_DIR / "extensions_embeddings.json"
DESCRIPTIONS_FILE = INPUT_DIR / "extensions_descriptions.json"
DATABASE_FILE = INPUT_DIR / "extensions_database.json"
FAISS_OUTPUT_DIR = INPUT_DIR / "extensions_faiss"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_embeddings(
    api_key: str = "", endpoint_url: str = None, embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> "NVIDIAEmbeddings":
    """Create NVIDIA embeddings instance.

    Args:
        api_key: NVIDIA API key for authentication
        endpoint_url: Optional custom endpoint URL

    Returns:
        NVIDIAEmbeddings instance
    """
    if not NVIDIA_EMBEDDINGS_AVAILABLE:
        raise ImportError("NVIDIA embeddings not available. Please install langchain-nvidia-ai-endpoints")

    if endpoint_url:
        return NVIDIAEmbeddings(
            base_url=endpoint_url,
            nvidia_api_key=api_key,
            truncate="END",
        )
    else:
        # Use default NVIDIA API
        return NVIDIAEmbeddings(
            model=embedding_model,
            nvidia_api_key=api_key,
            truncate="END",
        )


def build_faiss_database(
    embeddings_file: Path,
    descriptions_file: Path,
    database_file: Path,
    faiss_output_path: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str = "",
    endpoint_url: str = None,
) -> None:
    """Build FAISS database from extension embeddings.

    Args:
        embeddings_file: Path to the embeddings JSON file
        descriptions_file: Path to the descriptions JSON file
        database_file: Path to the main database JSON file
        faiss_output_path: Path to save the FAISS database
        embedding_model: Embedding model to use
        api_key: API key for authentication
        endpoint_url: Optional custom endpoint URL
    """
    logger.info("=" * 60)
    logger.info("Building FAISS Database for Extensions")
    logger.info("=" * 60)

    # Create embedder (needed for FAISS initialization)
    try:
        embedder = create_embeddings(api_key, endpoint_url, embedding_model)
        logger.info("Successfully created embedder")
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        raise

    # Load all three data files
    logger.info("Loading data files...")

    # Load embeddings
    if not embeddings_file.exists():
        logger.error(f"Embeddings file not found: {embeddings_file}")
        logger.error("Please run 'poetry run generate-embeddings' first")
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    with open(embeddings_file, "r") as f:
        embeddings_data = json.load(f)

    # Load descriptions
    if not descriptions_file.exists():
        logger.error(f"Descriptions file not found: {descriptions_file}")
        raise FileNotFoundError(f"Descriptions file not found: {descriptions_file}")

    with open(descriptions_file, "r") as f:
        descriptions_data = json.load(f)

    # Load main database for additional metadata
    if not database_file.exists():
        logger.error(f"Database file not found: {database_file}")
        raise FileNotFoundError(f"Database file not found: {database_file}")

    with open(database_file, "r") as f:
        database = json.load(f)

    # Get extensions from the main database
    if "extensions" in database:
        extensions_metadata = database["extensions"]
    else:
        extensions_metadata = {}

    logger.info(f"Loaded {len(embeddings_data.get('extensions', {}))} embeddings")

    # Prepare documents and embeddings for FAISS
    documents = []
    embedding_vectors = []
    skipped = 0

    # Get extensions from embeddings data
    extensions = embeddings_data.get("extensions", {})

    for ext_id, ext_embedding_data in extensions.items():
        try:
            # Get the embedding vector
            embedding_vector = ext_embedding_data.get("embedding", [])

            # Skip entries without valid embeddings
            if not embedding_vector or len(embedding_vector) == 0:
                logger.warning(f"Extension {ext_id} has no embedding vector, skipping")
                skipped += 1
                continue

            # Get the description text (for page_content)
            description_text = ""
            if ext_id in descriptions_data:
                description_text = descriptions_data[ext_id].get("description", f"Extension: {ext_id}")
            else:
                description_text = f"Extension: {ext_id}"

            # Get metadata from main database
            metadata = {}
            if ext_id in extensions_metadata:
                ext_meta = extensions_metadata[ext_id]
                metadata = {
                    "extension_id": ext_id,
                    "version": ext_meta.get("version", ""),
                    "title": ext_meta.get("title", ext_meta.get("description", "")),
                    "description": ext_meta.get("description", ""),
                    "long_description": ext_meta.get("long_description", ""),
                    "category": ext_meta.get("category", ""),
                    "keywords": ", ".join(ext_meta.get("keywords", [])),
                    "topics": ", ".join(ext_meta.get("topics", [])),
                    "has_python_api": ext_meta.get("has_python_api", False),
                    "has_overview": ext_meta.get("has_overview", False),
                    "total_modules": ext_meta.get("total_modules", 0),
                    "total_classes": ext_meta.get("total_classes", 0),
                    "total_methods": ext_meta.get("total_methods", 0),
                    "dependencies": ", ".join(ext_meta.get("dependencies", [])),
                    "codeatlas_token_count": ext_meta.get("codeatlas_token_count", 0),
                    "api_docs_token_count": ext_meta.get("api_docs_token_count", 0),
                    "overview_token_count": ext_meta.get("overview_token_count", 0),
                    "storage_path": ext_meta.get("storage_path", ""),
                }
            else:
                # Minimal metadata if not found in main database
                metadata = {
                    "extension_id": ext_id,
                    "version": ext_embedding_data.get("version", ""),
                    "has_overview": ext_embedding_data.get("has_overview", False),
                    "has_python_api": ext_embedding_data.get("has_python_api", False),
                }

            # Create document
            doc = Document(page_content=description_text, metadata=metadata)

            documents.append(doc)
            embedding_vectors.append(embedding_vector)

        except Exception as e:
            logger.error(f"Failed to process extension {ext_id}: {e}")
            skipped += 1
            continue

    logger.info(f"Prepared {len(documents)} documents for FAISS database ({skipped} skipped)")

    if len(documents) == 0:
        logger.error("No valid documents to create FAISS database")
        return

    # Create FAISS database
    try:
        logger.info("Creating FAISS index...")

        # Create FAISS index from documents and embeddings
        vectorstore = FAISS.from_embeddings(
            text_embeddings=[(doc.page_content, emb) for doc, emb in zip(documents, embedding_vectors)],
            embedding=embedder,
            metadatas=[doc.metadata for doc in documents],
        )

        # Save the FAISS database
        logger.info(f"Saving FAISS database to: {faiss_output_path}")
        faiss_output_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(faiss_output_path))

        logger.info(f"Successfully saved FAISS database to {faiss_output_path}")
        logger.info(f"Database contains {len(documents)} extension entries")

        # Save metadata about the database
        metadata_file = faiss_output_path / "metadata.json"
        metadata = {
            "total_extensions": len(documents),
            "skipped_extensions": skipped,
            "embedding_model": embedding_model,
            "embedding_dimension": len(embedding_vectors[0]) if embedding_vectors else 0,
            "created_at": None,
        }

        from datetime import datetime

        metadata["created_at"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_file}")

    except Exception as e:
        logger.error(f"Failed to create/save FAISS database: {e}")
        raise


def main():
    """Main function to run the FAISS database creation."""
    # Get API key from environment variable
    api_key = os.getenv("NVIDIA_API_KEY", "")

    if not api_key:
        logger.warning("No NVIDIA_API_KEY found in environment variables")
        logger.warning("You can set it with: export NVIDIA_API_KEY='your-api-key'")
        logger.warning("Get your API key from: https://build.nvidia.com/")

        # The embedder is needed for FAISS initialization even if we have embeddings
        response = input("Continue without API key? (y/n): ")
        if response.lower() != "y":
            return

    # Optional: Override endpoint URL if needed
    endpoint_url = os.getenv("EMBEDDING_ENDPOINT_URL", None)
    if endpoint_url:
        logger.info(f"Using custom endpoint: {endpoint_url}")

    # Build FAISS database
    try:
        build_faiss_database(
            embeddings_file=EMBEDDINGS_FILE,
            descriptions_file=DESCRIPTIONS_FILE,
            database_file=DATABASE_FILE,
            faiss_output_path=FAISS_OUTPUT_DIR,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            api_key=api_key,
            endpoint_url=endpoint_url,
        )

        logger.info("\n" + "=" * 60)
        logger.info("FAISS database creation completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Database location: {FAISS_OUTPUT_DIR}")
        logger.info("\nYou can now use this database for similarity search:")
        logger.info("  - Load with: FAISS.load_local('{FAISS_OUTPUT_DIR}', embedder)")
        logger.info("  - Search with: vectorstore.similarity_search(query)")

    except Exception as e:
        logger.error(f"FAISS database creation failed: {e}")
        raise


if __name__ == "__main__":
    main()
