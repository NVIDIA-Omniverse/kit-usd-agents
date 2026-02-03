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
Build FAISS database from settings embeddings.
Creates a searchable vector database for finding relevant Kit settings.
"""

import json
import logging
import os
from datetime import datetime
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
    print("Please install: pip install langchain-nvidia-ai-endpoints")
    NVIDIA_EMBEDDINGS_AVAILABLE = False

# Configuration
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
SETTING_DATA_DIR = Path("setting_data")
EMBEDDINGS_FILE = SETTING_DATA_DIR / "settings_embeddings.json"
SETTINGS_FILE = SETTING_DATA_DIR / "setting_summary.json"
FAISS_OUTPUT_DIR = SETTING_DATA_DIR / "settings_faiss"

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


def create_setting_page_content(setting_key: str, setting_data: Dict[str, Any]) -> str:
    """Create page content for a setting document.

    Args:
        setting_key: The canonical setting path
        setting_data: The setting data dictionary

    Returns:
        A formatted string for the document's page_content
    """
    # Create a comprehensive text representation of the setting
    content_parts = []

    # The setting path is the most important for search
    content_parts.append(f"Setting: {setting_key}")

    # Add type and default value
    if setting_data.get("type"):
        content_parts.append(f"Type: {setting_data['type']}")

    if setting_data.get("default_value") is not None:
        default_str = str(setting_data["default_value"])
        content_parts.append(f"Default: {default_str}")

    # Add documentation
    if setting_data.get("documentation"):
        content_parts.append(f"Documentation: {setting_data['documentation']}")

    if setting_data.get("description"):
        content_parts.append(f"Description: {setting_data['description']}")

    # Add extension information
    extensions = setting_data.get("extensions", [])
    if extensions:
        # Include up to 5 extension names for context
        ext_sample = extensions[:5]
        if len(extensions) > 5:
            ext_str = f"{', '.join(ext_sample)}, and {len(extensions) - 5} more"
        else:
            ext_str = ", ".join(ext_sample)
        content_parts.append(f"Used by: {ext_str}")

    return "\n".join(content_parts)


def build_faiss_database(
    embeddings_file: Path,
    settings_file: Path,
    faiss_output_path: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str = "",
    endpoint_url: str = None,
) -> None:
    """Build FAISS database from settings embeddings.

    Args:
        embeddings_file: Path to the embeddings JSON file
        settings_file: Path to the settings summary JSON file
        faiss_output_path: Path to save the FAISS database
        api_key: API key for authentication
        endpoint_url: Optional custom endpoint URL
    """
    logger.info("=" * 60)
    logger.info("Building FAISS Database for Settings")
    logger.info("=" * 60)

    # Create embedder (needed for FAISS initialization)
    try:
        embedder = create_embeddings(api_key, endpoint_url, embedding_model)
        logger.info("Successfully created embedder")
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        raise

    # Load data files
    logger.info("Loading data files...")

    # Load embeddings
    if not embeddings_file.exists():
        logger.error(f"Embeddings file not found: {embeddings_file}")
        logger.error("Please run generate_settings_embeddings.py first")
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    with open(embeddings_file, "r") as f:
        embeddings_data = json.load(f)

    # Load settings for metadata
    if not settings_file.exists():
        logger.error(f"Settings file not found: {settings_file}")
        raise FileNotFoundError(f"Settings file not found: {settings_file}")

    with open(settings_file, "r") as f:
        settings_data = json.load(f)

    # Extract settings
    if "settings" in settings_data:
        settings = settings_data["settings"]
    else:
        settings = settings_data

    logger.info(f"Loaded {len(embeddings_data.get('settings', {}))} embeddings")
    logger.info(f"Loaded {len(settings)} settings metadata")

    # Prepare documents and embeddings for FAISS
    documents = []
    embedding_vectors = []
    skipped = 0

    # Get embeddings from the embeddings data
    settings_embeddings = embeddings_data.get("settings", {})

    for setting_key, embedding_data in settings_embeddings.items():
        try:
            # Get the embedding vector
            embedding_vector = embedding_data.get("embedding", [])

            # Skip entries without valid embeddings
            if not embedding_vector or len(embedding_vector) == 0:
                logger.warning(f"Setting {setting_key} has no embedding vector, skipping")
                skipped += 1
                continue

            # Get the full setting data for metadata
            setting_info = settings.get(setting_key, {})

            # Create page content
            page_content = create_setting_page_content(setting_key, setting_info)

            # Prepare metadata for the document
            metadata = {
                "setting_key": setting_key,
                "type": setting_info.get("type", "unknown"),
                "default_value": str(setting_info.get("default_value", "")),
                "has_documentation": bool(setting_info.get("documentation") or setting_info.get("description")),
                "usage_count": setting_info.get("usage_count", len(setting_info.get("found_in", []))),
                "extensions_count": len(setting_info.get("extensions", [])),
            }

            # Add prefix information for filtering
            if setting_key.startswith("/"):
                parts = setting_key.split("/")
                if len(parts) > 1:
                    metadata["prefix"] = parts[1]  # e.g., 'exts', 'app', 'persistent'

                    # For /exts/ settings, extract the extension name
                    if parts[1] == "exts" and len(parts) > 2:
                        metadata["extension_name"] = parts[2]

            # Add documentation if available
            if setting_info.get("documentation"):
                metadata["documentation"] = setting_info["documentation"][:500]  # Limit length
            elif setting_info.get("description"):
                metadata["documentation"] = setting_info["description"][:500]

            # Add sample extensions
            extensions = setting_info.get("extensions", [])
            if extensions:
                metadata["sample_extensions"] = ", ".join(extensions[:3])
                metadata["all_extensions"] = ", ".join(extensions)

            # Add found_in information (first few locations)
            found_in = setting_info.get("found_in", [])
            if found_in:
                metadata["sample_locations"] = ", ".join(found_in[:3])

            # Create document
            doc = Document(page_content=page_content, metadata=metadata)

            documents.append(doc)
            embedding_vectors.append(embedding_vector)

        except Exception as e:
            logger.error(f"Failed to process setting {setting_key}: {e}")
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
        logger.info(f"Database contains {len(documents)} setting entries")

        # Save metadata about the database
        metadata_file = faiss_output_path / "metadata.json"
        metadata = {
            "total_settings": len(documents),
            "skipped_settings": skipped,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "embedding_dimension": len(embedding_vectors[0]) if embedding_vectors else 0,
            "created_at": datetime.now().isoformat(),
            "source_files": {"embeddings": str(embeddings_file), "settings": str(settings_file)},
            "statistics": {"settings_by_prefix": {}, "settings_with_documentation": 0, "average_usage_count": 0},
        }

        # Calculate statistics
        prefix_counts = {}
        doc_count = 0
        total_usage = 0

        for doc in documents:
            # Count by prefix
            prefix = doc.metadata.get("prefix", "root")
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

            # Count documentation
            if doc.metadata.get("has_documentation"):
                doc_count += 1

            # Sum usage counts
            total_usage += doc.metadata.get("usage_count", 0)

        metadata["statistics"]["settings_by_prefix"] = prefix_counts
        metadata["statistics"]["settings_with_documentation"] = doc_count
        metadata["statistics"]["average_usage_count"] = total_usage / len(documents) if documents else 0

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_file}")

        # Log statistics
        logger.info("\nDatabase Statistics:")
        logger.info(f"  Settings by prefix:")
        for prefix, count in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {prefix}: {count}")
        logger.info(f"  Settings with documentation: {doc_count}/{len(documents)}")
        logger.info(f"  Average usage count: {metadata['statistics']['average_usage_count']:.1f}")

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
            settings_file=SETTINGS_FILE,
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
        logger.info(f"  - Load with: FAISS.load_local('{FAISS_OUTPUT_DIR}', embedder)")
        logger.info("  - Search with: vectorstore.similarity_search(query)")
        logger.info("\nExample search queries:")
        logger.info("  - 'viewport settings'")
        logger.info("  - 'rendering configuration'")
        logger.info("  - 'extension autoload settings'")

    except Exception as e:
        logger.error(f"FAISS database creation failed: {e}")
        raise


if __name__ == "__main__":
    main()
