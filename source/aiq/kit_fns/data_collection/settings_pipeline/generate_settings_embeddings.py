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
Generate embedding vectors for Kit settings.
Creates embeddings using NVIDIA API and saves them to a separate embeddings file.
Each setting's full JSON entry is used as the document to embed.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import LangChain NVIDIA embeddings
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
SETTINGS_FILE = SETTING_DATA_DIR / "setting_summary.json"
OUTPUT_FILE = SETTING_DATA_DIR / "settings_embeddings.json"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_embeddings(api_key: str = "", endpoint_url: Optional[str] = None) -> Optional["NVIDIAEmbeddings"]:
    """Create NVIDIA embeddings instance.

    Args:
        api_key: NVIDIA API key for authentication
        endpoint_url: Optional custom endpoint URL

    Returns:
        NVIDIAEmbeddings instance or None if not available
    """
    if not NVIDIA_EMBEDDINGS_AVAILABLE:
        logger.error("NVIDIA embeddings not available. Please install langchain-nvidia-ai-endpoints")
        return None

    try:
        if endpoint_url:
            embedder = NVIDIAEmbeddings(
                base_url=endpoint_url,
                nvidia_api_key=api_key,
                truncate="END",
            )
        else:
            # Use default NVIDIA API
            embedder = NVIDIAEmbeddings(
                model=DEFAULT_EMBEDDING_MODEL,
                nvidia_api_key=api_key,
                truncate="END",
            )
        logger.info(f"Successfully created embedder with model: {DEFAULT_EMBEDDING_MODEL}")
        return embedder
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        return None


def create_setting_document(setting_key: str, setting_data: Dict[str, Any]) -> str:
    """Create a document string from a setting entry for embedding.

    The document includes the setting key and all its metadata to create
    a comprehensive representation for semantic search.

    Args:
        setting_key: The canonical setting path (e.g., /exts/omni.kit.window/enabled)
        setting_data: The setting data dictionary

    Returns:
        A formatted string containing all setting information
    """
    # Build a comprehensive document from the setting data
    doc_parts = []

    # Add the setting key itself (important for search)
    doc_parts.append(f"Setting: {setting_key}")

    # Add type information
    if setting_data.get("type"):
        doc_parts.append(f"Type: {setting_data['type']}")

    # Add default value
    if setting_data.get("default_value") is not None:
        doc_parts.append(f"Default: {setting_data['default_value']}")

    # Add documentation/description
    if setting_data.get("documentation"):
        doc_parts.append(f"Documentation: {setting_data['documentation']}")

    if setting_data.get("description"):
        doc_parts.append(f"Description: {setting_data['description']}")

    # Add extensions that use this setting
    if setting_data.get("extensions"):
        extensions_str = ", ".join(setting_data["extensions"][:10])  # Limit to first 10
        doc_parts.append(f"Used by extensions: {extensions_str}")

    # Add usage count for importance
    usage_count = setting_data.get("usage_count", len(setting_data.get("found_in", [])))
    if usage_count > 0:
        doc_parts.append(f"Usage count: {usage_count}")

    # Combine all parts into a single document
    document = "\n".join(doc_parts)
    return document


def generate_embeddings_for_settings(
    settings_file: Path,
    output_file: Path,
    embeddings_model: str,
    api_key: str = "",
    endpoint_url: Optional[str] = None,
    batch_size: int = 10,
) -> None:
    """Generate embeddings for all settings.

    Args:
        settings_file: Path to the settings summary JSON file
        output_file: Path to save the embeddings database
        api_key: NVIDIA API key
        endpoint_url: Optional custom endpoint URL
        batch_size: Number of settings to process in batch
    """
    # Load settings
    if not settings_file.exists():
        logger.error(f"Settings file not found: {settings_file}")
        logger.error("Please run scan_extension_settings.py first")
        return

    logger.info(f"Loading settings from: {settings_file}")
    with open(settings_file, "r") as f:
        settings_data = json.load(f)

    # Extract settings dictionary
    if "settings" in settings_data:
        settings = settings_data["settings"]
    else:
        # Assume the entire file is the settings dictionary
        settings = settings_data

    logger.info(f"Found {len(settings)} settings to process")

    # Create embedder
    embedder = create_embeddings(api_key, endpoint_url)
    if not embedder:
        logger.error("Failed to create embedder. Exiting.")
        return

    # Process settings and generate embeddings
    embeddings_database = {
        "metadata": {
            "model": embeddings_model,
            "total_settings": len(settings),
            "embedding_dimension": None,  # Will be set after first embedding
            "generated_at": None,
            "source_file": str(settings_file),
        },
        "settings": {},
    }

    processed = 0
    failed = 0

    # Convert to list for batch processing
    settings_list = list(settings.items())

    for i in range(0, len(settings_list), batch_size):
        batch = settings_list[i : i + batch_size]
        batch_texts = []
        batch_keys = []

        # Prepare batch
        for setting_key, setting_data in batch:
            # Create document text for embedding
            document = create_setting_document(setting_key, setting_data)
            batch_texts.append(document)
            batch_keys.append(setting_key)

        if not batch_texts:
            continue

        try:
            # Generate embeddings for batch
            batch_num = i // batch_size + 1
            total_batches = (len(settings_list) + batch_size - 1) // batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch_texts)} settings")

            if len(batch_texts) == 1:
                # Single item - use embed_query
                embeddings = [embedder.embed_query(batch_texts[0])]
            else:
                # Multiple items - use embed_documents
                embeddings = embedder.embed_documents(batch_texts)

            # Store embeddings with metadata
            for setting_key, embedding, (_, setting_data) in zip(batch_keys, embeddings, batch):
                embeddings_database["settings"][setting_key] = {
                    "embedding": embedding,
                    "type": setting_data.get("type", "unknown"),
                    "default_value": setting_data.get("default_value"),
                    "extensions_count": len(setting_data.get("extensions", [])),
                    "usage_count": setting_data.get("usage_count", len(setting_data.get("found_in", []))),
                    "has_documentation": bool(setting_data.get("documentation") or setting_data.get("description")),
                }
                processed += 1

                # Set embedding dimension from first successful embedding
                if embeddings_database["metadata"]["embedding_dimension"] is None:
                    embeddings_database["metadata"]["embedding_dimension"] = len(embedding)
                    logger.info(f"Embedding dimension: {len(embedding)}")

            # Log progress
            progress = (i + len(batch)) / len(settings_list) * 100
            logger.info(f"Progress: {processed}/{len(settings)} settings ({progress:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to process batch starting at {i}: {e}")
            # Add empty embeddings for failed batch
            for setting_key in batch_keys:
                embeddings_database["settings"][setting_key] = {
                    "embedding": [],
                    "type": settings.get(setting_key, {}).get("type", "unknown"),
                    "default_value": settings.get(setting_key, {}).get("default_value"),
                    "extensions_count": len(settings.get(setting_key, {}).get("extensions", [])),
                    "usage_count": settings.get(setting_key, {}).get("usage_count", 0),
                    "has_documentation": bool(settings.get(setting_key, {}).get("documentation")),
                    "error": str(e),
                }
                failed += 1

    # Add timestamp
    embeddings_database["metadata"]["generated_at"] = datetime.now().isoformat()
    embeddings_database["metadata"]["successful_embeddings"] = processed
    embeddings_database["metadata"]["failed_embeddings"] = failed

    # Save embeddings database
    logger.info(f"Saving embeddings database to: {output_file}")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(embeddings_database, f, indent=2)

    # Summary
    logger.info("=" * 60)
    logger.info("Settings Embedding Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Total settings: {len(settings)}")
    logger.info(f"Successful embeddings: {processed}")
    logger.info(f"Failed embeddings: {failed}")
    if embeddings_database["metadata"]["embedding_dimension"]:
        logger.info(f"Embedding dimension: {embeddings_database['metadata']['embedding_dimension']}")
    logger.info(f"Output saved to: {output_file}")


def main():
    """Main function to run embedding generation."""
    # Get API key from environment variable
    api_key = os.getenv("NVIDIA_API_KEY", "")

    if not api_key:
        logger.warning("No NVIDIA_API_KEY found in environment variables")
        logger.warning("You can set it with: export NVIDIA_API_KEY='your-api-key'")
        logger.warning("Get your API key from: https://build.nvidia.com/")

        # Try to continue anyway - some endpoints might not require API key
        response = input("Continue without API key? (y/n): ")
        if response.lower() != "y":
            return

    # Optional: Override endpoint URL if needed
    endpoint_url = os.getenv("EMBEDDING_ENDPOINT_URL", None)
    if endpoint_url:
        logger.info(f"Using custom endpoint: {endpoint_url}")

    # Check if langchain-nvidia-ai-endpoints is available
    if not NVIDIA_EMBEDDINGS_AVAILABLE:
        logger.error("Required package not installed.")
        logger.error("Install with: pip install langchain-nvidia-ai-endpoints")
        return

    # Generate embeddings
    try:
        generate_embeddings_for_settings(
            settings_file=SETTINGS_FILE,
            output_file=OUTPUT_FILE,
            embeddings_model=DEFAULT_EMBEDDING_MODEL,
            api_key=api_key,
            endpoint_url=endpoint_url,
            batch_size=10,  # Process 10 settings at a time
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
