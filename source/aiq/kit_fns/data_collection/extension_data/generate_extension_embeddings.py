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
Generate embedding vectors for extension descriptions.
Creates embeddings using NVIDIA API and saves them to a separate database file.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import LangChain NVIDIA embeddings
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    NVIDIA_EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("Error: langchain_nvidia_ai_endpoints not available.")
    print("Please run: poetry install")
    NVIDIA_EMBEDDINGS_AVAILABLE = False

# Default Configuration
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
DEFAULT_INPUT_DIR = Path("extension_detail")
DEFAULT_DESCRIPTIONS_FILE = "extensions_descriptions.json"
DEFAULT_OUTPUT_FILE = "extensions_embeddings.json"
DEFAULT_BATCH_SIZE = 10

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embedding vectors for extension descriptions using NVIDIA API"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing descriptions (default: {DEFAULT_INPUT_DIR})",
    )

    parser.add_argument(
        "--descriptions-file",
        type=str,
        default=DEFAULT_DESCRIPTIONS_FILE,
        help=f"Descriptions JSON filename (default: {DEFAULT_DESCRIPTIONS_FILE})",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output embeddings JSON filename (default: {DEFAULT_OUTPUT_FILE})",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("NVIDIA_API_KEY", ""),
        help="NVIDIA API key (default: from NVIDIA_API_KEY env var)",
    )

    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=os.getenv("EMBEDDING_ENDPOINT_URL", None),
        help="Custom embedding endpoint URL (default: from EMBEDDING_ENDPOINT_URL env var)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of descriptions to process in batch (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument("--force", action="store_true", help="Continue without API key (for testing)")

    return parser.parse_args()


def create_embeddings(
    api_key: str = "", endpoint_url: Optional[str] = None, model: str = DEFAULT_EMBEDDING_MODEL
) -> Optional["NVIDIAEmbeddings"]:
    """Create NVIDIA embeddings instance.

    Args:
        api_key: NVIDIA API key for authentication
        endpoint_url: Optional custom endpoint URL
        model: Embedding model to use

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
                model=model,
                nvidia_api_key=api_key,
                truncate="END",
            )
        logger.info(f"Successfully created embedder with model: {model}")
        return embedder
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        return None


def generate_embeddings_for_extensions(
    descriptions_file: Path,
    output_file: Path,
    api_key: str = "",
    endpoint_url: Optional[str] = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 10,
) -> None:
    """Generate embeddings for all extension descriptions.

    Args:
        descriptions_file: Path to the descriptions JSON file
        output_file: Path to save the embeddings database
        api_key: NVIDIA API key
        endpoint_url: Optional custom endpoint URL
        model: Embedding model to use
        batch_size: Number of descriptions to process in batch
    """
    # Load descriptions
    if not descriptions_file.exists():
        logger.error(f"Descriptions file not found: {descriptions_file}")
        logger.error("Please run 'poetry run generate-embeddings-desc' first")
        return

    logger.info(f"Loading descriptions from: {descriptions_file}")
    with open(descriptions_file, "r") as f:
        descriptions_data = json.load(f)

    logger.info(f"Found {len(descriptions_data)} extension descriptions")

    # Create embedder
    embedder = create_embeddings(api_key, endpoint_url, model)
    if not embedder:
        logger.error("Failed to create embedder. Exiting.")
        return

    # Process extensions and generate embeddings
    embeddings_database = {
        "metadata": {
            "model": model,
            "total_extensions": len(descriptions_data),
            "embedding_dimension": None,  # Will be set after first embedding
            "generated_at": None,
        },
        "extensions": {},
    }

    processed = 0
    failed = 0

    # Convert to list for batch processing
    extensions_list = list(descriptions_data.items())

    for i in range(0, len(extensions_list), batch_size):
        batch = extensions_list[i : i + batch_size]
        batch_texts = []
        batch_ids = []

        # Prepare batch
        for ext_id, ext_data in batch:
            if isinstance(ext_data, dict) and "description" in ext_data:
                batch_texts.append(ext_data["description"])
                batch_ids.append(ext_id)

        if not batch_texts:
            continue

        try:
            # Generate embeddings for batch
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_texts)} extensions")

            if len(batch_texts) == 1:
                # Single item - use embed_query
                embeddings = [embedder.embed_query(batch_texts[0])]
            else:
                # Multiple items - use embed_documents
                embeddings = embedder.embed_documents(batch_texts)

            # Store embeddings
            for ext_id, embedding in zip(batch_ids, embeddings):
                embeddings_database["extensions"][ext_id] = {
                    "embedding": embedding,
                    "version": descriptions_data[ext_id].get("version", ""),
                    "token_count": descriptions_data[ext_id].get("token_count", 0),
                    "has_overview": descriptions_data[ext_id].get("has_overview", False),
                    "has_python_api": descriptions_data[ext_id].get("has_python_api", False),
                }
                processed += 1

                # Set embedding dimension from first successful embedding
                if embeddings_database["metadata"]["embedding_dimension"] is None:
                    embeddings_database["metadata"]["embedding_dimension"] = len(embedding)
                    logger.info(f"Embedding dimension: {len(embedding)}")

            # Log progress
            progress = (i + len(batch)) / len(extensions_list) * 100
            logger.info(f"Progress: {processed}/{len(descriptions_data)} extensions ({progress:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to process batch starting at {i}: {e}")
            # Add empty embeddings for failed batch
            for ext_id in batch_ids:
                embeddings_database["extensions"][ext_id] = {
                    "embedding": [],
                    "version": descriptions_data[ext_id].get("version", ""),
                    "token_count": descriptions_data[ext_id].get("token_count", 0),
                    "has_overview": descriptions_data[ext_id].get("has_overview", False),
                    "has_python_api": descriptions_data[ext_id].get("has_python_api", False),
                    "error": str(e),
                }
                failed += 1

    # Add timestamp
    from datetime import datetime

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
    logger.info("Embedding Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Total extensions: {len(descriptions_data)}")
    logger.info(f"Successful embeddings: {processed}")
    logger.info(f"Failed embeddings: {failed}")
    if embeddings_database["metadata"]["embedding_dimension"]:
        logger.info(f"Embedding dimension: {embeddings_database['metadata']['embedding_dimension']}")
    logger.info(f"Output saved to: {output_file}")


def main():
    """Main function to run embedding generation."""
    # Parse command line arguments
    args = parse_arguments()

    # Construct full file paths
    descriptions_file = args.input_dir / args.descriptions_file
    output_file = args.input_dir / args.output_file

    # Check API key
    if not args.api_key and not args.force:
        logger.warning("No API key provided via --api-key or NVIDIA_API_KEY env var")
        logger.warning("You can set it with: export NVIDIA_API_KEY='your-api-key'")
        logger.warning("Get your API key from: https://build.nvidia.com/")
        logger.warning("Use --force to continue without API key (for testing)")
        return

    if args.endpoint_url:
        logger.info(f"Using custom endpoint: {args.endpoint_url}")

    # Check if langchain-nvidia-ai-endpoints is available
    if not NVIDIA_EMBEDDINGS_AVAILABLE:
        logger.error("Required package not installed.")
        logger.error("Install with: pip install langchain-nvidia-ai-endpoints")
        return

    # Generate embeddings
    try:
        generate_embeddings_for_extensions(
            descriptions_file=descriptions_file,
            output_file=output_file,
            api_key=args.api_key,
            endpoint_url=args.endpoint_url,
            model=args.model,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
