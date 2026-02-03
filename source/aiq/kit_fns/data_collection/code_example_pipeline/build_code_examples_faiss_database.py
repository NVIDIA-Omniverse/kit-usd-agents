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
Build FAISS database from code examples embeddings.
Creates a searchable vector database for finding relevant Kit code examples.

Usage:
    python build_code_examples_faiss_database.py --mode regular  # For production code
    python build_code_examples_faiss_database.py --mode tests    # For test code
    python build_code_examples_faiss_database.py --mode all      # For all code
"""

import argparse
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

# Paths will be set based on mode
CODE_EXAMPLE_DATA_DIR = None
EMBEDDINGS_FILE = None
EXTRACTED_METHODS_DIR = None
FAISS_OUTPUT_DIR = None

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def configure_paths(mode: str = "regular"):
    """Configure paths based on the mode (regular, tests, or all).

    Args:
        mode: One of "regular", "tests", or "all"
    """
    global CODE_EXAMPLE_DATA_DIR, EMBEDDINGS_FILE, EXTRACTED_METHODS_DIR, FAISS_OUTPUT_DIR

    # Set paths based on mode
    if mode == "regular":
        CODE_EXAMPLE_DATA_DIR = Path("code_example_data_regular")
        EXTRACTED_METHODS_DIR = Path("extracted_methods_regular")
    elif mode == "tests":
        CODE_EXAMPLE_DATA_DIR = Path("code_example_data_tests")
        EXTRACTED_METHODS_DIR = Path("extracted_methods_tests")
    elif mode == "all":
        CODE_EXAMPLE_DATA_DIR = Path("code_example_data_all")
        EXTRACTED_METHODS_DIR = Path("extracted_methods_all")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'regular', 'tests', or 'all'")

    EMBEDDINGS_FILE = CODE_EXAMPLE_DATA_DIR / "code_examples_embeddings.json"
    FAISS_OUTPUT_DIR = CODE_EXAMPLE_DATA_DIR / "code_examples_faiss"

    logger.info(f"Configured for mode: {mode}")
    logger.info(f"  Embeddings file: {EMBEDDINGS_FILE}")
    logger.info(f"  Extracted methods directory: {EXTRACTED_METHODS_DIR}")
    logger.info(f"  FAISS output directory: {FAISS_OUTPUT_DIR}")


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


def load_method_source_code(extracted_methods_dir: Path, extension_name: str, method_name: str) -> str:
    """Load the full source code for a method from the extracted JSON files.

    Args:
        extracted_methods_dir: Path to the extracted methods directory
        extension_name: Name of the extension
        method_name: Full name of the method

    Returns:
        Source code string or empty string if not found
    """
    json_file = extracted_methods_dir / f"{extension_name}.example.json"

    if not json_file.exists():
        return ""

    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Find the method in the methods list
        for method in data.get("methods", []):
            if method.get("name") == method_name:
                return method.get("source_code", "")
    except Exception as e:
        logger.warning(f"Failed to load source for {method_name}: {e}")

    return ""


def create_method_page_content(method_key: str, method_data: Dict[str, Any], source_code: str = None) -> str:
    """Create page content for a method document.

    Args:
        method_key: The unique method identifier
        method_data: The method metadata from embeddings
        source_code: Optional source code to include

    Returns:
        A formatted string for the document's page_content
    """
    # Create a comprehensive text representation of the method
    content_parts = []

    # The method name is the most important for search
    content_parts.append(f"Method: {method_data.get('name', method_key)}")

    # Add location information
    if method_data.get("extension"):
        content_parts.append(f"Extension: {method_data['extension']}")

    if method_data.get("module"):
        content_parts.append(f"Module: {method_data['module']}")

    if method_data.get("file_path"):
        content_parts.append(f"File: {method_data['file_path']}")

    # Add metrics
    if method_data.get("line_count"):
        content_parts.append(f"Lines: {method_data['line_count']}")

    if method_data.get("complexity_score"):
        content_parts.append(f"Complexity: {method_data['complexity_score']}")

    # Add characteristics
    if method_data.get("is_async"):
        content_parts.append("Type: Async method")

    # Add reasons why it's interesting
    reasons = method_data.get("reasons", [])
    if reasons:
        content_parts.append(f"Features: {'; '.join(reasons)}")

    # Add description
    if method_data.get("description"):
        content_parts.append(f"Description: {method_data['description']}")

    # Add truncated source code preview if available
    if source_code:
        # Include first 200 characters of source code for context
        preview = source_code[:200] + "..." if len(source_code) > 200 else source_code
        content_parts.append(f"\nCode preview:\n{preview}")

    return "\n".join(content_parts)


def build_faiss_database(
    extracted_methods_dir: Path,
    embeddings_file: Path,
    faiss_output_path: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str = "",
    endpoint_url: str = None,
) -> None:
    """Build FAISS database from code examples embeddings.

    Args:
        extracted_methods_dir: Path to the extracted methods directory
        embeddings_file: Path to the embeddings JSON file
        faiss_output_path: Path to save the FAISS database
        embedding_model: Embedding model to use
        api_key: API key for authentication
        endpoint_url: Optional custom endpoint URL
    """
    logger.info("=" * 60)
    logger.info("Building FAISS Database for Code Examples")
    logger.info("=" * 60)

    # Create embedder (needed for FAISS initialization)
    try:
        embedder = create_embeddings(api_key, endpoint_url, embedding_model)
        logger.info("Successfully created embedder")
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        raise

    # Load embeddings file
    logger.info("Loading embeddings file...")

    if not embeddings_file.exists():
        logger.error(f"Embeddings file not found: {embeddings_file}")
        logger.error("Please run generate_code_examples_embeddings.py first")
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    with open(embeddings_file, "r") as f:
        embeddings_data = json.load(f)

    # Extract methods
    methods_embeddings = embeddings_data.get("methods", {})

    logger.info(f"Loaded {len(methods_embeddings)} method embeddings")

    # Prepare documents and embeddings for FAISS
    documents = []
    embedding_vectors = []
    skipped = 0

    # Track statistics
    extensions_set = set()
    complexity_total = 0
    async_count = 0

    for method_key, embedding_data in methods_embeddings.items():
        try:
            # Get the embedding vector
            embedding_vector = embedding_data.get("embedding", [])

            # Skip entries without valid embeddings
            if not embedding_vector or len(embedding_vector) == 0:
                logger.warning(f"Method {method_key} has no embedding vector, skipping")
                skipped += 1
                continue

            # Try to load source code from original extraction
            extension_name = embedding_data.get("extension", "")
            method_name = embedding_data.get("name", "")
            source_code = ""

            if extension_name and method_name:
                source_code = load_method_source_code(extracted_methods_dir, extension_name, method_name)

            # Create page content
            page_content = create_method_page_content(method_key, embedding_data, source_code)

            # Prepare metadata for the document
            metadata = {
                "method_key": method_key,
                "name": embedding_data.get("name", "unknown"),
                "extension": embedding_data.get("extension", ""),
                "module": embedding_data.get("module", ""),
                "file_path": embedding_data.get("file_path", ""),
                "line_count": embedding_data.get("line_count", 0),
                "complexity_score": embedding_data.get("complexity_score", 0),
                "is_async": embedding_data.get("is_async", False),
                "description": embedding_data.get("description", ""),
                "token_count": embedding_data.get("token_count", 0),
            }

            # Add reasons as searchable metadata
            reasons = embedding_data.get("reasons", [])
            if reasons:
                metadata["reasons"] = "; ".join(reasons)
                # Add individual reason flags for filtering
                for reason in reasons:
                    if "Large method" in reason:
                        metadata["is_large"] = True
                    if "High complexity" in reason:
                        metadata["is_complex"] = True
                    if "recursion" in reason:
                        metadata["has_recursion"] = True
                    if "error handling" in reason:
                        metadata["has_error_handling"] = True
                    if "Async" in reason:
                        metadata["is_async"] = True

            # Add source code to metadata if available and not too large
            if source_code and len(source_code) < 5000:  # Limit to 5KB
                metadata["source_code"] = source_code

            # Create document
            doc = Document(page_content=page_content, metadata=metadata)

            documents.append(doc)
            embedding_vectors.append(embedding_vector)

            # Update statistics
            extensions_set.add(embedding_data.get("extension", ""))
            complexity_total += embedding_data.get("complexity_score", 0)
            if embedding_data.get("is_async"):
                async_count += 1

        except Exception as e:
            logger.error(f"Failed to process method {method_key}: {e}")
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
        logger.info(f"Database contains {len(documents)} code example entries")

        # Calculate statistics
        avg_complexity = complexity_total / len(documents) if documents else 0
        async_percentage = (async_count / len(documents) * 100) if documents else 0

        # Save metadata about the database
        metadata_file = faiss_output_path / "metadata.json"
        metadata = {
            "total_methods": len(documents),
            "skipped_methods": skipped,
            "embedding_model": embedding_model,
            "embedding_dimension": len(embedding_vectors[0]) if embedding_vectors else 0,
            "created_at": datetime.now().isoformat(),
            "source_files": {"embeddings": str(embeddings_file)},
            "statistics": {
                "unique_extensions": len(extensions_set),
                "average_complexity": avg_complexity,
                "async_methods_count": async_count,
                "async_methods_percentage": async_percentage,
                "methods_by_feature": {},
            },
        }

        # Count methods by feature
        feature_counts = {
            "large_methods": 0,
            "complex_methods": 0,
            "recursive_methods": 0,
            "error_handling": 0,
            "async_methods": async_count,
        }

        for doc in documents:
            if doc.metadata.get("is_large"):
                feature_counts["large_methods"] += 1
            if doc.metadata.get("is_complex"):
                feature_counts["complex_methods"] += 1
            if doc.metadata.get("has_recursion"):
                feature_counts["recursive_methods"] += 1
            if doc.metadata.get("has_error_handling"):
                feature_counts["error_handling"] += 1

        metadata["statistics"]["methods_by_feature"] = feature_counts

        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_file}")

        # Log statistics
        logger.info("\nDatabase Statistics:")
        logger.info(f"  Unique extensions: {len(extensions_set)}")
        logger.info(f"  Average complexity: {avg_complexity:.1f}")
        logger.info(f"  Async methods: {async_count} ({async_percentage:.1f}%)")
        logger.info(f"\n  Methods by feature:")
        for feature, count in feature_counts.items():
            logger.info(f"    {feature}: {count}")

    except Exception as e:
        logger.error(f"Failed to create/save FAISS database: {e}")
        raise


def main():
    """Main function to run the FAISS database creation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Build FAISS database from code examples embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Build FAISS database for production code:
    python build_code_examples_faiss_database.py --mode regular
    
  Build FAISS database for test code:
    python build_code_examples_faiss_database.py --mode tests
    
  Build FAISS database for all code:
    python build_code_examples_faiss_database.py --mode all
    
  Custom embedding endpoint:
    python build_code_examples_faiss_database.py --mode regular --endpoint-url http://localhost:8000
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["regular", "tests", "all"],
        default="regular",
        help="Type of code to process: 'regular' (production), 'tests', or 'all' (default: regular)",
    )

    parser.add_argument(
        "--api-key", type=str, default=None, help="NVIDIA API key (can also be set via NVIDIA_API_KEY env var)"
    )

    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="Custom endpoint URL (can also be set via EMBEDDING_ENDPOINT_URL env var)",
    )

    args = parser.parse_args()

    # Configure paths based on mode
    configure_paths(args.mode)

    # Get API key from args or environment variable
    api_key = args.api_key or os.getenv("NVIDIA_API_KEY", "")

    if not api_key:
        logger.warning("No NVIDIA_API_KEY found")
        logger.warning("You can set it with: export NVIDIA_API_KEY='your-api-key'")
        logger.warning("Or pass it as: --api-key 'your-api-key'")
        logger.warning("Get your API key from: https://build.nvidia.com/")

        # The embedder is needed for FAISS initialization even if we have embeddings
        response = input("Continue without API key? (y/n): ")
        if response.lower() != "y":
            return

    # Get endpoint URL from args or environment variable
    endpoint_url = args.endpoint_url or os.getenv("EMBEDDING_ENDPOINT_URL", None)
    if endpoint_url:
        logger.info(f"Using custom endpoint: {endpoint_url}")

    # Build FAISS database
    try:
        logger.info(f"Starting FAISS database creation for mode: {args.mode}")
        build_faiss_database(
            extracted_methods_dir=EXTRACTED_METHODS_DIR,
            embeddings_file=EMBEDDINGS_FILE,
            faiss_output_path=FAISS_OUTPUT_DIR,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            api_key=api_key,
            endpoint_url=endpoint_url,
        )

        logger.info("\n" + "=" * 60)
        logger.info("FAISS database creation completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Database location: {FAISS_OUTPUT_DIR}")
        logger.info(f"Mode: {args.mode}")
        logger.info("\nYou can now use this database for similarity search:")
        logger.info(f"  - Load with: FAISS.load_local('{FAISS_OUTPUT_DIR}', embedder)")
        logger.info("  - Search with: vectorstore.similarity_search(query)")

        if args.mode == "regular":
            logger.info("\nExample search queries for production code:")
            logger.info("  - 'async method for handling events'")
            logger.info("  - 'recursive function implementation'")
            logger.info("  - 'error handling patterns'")
            logger.info("  - 'complex UI rendering logic'")
        elif args.mode == "tests":
            logger.info("\nExample search queries for test code:")
            logger.info("  - 'test async method'")
            logger.info("  - 'unit test for validation'")
            logger.info("  - 'test fixture setup'")
            logger.info("  - 'mock object creation'")

    except Exception as e:
        logger.error(f"FAISS database creation failed: {e}")
        raise


if __name__ == "__main__":
    main()
