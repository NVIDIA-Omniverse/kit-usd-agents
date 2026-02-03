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
Generate embedding vectors for code examples extracted from Kit extensions.
Creates embeddings using NVIDIA API and saves them to a separate embeddings file.
Each method's source code and metadata are combined and truncated to 500 tokens.

Usage:
    python generate_code_examples_embeddings.py --mode regular  # For production code
    python generate_code_examples_embeddings.py --mode tests    # For test code
    python generate_code_examples_embeddings.py --mode all      # For all code
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken

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
MAX_TOKENS = 500  # Maximum tokens for truncation
ENCODING_MODEL = "cl100k_base"

# Paths will be set based on mode
EXTRACTED_METHODS_DIR = None
OUTPUT_DIR = None
OUTPUT_FILE = None

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def configure_paths(mode: str = "regular"):
    """Configure paths based on the mode (regular, tests, or all).

    Args:
        mode: One of "regular", "tests", or "all"
    """
    global EXTRACTED_METHODS_DIR, OUTPUT_DIR, OUTPUT_FILE

    # Set extracted methods directory based on mode
    if mode == "regular":
        EXTRACTED_METHODS_DIR = Path("extracted_methods_regular")
        OUTPUT_DIR = Path("code_example_data_regular")
    elif mode == "tests":
        EXTRACTED_METHODS_DIR = Path("extracted_methods_tests")
        OUTPUT_DIR = Path("code_example_data_tests")
    elif mode == "all":
        EXTRACTED_METHODS_DIR = Path("extracted_methods_all")
        OUTPUT_DIR = Path("code_example_data_all")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'regular', 'tests', or 'all'")

    OUTPUT_FILE = OUTPUT_DIR / "code_examples_embeddings.json"

    logger.info(f"Configured for mode: {mode}")
    logger.info(f"  Input directory: {EXTRACTED_METHODS_DIR}")
    logger.info(f"  Output directory: {OUTPUT_DIR}")


def truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to specified number of tokens.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text
    """
    try:
        # Load the encoding
        enc = tiktoken.get_encoding(ENCODING_MODEL)

        # Encode to tokens
        tokens = enc.encode(text)

        # Truncate the token list if needed
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # Decode back to text
        truncated_text = enc.decode(tokens)
        return truncated_text
    except Exception as e:
        logger.error(f"Error truncating text: {e}")
        # Fallback to character-based truncation
        return text[: max_tokens * 4]  # Rough approximation


def create_embeddings(
    embeddings_model: str, api_key: str = "", endpoint_url: Optional[str] = None
) -> Optional["NVIDIAEmbeddings"]:
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
                model=embeddings_model,
                nvidia_api_key=api_key,
                truncate="END",
            )
        logger.info(f"Successfully created embedder with model: {DEFAULT_EMBEDDING_MODEL}")
        return embedder
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        return None


def create_method_document(method_data: Dict[str, Any], max_tokens: int) -> str:
    """Create a document string from a method for embedding.

    Combines method source code with metadata, truncated to MAX_TOKENS.

    Args:
        method_data: The method data dictionary from extracted JSON
        max_tokens: Maximum number of tokens for truncation
    Returns:
        A formatted string containing method code and metadata
    """
    doc_parts = []

    # 1. Start with the method source code (primary content)
    source_code = method_data.get("source_code", "")
    if source_code:
        doc_parts.append(source_code)

    # 2. Add metadata to augment the embedding
    doc_parts.append("\n\n--- Method Metadata ---")

    # Method name and location
    doc_parts.append(f"Method: {method_data.get('name', 'unknown')}")
    doc_parts.append(f"Module: {method_data.get('module', 'unknown')}")
    doc_parts.append(f"File: {method_data.get('file_path', 'unknown')}")

    # Complexity and size metrics
    doc_parts.append(f"Lines: {method_data.get('line_count', 0)}")
    doc_parts.append(f"Complexity: {method_data.get('complexity_score', 0)}")

    # Method characteristics
    if method_data.get("is_async"):
        doc_parts.append("Type: Async method")

    if method_data.get("decorators"):
        decorators_str = ", ".join(method_data["decorators"][:5])
        doc_parts.append(f"Decorators: {decorators_str}")

    # Class usages
    if method_data.get("class_usages"):
        classes_str = ", ".join(method_data["class_usages"][:5])
        doc_parts.append(f"Uses classes: {classes_str}")

    # Reasons why it's interesting
    if method_data.get("reasons"):
        reasons_str = "; ".join(method_data["reasons"][:3])
        doc_parts.append(f"Features: {reasons_str}")

    # Method description
    description = method_data.get("description", "")
    if description:
        doc_parts.append(f"Description: {description}")
    # Combine all parts
    document = "\n".join(doc_parts)

    # Truncate to MAX_TOKENS
    return truncate_to_tokens(document, max_tokens)


def load_extracted_methods(extracted_methods_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all extracted method JSON files from the extracted_methods directory.

    Args:
        extracted_methods_dir: Path to the extracted methods directory

    Returns:
        Dictionary mapping extension names to their methods
    """
    all_methods = {}

    if not extracted_methods_dir.exists():
        logger.warning(f"Extracted methods directory not found: {extracted_methods_dir}")
        return all_methods

    # Load all .json files from the directory
    json_files = list(extracted_methods_dir.glob("*.example.json"))
    logger.info(f"Found {len(json_files)} extension method files")

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            extension_name = data.get("extension_name", json_file.stem.replace(".example", ""))
            methods = data.get("methods", [])

            if methods:
                all_methods[extension_name] = methods
                logger.info(f"  Loaded {len(methods)} methods from {extension_name}")
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")

    return all_methods


def generate_embeddings_for_code_examples(
    extracted_methods_dir: Path,
    output_dir: Path,
    encoding_model: str,
    embeddings_model: str,
    api_key: str,
    endpoint_url: str,
    max_tokens: int,
    batch_size: int,
) -> None:
    """Generate embeddings for all code examples.

    Args:
        extracted_methods_dir: Path to the extracted methods directory
        output_dir: Path to save the embeddings database
        api_key: NVIDIA API key
        endpoint_url: Optional custom endpoint URL
        max_tokens: Maximum number of tokens for truncation
        batch_size: Number of methods to process in batch
        encoding_model: Encoding model to use
        model: Embedding model to use
    """
    # Load all extracted methods
    all_methods = load_extracted_methods(extracted_methods_dir)

    if not all_methods:
        logger.error("No extracted methods found to process")
        return

    # Count total methods
    total_methods = sum(len(methods) for methods in all_methods.values())
    logger.info(f"Total methods to process: {total_methods}")

    # Create embedder
    embedder = create_embeddings(embeddings_model, api_key, endpoint_url)
    if not embedder:
        logger.error("Failed to create embedder. Exiting.")
        return

    # Initialize tiktoken encoder for token counting
    try:
        enc = tiktoken.get_encoding(encoding_model)
    except Exception as e:
        logger.error(f"Failed to initialize tiktoken: {e}")
        enc = None

    # Process methods and generate embeddings
    embeddings_database = {
        "metadata": {
            "model": embeddings_model,
            "total_methods": total_methods,
            "embedding_dimension": None,
            "generated_at": None,
            "max_tokens": max_tokens,
        },
        "methods": {},
    }

    processed = 0
    failed = 0

    # Process each extension's methods
    for extension_name, methods in all_methods.items():
        logger.info(f"\nProcessing extension: {extension_name}")

        # Process methods in batches
        for i in range(0, len(methods), batch_size):
            batch = methods[i : i + batch_size]
            batch_texts = []
            batch_keys = []

            # Prepare batch
            for method_data in batch:
                # Create document text for embedding
                document = create_method_document(method_data, max_tokens)

                # Create unique key for the method
                method_key = f"{extension_name}::{method_data.get('name', 'unknown')}"

                batch_texts.append(document)
                batch_keys.append(method_key)

            if not batch_texts:
                continue

            try:
                # Generate embeddings for batch
                batch_num = i // batch_size + 1
                total_batches = (len(methods) + batch_size - 1) // batch_size
                logger.info(f"  Batch {batch_num}/{total_batches}: {len(batch_texts)} methods")

                if len(batch_texts) == 1:
                    # Single item - use embed_query
                    embeddings = [embedder.embed_query(batch_texts[0])]
                else:
                    # Multiple items - use embed_documents
                    embeddings = embedder.embed_documents(batch_texts)

                # Store embeddings with metadata
                for method_key, embedding, method_data in zip(batch_keys, embeddings, batch):
                    # Count actual tokens if encoder available
                    token_count = 0
                    if enc:
                        doc = create_method_document(method_data, max_tokens)
                        token_count = len(enc.encode(doc))

                    embeddings_database["methods"][method_key] = {
                        "embedding": embedding,
                        "extension": extension_name,
                        "name": method_data.get("name", "unknown"),
                        "module": method_data.get("module", ""),
                        "file_path": method_data.get("file_path", ""),
                        "line_count": method_data.get("line_count", 0),
                        "complexity_score": method_data.get("complexity_score", 0),
                        "is_async": method_data.get("is_async", False),
                        "reasons": method_data.get("reasons", []),
                        "description": method_data.get("description", ""),
                        "token_count": token_count,
                    }
                    processed += 1

                    # Set embedding dimension from first successful embedding
                    if embeddings_database["metadata"]["embedding_dimension"] is None:
                        embeddings_database["metadata"]["embedding_dimension"] = len(embedding)
                        logger.info(f"  Embedding dimension: {len(embedding)}")

                # Log progress
                logger.info(f"  Progress: {processed}/{total_methods} methods")

            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                # Add empty embeddings for failed batch
                for method_key, method_data in zip(batch_keys, batch):
                    embeddings_database["methods"][method_key] = {
                        "embedding": [],
                        "extension": extension_name,
                        "name": method_data.get("name", "unknown"),
                        "error": str(e),
                    }
                    failed += 1

    # Add timestamp and statistics
    embeddings_database["metadata"]["generated_at"] = datetime.now().isoformat()
    embeddings_database["metadata"]["successful_embeddings"] = processed
    embeddings_database["metadata"]["failed_embeddings"] = failed

    output_file = output_dir / "code_examples_embeddings.json"
    # Save embeddings database
    logger.info(f"\nSaving embeddings database to: {output_file}")
    output_dir.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(embeddings_database, f, indent=2)

    # Summary
    logger.info("=" * 60)
    logger.info("Code Examples Embedding Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Total methods: {total_methods}")
    logger.info(f"Successful embeddings: {processed}")
    logger.info(f"Failed embeddings: {failed}")
    if embeddings_database["metadata"]["embedding_dimension"]:
        logger.info(f"Embedding dimension: {embeddings_database['metadata']['embedding_dimension']}")
    logger.info(f"Output saved to: {OUTPUT_FILE}")


def main():
    """Main function to run embedding generation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate embeddings for code examples from Kit extensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate embeddings for production code:
    python generate_code_examples_embeddings.py --mode regular
    
  Generate embeddings for test code:
    python generate_code_examples_embeddings.py --mode tests
    
  Generate embeddings for all code:
    python generate_code_examples_embeddings.py --mode all
    
  Custom batch size:
    python generate_code_examples_embeddings.py --mode regular --batch-size 20
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
        "--batch-size", type=int, default=10, help="Number of methods to process in each batch (default: 10)"
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

        # Try to continue anyway - some endpoints might not require API key
        response = input("Continue without API key? (y/n): ")
        if response.lower() != "y":
            return

    # Get endpoint URL from args or environment variable
    endpoint_url = args.endpoint_url or os.getenv("EMBEDDING_ENDPOINT_URL", None)
    if endpoint_url:
        logger.info(f"Using custom endpoint: {endpoint_url}")

    # Check if langchain-nvidia-ai-endpoints is available
    if not NVIDIA_EMBEDDINGS_AVAILABLE:
        logger.error("Required package not installed.")
        logger.error("Install with: pip install langchain-nvidia-ai-endpoints")
        return

    # Generate embeddings
    try:
        logger.info(f"Starting embedding generation for mode: {args.mode}")
        generate_embeddings_for_code_examples(
            EXTRACTED_METHODS_DIR,
            OUTPUT_DIR,
            ENCODING_MODEL,
            DEFAULT_EMBEDDING_MODEL,
            api_key,
            endpoint_url,
            MAX_TOKENS,
            args.batch_size,
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
