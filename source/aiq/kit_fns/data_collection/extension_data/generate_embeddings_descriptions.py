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
Generate embedding descriptions for extensions.
Creates token-limited descriptions combining metadata, overview, and database entry.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import tiktoken


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embedding descriptions for extensions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum number of tokens per description")

    parser.add_argument("--encoding-model", type=str, default="cl100k_base", help="Tiktoken encoding model to use")

    parser.add_argument(
        "--extensions-dir",
        type=Path,
        default=Path("/home/horde/repos/kit-app-template/_build/linux-x86_64/release/extscache"),
        help="Path to the extensions directory",
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("extension_detail"), help="Output directory for generated files"
    )

    return parser.parse_args()


def truncate_to_tokens(text: str, max_tokens: int, encoding_model: str) -> str:
    """Truncate text to specified number of tokens."""
    try:
        # Load the encoding
        enc = tiktoken.get_encoding(encoding_model)

        # Encode to tokens
        tokens = enc.encode(text)

        # Truncate the token list if needed
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # Decode back to text
        truncated_text = enc.decode(tokens)
        return truncated_text
    except Exception as e:
        print(f"Error truncating text: {e}")
        # Fallback to character-based truncation
        return text[: max_tokens * 4]  # Rough approximation


def load_overview_content(extension_path: Path) -> str:
    """Load Overview.md content if it exists."""
    overview_paths = [extension_path / "docs" / "Overview.md", extension_path / "Overview.md"]

    for overview in overview_paths:
        if overview.exists():
            try:
                with open(overview, "r", encoding="utf-8") as f:
                    return re.sub(r"^```.*?```\s*", "", f.read(), flags=re.DOTALL)
            except Exception as e:
                print(f"  Error reading {overview}: {e}")

    return ""


def generate_extension_description(
    ext_id: str, ext_data: Dict[str, Any], extscache_path: Path, max_tokens: int, encoding_model: str
) -> str:
    """Generate a comprehensive description for embedding."""
    parts = []

    # 1. Extension name and title
    parts.append(f"Extension: {ext_id}")

    # 2. Basic metadata
    title = ext_data.get("title") or ext_data.get("description", "")
    if title:
        parts.append(f"Title: {title}")

    if ext_data.get("long_description"):
        parts.append(f"Details: {ext_data['long_description']}")

    # 3. Keywords and topics
    if ext_data.get("keywords"):
        parts.append(f"Keywords: {', '.join(ext_data['keywords'])}")

    if ext_data.get("topics"):
        parts.append(f"Topics: {', '.join(ext_data['topics'])}")

    # 4. Category
    if ext_data.get("category"):
        parts.append(f"Category: {ext_data['category']}")

    # 5. Dependencies
    if ext_data.get("dependencies"):
        parts.append(f"Dependencies: {', '.join(ext_data['dependencies'])}")

    # 6. API information
    api_info = []
    if ext_data.get("has_python_api"):
        api_info.append("Has Python API")
    if ext_data.get("has_overview"):
        api_info.append("Has documentation overview")
    if api_info:
        parts.append(f"Features: {', '.join(api_info)}")

    # 7. Code statistics
    if ext_data.get("total_modules", 0) > 0:
        parts.append(
            f"Code Statistics: {ext_data['total_modules']} modules, {ext_data['total_classes']} classes, {ext_data['total_methods']} methods"
        )

    # Join basic metadata
    description = "\n\n".join(parts)

    # 8. Load and append Overview.md content if available
    if ext_data.get("has_overview"):
        storage_path = ext_data.get("storage_path", "").rstrip("/")
        if storage_path:
            extension_path = extscache_path / storage_path.replace("extscache/", "")
            overview_content = load_overview_content(extension_path)
            if overview_content:
                description += "\n\n--- Overview ---\n" + overview_content

    # 9. Append JSON database entry
    # Create a cleaned version without large fields
    clean_entry = {
        k: v
        for k, v in ext_data.items()
        if k not in ["full_api_details_path", "config_file", "storage_path", "embedding_vector_index"]
    }
    description += "\n\n--- Database Entry ---\n" + json.dumps(clean_entry, indent=2)

    # Truncate to max tokens
    return truncate_to_tokens(description, max_tokens, encoding_model)


def generate_extensions_descriptions(
    extensions_dir: Path, database_file: Path, output_dir: Path, max_tokens: int, encoding_model: str
) -> dict:
    """Generate embedding descriptions for all extensions."""

    # Load the extensions database
    if not database_file.exists():
        print(f"Error: Database file not found: {database_file}")
        return

    print(f"Loading database from: {database_file}")
    with open(database_file, "r") as f:
        database = json.load(f)

    # Get the extensions from the database
    if "extensions" in database:
        extensions = database["extensions"]
    else:
        # Fallback for old format
        extensions = {
            k: v
            for k, v in database.items()
            if isinstance(v, dict) and k not in ["database_version", "generated_at", "total_extensions"]
        }

    print(f"Found {len(extensions)} extensions in database\n")

    # Generate descriptions for each extension
    descriptions = {}
    enc = tiktoken.get_encoding(encoding_model)

    for ext_id, ext_data in extensions.items():

        print(f"Processing: {ext_id}")

        try:
            # Generate the description
            description = generate_extension_description(ext_id, ext_data, extensions_dir, max_tokens, encoding_model)

            # Count actual tokens
            token_count = len(enc.encode(description))

            # Store the description
            descriptions[ext_id] = {
                "description": description,
                "token_count": token_count,
                "version": ext_data.get("version", ""),
                "has_overview": ext_data.get("has_overview", False),
                "has_python_api": ext_data.get("has_python_api", False),
            }

            print(f"  ✓ Generated description: {token_count} tokens")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            descriptions[ext_id] = {
                "description": f"Extension: {ext_id}",
                "token_count": 2,
                "version": ext_data.get("version", ""),
                "error": str(e),
            }

    # Save the descriptions
    descriptions_file = output_dir / "extensions_descriptions.json"
    print(f"\nSaving descriptions to: {descriptions_file}")
    output_dir.mkdir(exist_ok=True)

    with open(descriptions_file, "w") as f:
        json.dump(descriptions, f, indent=2)

    # Summary statistics
    total_tokens = sum(d["token_count"] for d in descriptions.values())
    avg_tokens = total_tokens / len(descriptions) if descriptions else 0

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total extensions processed: {len(descriptions)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per description: {avg_tokens:.1f}")
    print(f"Output saved to: {descriptions_file}")


def main():
    """Generate embedding descriptions for all extensions."""
    # Parse command line arguments
    args = parse_arguments()

    # Compute derived paths
    database_file = args.output_dir / "extensions_database.json"

    print("=" * 60)
    print("Generating Extension Embedding Descriptions")
    print("=" * 60)
    print(f"Max tokens per description: {args.max_tokens}")
    print(f"Encoding: {args.encoding_model}")
    print(f"Extensions directory: {args.extensions_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    generate_extensions_descriptions(
        args.extensions_dir, database_file, args.output_dir, args.max_tokens, args.encoding_model
    )


if __name__ == "__main__":
    main()
