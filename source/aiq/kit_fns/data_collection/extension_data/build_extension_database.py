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
Extension Database Builder
Processes all extensions in the Kit extscache folder to:
1. Extract metadata from extension.toml
2. Check for documentation files
3. Generate Code Atlas JSON for Python APIs using lc_agent
4. Create extension detail files
5. Build extensions database index
"""

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import toml

# Token counting with tiktoken
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("Warning: tiktoken not available. Install with: pip install tiktoken")
    TIKTOKEN_AVAILABLE = False

# Import Code Atlas from lc_agent - using public API
from lc_agent.code_atlas import CodeAtlasCache

print("Using Code Atlas from lc_agent")

# ============================================================================
# Configuration
# ============================================================================

# Extensions to exclude from processing (folder name patterns)
EXCLUDED_EXTENSIONS: Set[str] = {}


def should_exclude_extension(extension_name: str) -> bool:
    """Check if an extension should be excluded from processing."""
    # Check exact matches
    if extension_name in EXCLUDED_EXTENSIONS:
        return True

    # Check partial matches (if folder name contains excluded pattern)
    for excluded in EXCLUDED_EXTENSIONS:
        if excluded in extension_name:
            return True

    return False


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    if not TIKTOKEN_AVAILABLE:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

    try:
        encoding = tiktoken.get_encoding(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return len(text) // 4  # Fallback estimation


# ============================================================================
# Extension Processor using lc_agent Code Atlas
# ============================================================================


class ExtensionProcessor:
    """Process a single extension to extract metadata and generate Code Atlas."""

    def __init__(
        self,
        extension_path: Path,
        embedding_index: int = 0,
        include_source_code: bool = False,
        excluded_modules: List[str] = None,
    ):
        self.extension_path = extension_path
        self.extension_name = None
        self.extension_version = None
        self.embedding_index = embedding_index
        self.include_source_code = include_source_code
        self.excluded_modules = excluded_modules
        self.metadata = {}
        self.code_atlas = {"modules": {}, "classes": {}, "methods": {}, "used_classes": {}}

    def process(self) -> tuple[dict, dict]:
        """Process the extension and return metadata and code atlas."""
        print(f"\n{'='*60}")
        print(f"Processing: {self.extension_path.name}")
        print(f"{'='*60}")

        # Extract extension name and version from folder name
        self._parse_folder_name()

        # Extract metadata from extension.toml
        self._extract_metadata()

        # Check for documentation files
        self._check_documentation()

        # Generate Code Atlas for Python modules
        self._scan_python_modules()

        return self.metadata, self.code_atlas

    def _parse_folder_name(self):
        """Parse extension name and version from folder name."""
        folder_name = self.extension_path.name
        # Split on last hyphen followed by version pattern (digit at start)
        # Using non-backtracking approach: find last hyphen followed by digit
        last_hyphen_idx = -1
        for i in range(len(folder_name) - 1, -1, -1):
            if folder_name[i] == "-" and i + 1 < len(folder_name) and folder_name[i + 1].isdigit():
                last_hyphen_idx = i
                break
        if last_hyphen_idx > 0:
            self.extension_name = folder_name[:last_hyphen_idx]
            self.extension_version = folder_name[last_hyphen_idx + 1 :]
        else:
            self.extension_name = folder_name
            self.extension_version = "unknown"

        print(f"  Extension: {self.extension_name}")
        print(f"  Version: {self.extension_version}")

    def _extract_metadata(self):
        """Extract metadata from extension.toml file."""
        toml_path = self.extension_path / "config" / "extension.toml"
        if not toml_path.exists():
            toml_path = self.extension_path / "extension.toml"

        if toml_path.exists():
            print(f"  ✓ Found extension.toml")
            try:
                with open(toml_path, "r", encoding="utf-8") as f:
                    data = toml.load(f)

                # Extract package info
                package = data.get("package", {})
                self.metadata["description"] = package.get("description", "")
                self.metadata["keywords"] = package.get("keywords", [])
                self.metadata["version"] = package.get("version", self.extension_version)

                # Extract Python modules
                python_modules = data.get("python", {}).get("modules", [])
                self.metadata["python_modules"] = python_modules

                # Extract dependencies and split optional ones (e.g., "ext.id" = { optional = true })
                dependencies = data.get("dependencies", {})
                required_deps = []
                optional_deps = []
                if isinstance(dependencies, dict):
                    for dep_name, dep_cfg in dependencies.items():
                        is_optional = False
                        if isinstance(dep_cfg, dict):
                            is_optional = bool(dep_cfg.get("optional", False))
                        if is_optional:
                            optional_deps.append(dep_name)
                        else:
                            required_deps.append(dep_name)
                else:
                    required_deps = list(dependencies or [])

                self.metadata["dependencies"] = required_deps
                self.metadata["optional_dependencies"] = optional_deps

            except Exception as e:
                print(f"  ⚠ Error reading extension.toml: {e}")
        else:
            print(f"  ⚠ No extension.toml found")

        # Add computed metadata
        self.metadata["extension_id"] = self.extension_name
        self.metadata["storage_path"] = self.extension_path.name
        self.metadata["api_extraction_timestamp"] = datetime.now().isoformat()

    def _check_documentation(self):
        """Check for documentation files and count Overview.md tokens."""
        # Check for python_api.md in config folder
        api_doc = self.extension_path / "config" / "python_api.md"
        self.metadata["has_python_api"] = api_doc.exists()
        if self.metadata["has_python_api"]:
            print(f"  ✓ Found python_api.md")

        # Check for Overview.md and count tokens
        overview = self.extension_path / "docs" / "Overview.md"
        if not overview.exists():
            overview = self.extension_path / "Overview.md"
        self.metadata["has_overview"] = overview.exists()
        self.metadata["overview_token_count"] = 0

        if self.metadata["has_overview"]:
            print(f"  ✓ Found Overview.md")
            try:
                with open(overview, "r", encoding="utf-8") as f:
                    overview_content = f.read()
                    self.metadata["overview_token_count"] = count_tokens(overview_content)
                    print(f"    Overview.md: {self.metadata['overview_token_count']:,} tokens")
            except Exception as e:
                print(f"    Error reading Overview.md: {e}")

        # Check for README.md
        readme = self.extension_path / "docs" / "README.md"
        if not readme.exists():
            readme = self.extension_path / "README.md"
        if readme.exists():
            print(f"  ✓ Found README.md")

    def _scan_python_modules(self):
        """Scan Python modules and generate Code Atlas using lc_agent."""
        print(f"  Scanning Python modules...")

        try:
            # Use CodeAtlasCache to scan the entire extension directory
            cache = CodeAtlasCache()
            omni_path = Path(self.extension_path) / "omni"
            if not omni_path.exists():
                return
            cache.scan(str(self.extension_path), excluded_modules=self.excluded_modules)

            # Extract and serialize the scanned data
            for module_full_name, module_info in cache._modules.items():
                self.code_atlas["modules"][module_full_name] = module_info.model_dump(
                    by_alias=True, exclude_defaults=True
                )

            for class_full_name, class_info in cache._classes.items():
                self.code_atlas["classes"][class_full_name] = class_info.model_dump(
                    by_alias=True, exclude_defaults=True
                )

            for method_full_name, method_info in cache._methods.items():
                # If we don't want to include source code, remove it from the serialized data
                method_dict = method_info.model_dump(by_alias=True, exclude_defaults=True)
                if not self.include_source_code and "source_code" in method_dict:
                    method_dict["source_code"] = None
                self.code_atlas["methods"][method_full_name] = method_dict

            # Store used_classes information
            self.code_atlas["used_classes"] = dict(cache._used_classes)

            # Update metadata with counts
            self.metadata["total_modules"] = len(cache._modules)
            self.metadata["total_classes"] = len(cache._classes)
            self.metadata["total_methods"] = len(cache._methods)

            print(
                f"    Found: {len(cache._modules)} modules, {len(cache._classes)} classes, {len(cache._methods)} methods"
            )

        except Exception as e:
            print(f"    ✗ Error scanning extension: {e}")
            # Set counts to 0 if scanning failed
            self.metadata["total_modules"] = 0
            self.metadata["total_classes"] = 0
            self.metadata["total_methods"] = 0

        # Extract topics from description
        self._extract_topics()

        # Generate long description
        self._generate_long_description()

    def _extract_topics(self):
        """Extract topics from description and keywords."""
        topics = []

        # Add keywords as topics
        if "keywords" in self.metadata:
            topics.extend(self.metadata["keywords"][:7])

        # Extract from description if needed
        description = self.metadata.get("description", "").lower()
        common_topics = ["ui", "rendering", "physics", "animation", "graph", "viewport", "window"]
        for topic in common_topics:
            if topic in description and topic not in topics:
                topics.append(topic)

        self.metadata["topics"] = topics[:7]  # Limit to 7 topics

    def _generate_long_description(self):
        """Generate a long description if not available."""
        # Prefer Overview.md content if available
        try:
            overview = self.extension_path / "docs" / "Overview.md"

            if overview.exists():
                with open(overview, "r", encoding="utf-8") as f:
                    overview_content = f.read()
                # Store full overview content without compression here
                self.metadata["long_description"] = re.sub(r"^```.*?```\s*", "", overview_content, flags=re.DOTALL)
                return
        except Exception as e:
            print(f"    Error reading Overview.md for long_description: {e}")

        # Fallback to description
        description_text = self.metadata.get("description", "").strip()
        if description_text:
            # Store description as-is without compression here
            self.metadata["long_description"] = description_text
        else:
            self.metadata["long_description"] = f"Extension {self.extension_name} for Omniverse Kit"


# ============================================================================
# API Documentation Generator
# ============================================================================


def _parse_python_api_md(md_path: str, code_atlas: dict) -> Optional[dict]:
    """Parse config/python_api.md for a given extension and return API listing.

    The markdown format is structured as bullet lists of classes and methods. We perform
    a lightweight parse to extract class names and method/property signatures.

    Args:
        md_path: The path to the python_api.md file

    Returns:
        API documentation dictionary
    """

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception:
        return None

    # Try to parse the python_api.md file
    import re

    segments = re.split(r"^# Public API for module ([\w\.]+):\n", source, flags=re.MULTILINE)
    module_sources = {segments[idx]: segments[idx + 1].splitlines() for idx in range(1, len(segments), 2)}
    api_docs = {"classes": {}, "methods": {}}

    for module_name, lines in module_sources.items():
        classes: Dict[str, Dict[str, Any]] = {}
        functions: Dict[str, Dict[str, Any]] = {}
        current_class_stack: List[str] = []

        for line in lines:
            # Use regex to match the markdown list structure: indentation, decorators, type, name, params, return type
            # Use negated character classes and bounded quantifiers to avoid catastrophic backtracking (ReDoS)
            # Allow nested parens with a reasonable depth limit for function signatures
            match = re.match(
                r"(\s*)- (\[[^\]]+\] )*((?:static|class) )?(def|class)\s+(\w+)(?:(\([^()]*(?:\([^()]*\)[^()]*)*\)?)(?:\s*->\s*([^\n]+)|$)|$)",
                line,
            )
            if match is None:
                continue

            indent, decorators, static_modifier, type_keyword, name, signature, return_type = match.groups()

            # Reset current_class when we've moved outside a class scope (based on indentation)
            if indent != " " * len(current_class_stack) * 2:  # Top-level items reset the class context
                current_class_stack.pop()

            # Detect class definitions
            if type_keyword == "class":
                current_class_stack.append(name)
                classes.setdefault(
                    ".".join(current_class_stack),
                    {
                        "name": ".".join(current_class_stack),
                        "methods": [],
                        "parent_classes": signature[1:-1].split(", ") if signature else [],
                    },
                )

            # Detect method definitions within a class
            if type_keyword == "def":
                method_info = {
                    "name": name,
                    "signature": signature,
                    "return_type": return_type.strip() if return_type else None,
                }

                if current_class_stack:
                    classes[".".join(current_class_stack)]["methods"].append(method_info)
                else:
                    functions.setdefault(name, method_info)

        # Convert to the expected format (matching generate_api_docs output)
        # Process classes
        for class_name, class_info in classes.items():
            class_full_name = f"{module_name}.{class_name}"
            api_docs["classes"][class_full_name] = {
                "parent_classes": class_info.get("parent_classes", []),
                "docstring": code_atlas.get("classes", {}).get(class_full_name, {}).get("docstring", ""),
                "methods": [method["name"] for method in class_info.get("methods", [])],
            }

            # Add methods to the methods dict
            for method in class_info.get("methods", []):
                method_name = method["name"]
                full_method_name = f"{module_name}.{class_name}.{method_name}"
                api_docs["methods"][full_method_name] = {
                    "name": method_name,
                    "signature": method.get("signature", ""),
                    "docstring": code_atlas.get("methods", {}).get(full_method_name, {}).get("docstring", ""),
                    "parent_class": class_name,
                    "return_type": method.get("return_type", ""),
                }

        # Process standalone functions
        for function_name, function_info in functions.items():
            full_method_name = f"{module_name}.{function_name}"
            api_docs["methods"][full_method_name] = {
                "name": function_name,
                "signature": function_info.get("signature", ""),
                "docstring": code_atlas.get("methods", {}).get(full_method_name, {}).get("docstring", ""),
                "parent_class": None,
                "return_type": function_info.get("return_type", ""),
            }

    return api_docs


def generate_api_docs(code_atlas: dict, extension_path: Path) -> dict:
    """Generate simplified API documentation from python_api.md or Code Atlas.

    First tries to parse python_api.md file for the extension. If that succeeds,
    returns the parsed API documentation. Otherwise, falls back to parsing the
    Code Atlas dictionary.

    Extracts only public methods and classes with their docstrings and parameters.
    Filters out private/internal methods (starting with _).

    Args:
        code_atlas: The Code Atlas dictionary with modules, classes, methods
        extension_path: The path to the extension

    Returns:
        Simplified API documentation dictionary
    """
    if extension_path is not None:
        md_path = extension_path / "config" / "python_api.md"
        if md_path.exists():
            api_docs = _parse_python_api_md(md_path, code_atlas)
            if api_docs is not None:
                return api_docs

    # Fall back to Code Atlas parsing
    api_docs = {"classes": {}, "methods": {}}

    python_modules = set()
    if extension_path is not None:
        config_path = extension_path / "config" / "extension.toml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = toml.load(f)
            python_modules = set([module.get("name") for module in config.get("python", {}).get("module", [])])

    python_modules.add(os.path.basename(extension_path).split("-")[0])

    # Process classes - only include public classes
    for class_full_name, class_info in code_atlas.get("classes", {}).items():
        # Only include classes in python_modules
        if class_info.get("module_name") not in python_modules:
            continue

        # extract just methods names get

        api_docs["classes"][class_full_name] = {
            "parent_classes": class_info.get("parent_classes", []),
            "docstring": class_info.get("docstring"),
            "methods": class_info.get("methods", []),
        }

    # Process methods - only include public methods
    for method_full_name, method_info in code_atlas.get("methods", {}).items():
        # Only include methods in python_modules
        if method_info.get("module_name") not in python_modules:
            continue
        # Skip private/internal methods (but keep __init__)
        method_short_name = method_info.get("name", "")
        if method_short_name.startswith("_") and method_short_name not in [
            "__init__",
            "__call__",
            "__str__",
            "__repr__",
        ]:
            continue

        # Build parameter signature
        parameters = []
        for arg in method_info.get("arguments", []):
            if arg.get("name") == "self":
                continue  # Skip self parameter

            # Build parameter string
            param_str = arg.get("name", "")
            if arg.get("type_annotation"):
                param_str = f"{param_str}: {arg['type_annotation']}"
            if arg.get("default_value"):
                param_str = f"{param_str} = {arg['default_value']}"

            parameters.append(param_str)

        # Create the method signature
        signature = f"({', '.join(parameters)})"

        api_docs["methods"][method_full_name] = {
            "name": method_info.get("name"),
            "signature": signature,
            "docstring": method_info.get("docstring"),
            "parent_class": method_info.get("parent_class"),
            "return_type": method_info.get("return_type"),
        }

    return api_docs


# ============================================================================
# Extensions Database Builder
# ============================================================================


class ExtensionsDatabaseBuilder:
    """Build the main extensions database index."""

    def __init__(self):
        self.database = {}
        self.embedding_counter = 0

    def add_extension(
        self,
        metadata: dict,
        codeatlas_filename: str = None,
        codeatlas_token_count: int = 0,
        api_docs_token_count: int = 0,
    ):
        """Add an extension to the database index."""
        ext_id = metadata["extension_id"]

        # Build the index entry matching the required format
        entry = {
            "version": metadata["version"],
            "storage_path": f"extscache/{metadata['storage_path']}/",
            "config_file": f"extscache/{metadata['storage_path']}/config/extension.toml",
            # Descriptions
            "title": metadata.get("title", metadata["extension_id"]),  # One-liner
            "description": metadata.get("description", ""),  # One-liner
            "long_description": metadata.get("long_description", ""),  # 3-5 lines
            "category": metadata.get("category", ""),
            "keywords": metadata.get("keywords", []),
            # API information
            "has_python_api": metadata.get("has_python_api", False),
            "has_overview": metadata.get("has_overview", False),
            "full_api_details_path": f"extension_detail/codeatlas/{codeatlas_filename}" if codeatlas_filename else None,
            "embedding_vector_index": self.embedding_counter,
            # Token counts for all documentation
            "codeatlas_token_count": codeatlas_token_count,
            "api_docs_token_count": api_docs_token_count,
            "overview_token_count": metadata.get("overview_token_count", 0),
            # Additional metadata
            "total_modules": metadata.get("total_modules", 0),
            "total_classes": metadata.get("total_classes", 0),
            "total_methods": metadata.get("total_methods", 0),
            "dependencies": metadata.get("dependencies", []),
            "optional_dependencies": metadata.get("optional_dependencies", []),
        }

        self.database[ext_id] = entry
        self.embedding_counter += 1

    def save(self, output_path: Path):
        """Save the extensions database to JSON."""
        db_file = output_path / "extensions_database.json"

        # Add metadata
        output = {
            "database_version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "total_extensions": len(self.database),
            "extensions": self.database,
        }

        with open(db_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"  → Saved extensions database: {db_file}")
        return db_file


# ============================================================================
# Main Processing Script
# ============================================================================


def process_all_extensions(
    extscache_path: str,
    output_dir: str = "extension_detail",
    include_source_code: bool = False,
    excluded_modules: List[str] = None,
):
    """Process all extensions in the extscache directory.

    Args:
        extscache_path: Path to the extscache directory
        output_dir: Output directory for generated files
        include_source_code: Whether to include function source code in Code Atlas (default: False)
        excluded_modules: List of module names to exclude when scanning Python code
    """
    extscache = Path(extscache_path)
    if not extscache.exists():
        print(f"Error: Directory not found: {extscache_path}")
        return

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create subdirectories for different file types
    codeatlas_dir = output_path / "codeatlas"
    codeatlas_dir.mkdir(exist_ok=True)

    api_docs_dir = output_path / "api_docs"
    api_docs_dir.mkdir(exist_ok=True)

    # Get all extension directories and filter out excluded ones
    all_extensions = [d for d in extscache.iterdir() if d.is_dir()]
    extensions = []
    excluded_count = 0

    for ext_dir in all_extensions:
        if should_exclude_extension(ext_dir.name):
            print(f"  ⊗ Excluding: {ext_dir.name}")
            excluded_count += 1
        else:
            extensions.append(ext_dir)

    print(f"Found {len(all_extensions)} total extensions")
    print(f"  - Processing: {len(extensions)}")
    print(f"  - Excluded: {excluded_count}")

    # Initialize database builder
    db_builder = ExtensionsDatabaseBuilder()

    # Summary data
    summary = {"total_extensions": len(extensions), "processed": 0, "failed": 0, "extensions": []}

    # Process each extension
    for idx, ext_dir in enumerate(extensions):
        try:
            processor = ExtensionProcessor(
                ext_dir, embedding_index=idx, include_source_code=include_source_code, excluded_modules=excluded_modules
            )
            metadata, code_atlas = processor.process()

            # Check if there's actual API content
            has_api_content = (
                metadata.get("total_modules", 0) > 0
                or metadata.get("total_classes", 0) > 0
                or metadata.get("total_methods", 0) > 0
            )

            codeatlas_filename = None
            codeatlas_token_count = 0
            api_docs_token_count = 0

            if has_api_content:
                # Save Code Atlas JSON
                codeatlas_filename = f"{metadata['extension_id']}-{metadata['version']}.codeatlas.json"
                codeatlas_file = codeatlas_dir / codeatlas_filename

                # Convert to JSON string for token counting
                codeatlas_json_str = json.dumps(code_atlas, indent=2)
                codeatlas_token_count = count_tokens(codeatlas_json_str)

                # Save the JSON file
                with open(codeatlas_file, "w", encoding="utf-8") as f:
                    f.write(codeatlas_json_str)
                print(f"  → Saved: codeatlas/{codeatlas_filename} ({codeatlas_token_count:,} tokens)")

                # Generate and save API documentation
                api_docs = generate_api_docs(code_atlas, ext_dir)

                # Count public APIs
                num_public_classes = len(api_docs.get("classes", {}))
                num_public_methods = len(api_docs.get("methods", {}))

                # Only save API docs if there are public APIs
                if num_public_classes > 0 or num_public_methods > 0:
                    api_docs_filename = f"{metadata['extension_id']}-{metadata['version']}.api_docs.json"
                    api_docs_file = api_docs_dir / api_docs_filename

                    # Convert to JSON and count tokens
                    api_docs_json_str = json.dumps(api_docs, indent=2)
                    api_docs_token_count = count_tokens(api_docs_json_str)

                    # Save API docs JSON
                    with open(api_docs_file, "w", encoding="utf-8") as f:
                        f.write(api_docs_json_str)
                    print(
                        f"  → Saved: api_docs/{api_docs_filename} ({num_public_classes} classes, {num_public_methods} methods, {api_docs_token_count:,} tokens)"
                    )
                else:
                    print(f"  → No public APIs to document")
            else:
                print(f"  → No Python API content found")

            # Add to database with token counts
            db_builder.add_extension(metadata, codeatlas_filename, codeatlas_token_count, api_docs_token_count)

            # Add to summary
            summary_entry = {
                "extension_id": metadata["extension_id"],
                "version": metadata["version"].split("+")[0],
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "has_python_api": metadata.get("has_python_api", False),
                "has_overview": metadata.get("has_overview", False),
                "total_modules": metadata.get("total_modules", 0),
                "total_classes": metadata.get("total_classes", 0),
                "total_methods": metadata.get("total_methods", 0),
                "codeatlas_token_count": codeatlas_token_count,
                "api_docs_token_count": api_docs_token_count,
                "overview_token_count": metadata.get("overview_token_count", 0),
            }
            summary["extensions"].append(summary_entry)
            summary["processed"] += 1

        except Exception as e:
            print(f"  ✗ Error processing {ext_dir.name}: {e}")
            traceback.print_exc()
            summary["failed"] += 1

    # Save the database
    db_file = db_builder.save(output_path)

    # Save summary
    summary_file = output_path / "extensions_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"  Total extensions: {len(extensions)}")
    print(f"  Processed: {summary['processed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Extensions database: {db_file}")
    print(f"  Summary saved to: {summary_file}")
    print(f"{'='*60}")


def main():
    """Main entry point for processing all extensions."""
    # Path to the extscache directory
    parser = argparse.ArgumentParser(description="Process all extensions in the extscache directory.")
    parser.add_argument(
        "--extensions-dir",
        type=str,
        default="/home/horde/repos/kit-app-template/_build/linux-x86_64/release/extscache",
        help="Path to the extensions directory",
    )
    parser.add_argument(
        "--include-source-code",
        type=bool,
        default=False,
        help="Set to True to include function source code (increases size significantly)",
    )
    args = parser.parse_args()

    print("\nProcessing ALL extensions...")
    print(f"Include source code: {args.include_source_code}")
    process_all_extensions(args.extensions_dir, include_source_code=args.include_source_code)


if __name__ == "__main__":
    # Default to processing all extensions
    main()
