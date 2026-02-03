#!/usr/bin/env python
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

"""Test script to verify the Kit MCP services work with generated FAISS database and Code Atlas data."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import services
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_mcp.services.api_service import APIService
from kit_mcp.services.extension_service import ExtensionService
from kit_mcp.services.kit_exts_atlas import KitExtensionsAtlasService


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def test_atlas_service():
    """Test the Kit Extensions Atlas service."""
    print_section("Testing Kit Extensions Atlas Service")

    atlas_service = KitExtensionsAtlasService()

    # Test availability
    print(f"Atlas service available: {atlas_service.is_available()}")

    if not atlas_service.is_available():
        print("ERROR: Atlas service not available. Run build_extension_database.py first.")
        return False

    # Get extension list
    extensions = atlas_service.get_extension_list()
    print(f"Total extensions in database: {len(extensions)}")

    if extensions:
        # Test with first extension
        test_ext = extensions[0]
        print(f"\nTesting with extension: {test_ext}")

        # Get metadata
        metadata = atlas_service.get_extension_metadata(test_ext)
        if metadata:
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  Version: {metadata.get('version', 'N/A')}")
            print(f"  Has Python API: {metadata.get('has_python_api', False)}")
            print(f"  Total classes: {metadata.get('total_classes', 0)}")
            print(f"  Total methods: {metadata.get('total_methods', 0)}")

        # Try to load Code Atlas
        codeatlas = atlas_service.load_codeatlas(test_ext)
        if codeatlas:
            print(f"  Code Atlas loaded: ✓")
            print(f"    Modules: {len(codeatlas.get('modules', {}))}")
            print(f"    Classes: {len(codeatlas.get('classes', {}))}")
            print(f"    Methods: {len(codeatlas.get('methods', {}))}")
        else:
            print(f"  Code Atlas: Not available")

        # Try to load API docs
        api_docs = atlas_service.load_api_docs(test_ext)
        if api_docs:
            print(f"  API docs loaded: ✓")
            print(f"    Classes: {len(api_docs.get('classes', {}))}")
            print(f"    Functions: {len(api_docs.get('functions', {}))}")
        else:
            print(f"  API docs: Not available")

        # Get API symbols
        api_symbols = atlas_service.get_api_symbols(test_ext)
        if api_symbols:
            print(f"  API symbols: {len(api_symbols)} found")
            if len(api_symbols) > 0:
                print(f"    Example: {api_symbols[0]['api_reference']}")

    return True


def test_extension_service():
    """Test the Extension service with FAISS."""
    print_section("Testing Extension Service with FAISS")

    extension_service = ExtensionService()

    # Test availability
    print(f"Extension service available: {extension_service.is_available()}")

    if not extension_service.is_available():
        print("ERROR: Extension service not available.")
        return False

    # Test semantic search
    print("\nTesting semantic search...")

    test_queries = ["user interface widgets", "viewport rendering", "physics simulation", "USD scene management"]

    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = extension_service.search_extensions(query, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['id']} (score: {result.get('relevance_score', 0):.2f})")
                print(f"     {result.get('description', 'No description')[:100]}...")
        else:
            print("  No results found")

    # Test get_extension_details
    print("\nTesting get_extension_details...")
    test_extensions = ["omni.ui", "omni.kit.window.console", "non.existent.extension"]

    details = extension_service.get_extension_details(test_extensions)
    for detail in details:
        if "error" in detail:
            print(f"  {detail.get('matched_query', 'Unknown')}: {detail['error']}")
        else:
            print(f"  {detail['id']}: {detail.get('name', 'N/A')}")
            print(f"    Features: {', '.join(detail.get('features', []))}")
            print(f"    APIs: {detail.get('total_apis', 0)}")

    # Test dependency analysis
    print("\nTesting dependency analysis...")
    if extension_service.get_extension_list():
        test_ext = extension_service.get_extension_list()[0]
        deps = extension_service.get_extension_dependencies(test_ext, depth=1)
        if "error" not in deps:
            print(f"  Dependencies for {deps['extension_id']}:")
            required = deps.get("dependencies", {}).get("required", [])
            print(f"    Required: {', '.join(required[:3]) if required else 'None'}")

    return True


def test_api_service():
    """Test the API service."""
    print_section("Testing API Service")

    api_service = APIService()

    # Test availability
    print(f"API service available: {api_service.is_available()}")

    if not api_service.is_available():
        print("ERROR: API service not available.")
        return False

    # Find extensions with APIs
    atlas_service = KitExtensionsAtlasService()
    extensions_with_apis = []

    for ext_id in atlas_service.get_extension_list()[:20]:  # Check first 20
        metadata = atlas_service.get_extension_metadata(ext_id)
        if metadata and metadata.get("has_python_api"):
            extensions_with_apis.append(ext_id)
            if len(extensions_with_apis) >= 2:
                break

    if not extensions_with_apis:
        print("No extensions with APIs found in first 20 extensions")
        return False

    print(f"Testing with extensions: {extensions_with_apis}")

    # Test get_extension_apis
    print("\nTesting get_extension_apis...")
    api_results = api_service.get_extension_apis(extensions_with_apis)

    for result in api_results:
        print(f"\n  Extension: {result['extension_id']}")
        print(f"    API count: {result.get('api_count', 0)}")
        print(f"    Classes: {result.get('class_count', 0)}")
        print(f"    Functions: {result.get('function_count', 0)}")

        if result.get("apis"):
            print(f"    Sample APIs:")
            for api in result["apis"][:3]:
                print(f"      - {api['symbol']} ({api['type']})")

    # Test get_api_details
    print("\nTesting get_api_details...")

    # Collect some API references
    api_refs = []
    for result in api_results:
        if result.get("apis"):
            for api in result["apis"][:2]:
                api_refs.append(api["api_reference"])
                if len(api_refs) >= 3:
                    break
        if len(api_refs) >= 3:
            break

    if api_refs:
        print(f"Getting details for: {api_refs}")
        api_details = api_service.get_api_details(api_refs)

        for detail in api_details:
            if "error" in detail:
                print(f"\n  {detail['api_reference']}: ERROR")
                print(f"    {detail['error']}")
                if detail.get("suggestion"):
                    print(f"    Suggestion: {detail['suggestion']}")
            else:
                print(f"\n  {detail['api_reference']}: SUCCESS")
                print(f"    Type: {detail.get('type', 'Unknown')}")
                print(f"    Signature: {detail.get('signature', 'N/A')}")
                docstring = detail.get("docstring", "")
                if docstring:
                    print(f"    Docstring: {docstring[:100]}...")

    # Test API list
    print("\nTesting get_api_list...")
    all_apis = api_service.get_api_list()
    print(f"Total API references available: {len(all_apis)}")
    if all_apis:
        print(f"Sample API references:")
        for ref in all_apis[:5]:
            print(f"  - {ref}")

    return True


async def test_functions():
    """Test the actual MCP functions."""
    print_section("Testing MCP Functions")

    # Import functions
    from kit_mcp.functions.get_api_details import get_api_details
    from kit_mcp.functions.get_extension_apis import get_extension_apis
    from kit_mcp.functions.get_extension_details import get_extension_details
    from kit_mcp.functions.search_extensions import search_extensions

    # Test search_extensions
    print("Testing search_extensions function...")
    result = await search_extensions("viewport rendering", top_k=3)
    if result["success"]:
        print("  ✓ search_extensions works")
        # Print first few lines of result
        lines = result["result"].split("\n")[:5]
        for line in lines:
            print(f"    {line}")
    else:
        print(f"  ✗ search_extensions failed: {result['error']}")

    # Test get_extension_details
    print("\nTesting get_extension_details function...")
    result = await get_extension_details(["omni.ui"])
    if result["success"]:
        print("  ✓ get_extension_details works")
        data = json.loads(result["result"])
        print(f"    Extension: {data.get('id', 'N/A')}")
        print(f"    Name: {data.get('name', 'N/A')}")
        print(f"    APIs: {data.get('total_apis', 0)}")
    else:
        print(f"  ✗ get_extension_details failed: {result['error']}")

    # Test get_extension_apis
    print("\nTesting get_extension_apis function...")
    result = await get_extension_apis(["omni.ui"])
    if result["success"]:
        print("  ✓ get_extension_apis works")
        data = json.loads(result["result"])
        print(f"    Extension: {data.get('extension_id', 'N/A')}")
        print(f"    API count: {data.get('api_count', 0)}")
    else:
        print(f"  ✗ get_extension_apis failed: {result['error']}")

    # Test get_api_details
    print("\nTesting get_api_details function...")
    result = await get_api_details(["omni.ui@Window"])
    if result["success"]:
        print("  ✓ get_api_details works")
        data = json.loads(result["result"])
        print(f"    API: {data.get('symbol', 'N/A')}")
        print(f"    Type: {data.get('type', 'N/A')}")
    else:
        print(f"  ✗ get_api_details failed: {result['error']}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print(" Kit MCP Services Test Suite")
    print("=" * 60)

    # Check if NVIDIA_API_KEY is set
    if not os.getenv("NVIDIA_API_KEY"):
        print("\nWARNING: NVIDIA_API_KEY not set. FAISS search may not work.")
        print("Set it with: export NVIDIA_API_KEY=your_api_key")

    # Run tests
    tests_passed = []

    # Test Atlas service
    tests_passed.append(("Atlas Service", test_atlas_service()))

    # Test Extension service
    tests_passed.append(("Extension Service", test_extension_service()))

    # Test API service
    tests_passed.append(("API Service", test_api_service()))

    # Test MCP functions
    tests_passed.append(("MCP Functions", asyncio.run(test_functions())))

    # Print summary
    print_section("Test Summary")

    all_passed = True
    for test_name, passed in tests_passed:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
