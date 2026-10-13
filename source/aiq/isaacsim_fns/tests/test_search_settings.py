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

"""Test script for the search_settings functionality."""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isaacsim_fns.functions.search_settings import search_settings
from isaacsim_fns.services.settings_service import SettingsService

# These tests query the hosted embedding endpoint. Without a key the backend
# answers 401, which the search services now report as an error rather than
# falling back to keyword matches.
requires_nvidia_api_key = pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"), reason="NVIDIA_API_KEY not set; live embedding endpoint required"
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


@requires_nvidia_api_key
def test_settings_service():
    """Test the Settings service directly."""
    print_section("Testing Settings Service")

    settings_service = SettingsService()

    # Test availability
    print(f"Settings service available: {settings_service.is_available()}")

    if not settings_service.is_available():
        print("[FAIL] Settings service not available. Run the settings pipeline first.")
        assert False, "Settings service not available. Run the settings pipeline first."

    print("[OK] Settings service is available")

    # Get statistics
    stats = settings_service.get_settings_statistics()
    print(f"Total settings: {stats.get('total_settings', 0)}")
    print(f"Extensions scanned: {stats.get('total_extensions_scanned', 0)}")
    print(f"FAISS available: {stats.get('faiss_available', False)}")
    print(
        f"Documented settings: {stats.get('documented_settings', 0)} ({stats.get('documentation_percentage', 0):.1f}%)"
    )

    # Show type distribution
    if "type_distribution" in stats:
        print("\nSetting types:")
        for type_name, count in sorted(stats["type_distribution"].items(), key=lambda x: (x[0] is None, x[0] or "")):
            print(f"  {type_name}: {count}")

    # Show prefix distribution
    if "prefix_distribution" in stats:
        print("\nSetting prefixes:")
        for prefix, count in sorted(stats["prefix_distribution"].items()):
            print(f"  /{prefix}/: {count} settings")

    # Test search
    print("\n" + "-" * 40)
    print("Testing semantic search...")

    test_queries = [
        ("viewport rendering", None, None),
        ("enable debug", None, "bool"),
        # Settings corpus only contains /exts/-prefixed entries; the previous
        # ("window settings", "app", ...) test always returned 0 hits because
        # no setting has the "app" prefix in the current data.
        ("isaac sim startup", "exts", None),
    ]

    all_tests_passed = True
    for query, prefix, type_filter in test_queries:
        filter_desc = []
        if prefix:
            filter_desc.append(f"prefix={prefix}")
        if type_filter:
            filter_desc.append(f"type={type_filter}")
        filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""

        print(f"\nSearching for: '{query}'{filter_str}")
        results = settings_service.search_settings(query, top_k=3, prefix_filter=prefix, type_filter=type_filter)

        if results:
            print(f"[OK] Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['setting']}")
                print(f"     Type: {result.get('type', 'unknown')}, Default: {result.get('default_value', 'None')}")
                print(f"     Score: {result.get('relevance_score', 0):.2f}")
                doc = result.get("documentation", "") or result.get("description", "")
                if doc:
                    doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
                    print(f"     Doc: {doc_preview}")
        else:
            print("[FAIL] No results found")
            all_tests_passed = False

    # Test getting settings by extension
    print("\n" + "-" * 40)
    print("Testing get settings by extension...")
    test_ext = "omni.kit.viewport.window"
    ext_settings = settings_service.get_settings_by_extension(test_ext)
    print(f"Settings for {test_ext}: {len(ext_settings)} found")
    if ext_settings:
        print(f"[OK] Found settings for extension")
        for setting in ext_settings[:3]:
            print(f"  - {setting['setting']} ({setting['type']})")
    else:
        print(f"[INFO] No settings found for {test_ext} (may be expected)")

    # Test get_setting_details
    print("\n" + "-" * 40)
    print("Testing get_setting_details...")

    # Get a valid setting key from search results
    test_results = settings_service.search_settings("viewport", top_k=1)
    if test_results:
        test_key = test_results[0]["setting"]
        details = settings_service.get_setting_details([test_key])
        if details and "error" not in details[0]:
            print(f"[OK] Retrieved details for {test_key}")
            print(f"  Type: {details[0].get('type')}")
            print(f"  Extensions: {len(details[0].get('extensions', []))} extensions")
        else:
            print(f"[FAIL] Failed to retrieve details for {test_key}")
            all_tests_passed = False

    # Test with invalid setting key
    invalid_details = settings_service.get_setting_details(["invalid.setting.key"])
    if invalid_details and "error" in invalid_details[0]:
        print("[OK] Correctly handled invalid setting key")
    else:
        print("[FAIL] Did not handle invalid setting key properly")
        all_tests_passed = False

    assert all_tests_passed, "Some settings service tests failed"


@requires_nvidia_api_key
async def test_search_function():
    """Test the search_settings function."""
    print_section("Testing search_settings Function")

    all_tests_passed = True

    # Test basic search
    print("Test 1: Basic search...")
    result = await search_settings("viewport grid settings", top_k=5)

    if result["success"]:
        print("[OK] Basic search successful")
        # Print first few lines of result
        lines = result["result"].split("\n")[:10]
        for line in lines:
            if line.strip():
                print(f"  {line}")
    else:
        print(f"[FAIL] Basic search failed: {result['error']}")
        all_tests_passed = False

    # Test with filters
    print("\nTest 2: Search with prefix and type filters...")
    result = await search_settings("enable features", top_k=5, prefix_filter="exts", type_filter="bool")

    if result["success"]:
        print("[OK] Filtered search successful")
        # Count results
        import re

        matches = re.findall(r"## \d+\.", result["result"])
        print(f"  Found {len(matches)} settings with filters")

        # Verify filters were applied
        if "Filtered by prefix: exts" in result["result"]:
            print("[OK] Prefix filter applied")
        else:
            print("[FAIL] Prefix filter not shown in results")
            all_tests_passed = False

        if "Filtered by type: bool" in result["result"]:
            print("[OK] Type filter applied")
        else:
            print("[FAIL] Type filter not shown in results")
            all_tests_passed = False
    else:
        print(f"[FAIL] Filtered search failed: {result['error']}")
        all_tests_passed = False

    # Test with no results (or low relevance results with FAISS)
    print("\nTest 3: Search with unlikely query...")
    result = await search_settings("xyzabc123impossiblequery9876", top_k=5)

    if result["success"]:
        # With FAISS semantic search, it will return results (lowest similarity)
        # Without FAISS, it should return "No settings found"
        if "No settings found" in result["result"] or "Found" in result["result"]:
            print("[OK] Search handled nonsensical query correctly")
            # Check that relevance scores are low if results were returned
            if "Relevance Score:" in result["result"]:
                print("[INFO] FAISS returned low-similarity results (expected)")
        else:
            print("[FAIL] Unexpected response format")
            all_tests_passed = False
    else:
        print(f"[FAIL] Search failed: {result['error']}")
        all_tests_passed = False

    # Test with different top_k values
    print("\nTest 4: Search with different top_k values...")
    for top_k in [1, 5, 10]:
        result = await search_settings("viewport", top_k=top_k)
        if result["success"]:
            import re

            matches = re.findall(r"## \d+\.", result["result"])
            actual_count = len(matches)
            if actual_count <= top_k:
                print(f"[OK] top_k={top_k}: returned {actual_count} results (as expected)")
            else:
                print(f"[FAIL] top_k={top_k}: returned {actual_count} results (more than requested)")
                all_tests_passed = False
        else:
            print(f"[FAIL] Search with top_k={top_k} failed: {result['error']}")
            all_tests_passed = False

    # Test return format validation
    print("\nTest 5: Validate return format...")
    result = await search_settings("rendering", top_k=3)

    required_keys = ["success", "result", "error"]
    if all(key in result for key in required_keys):
        print("[OK] Return format contains all required keys")
    else:
        print(f"[FAIL] Return format missing keys. Has: {result.keys()}")
        all_tests_passed = False

    if result["success"] and isinstance(result["result"], str):
        print("[OK] Result is a string when successful")
    elif not result["success"] and result["error"]:
        print("[OK] Error message provided when unsuccessful")
    else:
        print("[FAIL] Return format inconsistent")
        all_tests_passed = False

    # Test that telemetry doesn't break the function
    print("\nTest 6: Function completes despite telemetry...")
    result = await search_settings("test query", top_k=1)
    if result["success"] or result["error"]:
        print("[OK] Function returns properly with telemetry")
    else:
        print("[FAIL] Function did not complete properly")
        all_tests_passed = False

    assert all_tests_passed, "Some search function tests failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefix_filter", "use_hybrid"),
    [("persistent", False), (None, True)],
    ids=["dense-filtered", "hybrid"],
)
async def test_search_settings_reports_embedding_backend_error(tmp_path, monkeypatch, prefix_filter, use_hybrid):
    """A failing embedding backend must return success=False, not keyword-fallback settings."""
    import json

    import isaacsim_fns.services.settings_service as settings_module
    from mock_embedding_backend import FailingEmbedder, stub_hybrid_shim, write_faiss_bundle

    setting_key = "/persistent/isaac/asset_root/timeout"
    bundle = write_faiss_bundle(
        tmp_path / "settings_faiss",
        [(setting_key, {"setting_key": setting_key, "prefix": "persistent", "type": "int"})],
    )
    summary = tmp_path / "setting_summary.json"
    summary.write_text(
        json.dumps(
            {
                "settings": {
                    setting_key: {"type": "int", "default_value": 10, "documentation": "Asset root timeout"},
                },
                "metadata": {},
            }
        )
    )

    monkeypatch.setenv("OVAI_RETRIEVAL_MODE", "semantic")
    stub_hybrid_shim(monkeypatch)
    embedder = FailingEmbedder(status_code=410, reason="Gone")
    with patch.object(settings_module.EmbedderFactory, "create", return_value=embedder):
        service = SettingsService(faiss_db_path=str(bundle), settings_summary_path=str(summary))
    assert service.vectorstore is not None, "fixture index failed to load"
    if use_hybrid:
        # Mirrors ovgenai_retrieval's SemanticSearcher: the hybrid retriever embeds the
        # query with the service's (guarded) embedder and does not catch its errors.
        service._hybrid = _QueryEmbeddingHybridRetriever(service.embedder)

    with patch("isaacsim_fns.functions.search_settings.get_settings_service", return_value=service):
        with patch("isaacsim_fns.functions.search_settings.ensure_telemetry_initialized", new_callable=AsyncMock):
            with patch("isaacsim_fns.functions.search_settings.telemetry") as mock_telemetry:
                mock_telemetry.capture_call = AsyncMock()

                result = await search_settings("asset root timeout", top_k=5, prefix_filter=prefix_filter)

    assert embedder.calls >= 1, "the embedding backend was never queried"
    assert result["success"] is False, f"backend failure reported as success: {result!r}"
    assert "410" in result["error"]
    assert "embedding" in result["error"].lower()
    assert setting_key not in result["result"]
    if use_hybrid:
        # The hybrid failure must be raised directly, not retried via the dense FAISS path.
        assert embedder.calls == 1


class _QueryEmbeddingHybridRetriever:
    """Hybrid retriever stand-in whose retrieve() embeds the query like the real semantic leg."""

    def __init__(self, embedder):
        self._embedder = embedder

    def retrieve(self, query, top_k):
        self._embedder.embed_query(query)
        return []


def main():
    """Run all tests."""
    print("=" * 60)
    print(" Isaac Sim Settings Search Test Suite")
    print("=" * 60)

    # Check if NVIDIA_API_KEY is set
    if not os.getenv("NVIDIA_API_KEY"):
        print("\n[INFO] NVIDIA_API_KEY not set. FAISS search may not work.")
        print("[INFO] Falling back to keyword search.")

    # Run tests
    tests_passed = []

    # Test Settings service
    print("\n[INFO] Running Settings Service tests...")
    try:
        test_settings_service()
        tests_passed.append(("Settings Service", True))
    except AssertionError as e:
        print(f"[FAIL] Settings service test failed: {e}")
        tests_passed.append(("Settings Service", False))

    # Test search function
    print("\n[INFO] Running Search Function tests...")
    try:
        asyncio.run(test_search_function())
        tests_passed.append(("Search Function", True))
    except AssertionError as e:
        print(f"[FAIL] Search function test failed: {e}")
        tests_passed.append(("Search Function", False))

    # Print summary
    print_section("Test Summary")

    all_passed = True
    for test_name, passed in tests_passed:
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("[OK] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
