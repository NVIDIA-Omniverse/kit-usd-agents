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

"""Test suite for get_extension_details function."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from kit_fns.functions.get_extension_details import get_extension_details
from kit_fns.services.extension_service import ExtensionService


class MockExtensionService:
    """Mock Extension Service for testing."""

    def __init__(self):
        self.available = True
        self.mock_extensions = {
            "omni.kit.window.viewport": {
                "title": "Viewport Window",
                "version": "1.2.3",
                "description": "Main viewport window for 3D scene viewing",
                "long_description": "Provides a comprehensive viewport interface with rendering capabilities, camera controls, and USD scene integration.",
                "category": "Rendering",
                "keywords": ["viewport", "rendering", "usd", "3d"],
                "dependencies": ["omni.ui", "omni.kit.viewport.core"],
                "has_python_api": True,
                "has_overview": True,
                "total_modules": 5,
                "total_classes": 20,
                "total_methods": 150,
                "codeatlas_token_count": 50000,
                "api_docs_token_count": 30000,
                "overview_token_count": 2000,
                "storage_path": "/path/to/extension",
            },
            "omni.ui": {
                "title": "UI Framework",
                "version": "2.0.0",
                "description": "Core UI framework with widgets and layouts",
                "long_description": "Comprehensive UI framework providing widgets, layouts, and interface components for building user interfaces.",
                "category": "UI",
                "keywords": ["ui", "widgets", "interface"],
                "dependencies": [],
                "has_python_api": True,
                "has_overview": True,
                "total_modules": 10,
                "total_classes": 50,
                "total_methods": 300,
                "codeatlas_token_count": 80000,
                "api_docs_token_count": 50000,
                "overview_token_count": 3000,
                "storage_path": "/path/to/ui",
            },
            "omni.physics": {
                "title": "Physics Engine",
                "version": "3.1.0",
                "description": "Physics simulation and dynamics",
                "long_description": "Advanced physics simulation engine with collision detection and dynamics.",
                "category": "Physics",
                "keywords": ["physics", "simulation", "collision"],
                "dependencies": ["omni.kit.viewport.core"],
                "has_python_api": True,
                "has_overview": False,
                "total_modules": 8,
                "total_classes": 35,
                "total_methods": 200,
                "codeatlas_token_count": 60000,
                "api_docs_token_count": 40000,
                "overview_token_count": 0,
                "storage_path": "/path/to/physics",
            },
        }

        self.api_symbols = {
            "omni.kit.window.viewport": [
                {"symbol": "ViewportWindow"},
                {"symbol": "ViewportAPI"},
                {"symbol": "CameraController"},
            ],
            "omni.ui": [
                {"symbol": "Widget"},
                {"symbol": "Layout"},
                {"symbol": "Button"},
                {"symbol": "Label"},
            ],
        }

    def is_available(self):
        return self.available

    def get_extension_list(self):
        return list(self.mock_extensions.keys())

    def get_extension_details(self, extension_ids):
        results = []
        for ext_id in extension_ids:
            ext_data = self.mock_extensions.get(ext_id)
            if ext_data:
                result = {
                    "id": ext_id,
                    "name": ext_data.get("title", ext_id),
                    "version": ext_data.get("version", ""),
                    "description": ext_data.get("description", ""),
                    "long_description": ext_data.get("long_description", ""),
                    "category": ext_data.get("category", ""),
                    "keywords": ext_data.get("keywords", []),
                    "dependencies": ext_data.get("dependencies", []),
                    "optional_dependencies": [],
                    "has_python_api": ext_data.get("has_python_api", False),
                    "has_overview": ext_data.get("has_overview", False),
                    "total_modules": ext_data.get("total_modules", 0),
                    "total_classes": ext_data.get("total_classes", 0),
                    "total_methods": ext_data.get("total_methods", 0),
                    "codeatlas_token_count": ext_data.get("codeatlas_token_count", 0),
                    "api_docs_token_count": ext_data.get("api_docs_token_count", 0),
                    "overview_token_count": ext_data.get("overview_token_count", 0),
                    "storage_path": ext_data.get("storage_path", ""),
                    "features": self._extract_features(ext_data),
                }

                # Add API symbols if available
                if ext_id in self.api_symbols:
                    result["apis"] = [s["symbol"] for s in self.api_symbols[ext_id]]
                    result["total_apis"] = len(self.api_symbols[ext_id])

                results.append(result)
            else:
                results.append(
                    {
                        "error": f"Extension '{ext_id}' not found",
                        "suggestion": "Use search_extensions to find available extensions",
                    }
                )
        return results

    def _extract_features(self, ext_data):
        features = []
        desc_lower = (ext_data.get("long_description", "") + " " + ext_data.get("description", "")).lower()
        if "rendering" in desc_lower:
            features.append("Rendering and visualization")
        if "ui" in desc_lower or "interface" in desc_lower or "widget" in desc_lower:
            features.append("User interface components")
        if "physics" in desc_lower:
            features.append("Physics simulation")
        if "usd" in desc_lower or "scene" in desc_lower:
            features.append("USD/Scene management")
        if ext_data.get("has_python_api"):
            features.append("Python API available")
        if ext_data.get("has_overview"):
            features.append("Comprehensive documentation")
        return features


# Mock telemetry
async def mock_ensure_telemetry_initialized():
    pass


class MockTelemetry:
    async def capture_call(self, **kwargs):
        pass


mock_telemetry = MockTelemetry()


async def run_tests():
    """Run all tests for get_extension_details."""
    print("=" * 80)
    print("TEST SUITE: get_extension_details")
    print("=" * 80)

    test_count = 0
    passed_count = 0
    failed_count = 0

    # Create mock service
    mock_service = MockExtensionService()

    with (
        patch("kit_fns.functions.get_extension_details.get_extension_service", return_value=mock_service),
        patch(
            "kit_fns.functions.get_extension_details.ensure_telemetry_initialized",
            side_effect=mock_ensure_telemetry_initialized,
        ),
        patch("kit_fns.functions.get_extension_details.telemetry", mock_telemetry),
    ):

        # Test 1: Single extension ID (string)
        test_count += 1
        print(f"\n[Test {test_count}] Single extension ID (string)")
        print("-" * 80)
        try:
            result = await get_extension_details("omni.kit.window.viewport")
            assert result["success"] is True, "Expected success=True"
            assert result["error"] is None, "Expected error=None"
            assert result["result"] != "", "Expected non-empty result"

            # Parse result JSON
            result_data = json.loads(result["result"])
            assert result_data["id"] == "omni.kit.window.viewport", "Expected correct extension ID"
            assert result_data["name"] == "Viewport Window", "Expected correct name"
            assert result_data["version"] == "1.2.3", "Expected correct version"
            assert "viewport" in result_data["description"].lower(), "Expected description to mention viewport"
            assert isinstance(result_data["dependencies"], list), "Expected dependencies to be a list"
            assert result_data["has_python_api"] is True, "Expected has_python_api=True"
            assert "apis" in result_data, "Expected API symbols to be included"
            assert len(result_data["features"]) > 0, "Expected features to be extracted"

            print(f"[OK] Single extension retrieved successfully")
            print(f"     ID: {result_data['id']}")
            print(f"     Name: {result_data['name']}")
            print(f"     Version: {result_data['version']}")
            print(f"     Features: {result_data['features']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 2: Multiple extension IDs (list)
        test_count += 1
        print(f"\n[Test {test_count}] Multiple extension IDs (list)")
        print("-" * 80)
        try:
            result = await get_extension_details(["omni.kit.window.viewport", "omni.ui", "omni.physics"])
            assert result["success"] is True, "Expected success=True"
            assert result["error"] is None, "Expected error=None"

            # Parse result JSON
            result_data = json.loads(result["result"])
            assert "extensions" in result_data, "Expected 'extensions' key in result"
            assert result_data["total_requested"] == 3, "Expected 3 requested extensions"
            assert result_data["successful"] == 3, "Expected 3 successful results"
            assert result_data["failed"] == 0, "Expected 0 failed results"
            assert len(result_data["extensions"]) == 3, "Expected 3 extension entries"

            # Verify each extension
            ext_ids = [e["id"] for e in result_data["extensions"]]
            assert "omni.kit.window.viewport" in ext_ids, "Expected viewport extension"
            assert "omni.ui" in ext_ids, "Expected UI extension"
            assert "omni.physics" in ext_ids, "Expected physics extension"

            print(f"[OK] Multiple extensions retrieved successfully")
            print(f"     Total requested: {result_data['total_requested']}")
            print(f"     Successful: {result_data['successful']}")
            print(f"     Failed: {result_data['failed']}")
            for ext in result_data["extensions"]:
                print(f"     - {ext['id']}: {ext['name']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 3: Non-existent extension
        test_count += 1
        print(f"\n[Test {test_count}] Non-existent extension")
        print("-" * 80)
        try:
            result = await get_extension_details("omni.nonexistent.extension")
            assert result["success"] is False, "Expected success=False for non-existent extension"
            assert result["error"] is not None, "Expected error message"
            assert "not found" in result["error"].lower(), "Expected 'not found' in error message"

            print(f"[OK] Non-existent extension handled correctly")
            print(f"     Error: {result['error']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 4: Mixed valid and invalid extensions
        test_count += 1
        print(f"\n[Test {test_count}] Mixed valid and invalid extensions")
        print("-" * 80)
        try:
            result = await get_extension_details(["omni.ui", "omni.invalid.extension", "omni.physics"])
            assert result["success"] is True, "Expected success=True (partial success)"

            result_data = json.loads(result["result"])
            assert result_data["total_requested"] == 3, "Expected 3 requested extensions"
            assert result_data["successful"] == 2, "Expected 2 successful results"
            assert result_data["failed"] == 1, "Expected 1 failed result"

            # Check for error entries
            error_entries = [e for e in result_data["extensions"] if "error" in e]
            assert len(error_entries) == 1, "Expected 1 error entry"

            print(f"[OK] Mixed extensions handled correctly")
            print(f"     Successful: {result_data['successful']}")
            print(f"     Failed: {result_data['failed']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 5: Empty string input
        test_count += 1
        print(f"\n[Test {test_count}] Empty string input")
        print("-" * 80)
        try:
            result = await get_extension_details("")
            assert result["success"] is False, "Expected success=False for empty string"
            assert "cannot be empty" in result["error"].lower(), "Expected 'cannot be empty' in error"

            print(f"[OK] Empty string input rejected correctly")
            print(f"     Error: {result['error']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 6: Empty list input
        test_count += 1
        print(f"\n[Test {test_count}] Empty list input")
        print("-" * 80)
        try:
            result = await get_extension_details([])
            assert result["success"] is False, "Expected success=False for empty list"
            assert "cannot be empty" in result["error"].lower(), "Expected 'cannot be empty' in error"

            print(f"[OK] Empty list input rejected correctly")
            print(f"     Error: {result['error']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 7: None input (get available extensions info)
        test_count += 1
        print(f"\n[Test {test_count}] None input (get available extensions info)")
        print("-" * 80)
        try:
            result = await get_extension_details(None)
            assert result["success"] is True, "Expected success=True for None input"
            assert result["error"] is None, "Expected error=None"

            result_data = json.loads(result["result"])
            assert "available_extensions" in result_data, "Expected 'available_extensions' key"
            assert "total_count" in result_data, "Expected 'total_count' key"
            assert result_data["total_count"] == 3, "Expected 3 available extensions"
            assert "usage" in result_data, "Expected usage information"

            print(f"[OK] None input handled correctly")
            print(f"     Total available: {result_data['total_count']}")
            print(f"     Extensions: {result_data['available_extensions']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 8: Invalid type input (number)
        test_count += 1
        print(f"\n[Test {test_count}] Invalid type input (number)")
        print("-" * 80)
        try:
            result = await get_extension_details(123)
            assert result["success"] is False, "Expected success=False for invalid type"
            assert "must be" in result["error"].lower(), "Expected type error message"

            print(f"[OK] Invalid type input rejected correctly")
            print(f"     Error: {result['error']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 9: Whitespace-only string input
        test_count += 1
        print(f"\n[Test {test_count}] Whitespace-only string input")
        print("-" * 80)
        try:
            result = await get_extension_details("   ")
            assert result["success"] is False, "Expected success=False for whitespace string"
            assert "cannot be empty" in result["error"].lower(), "Expected 'cannot be empty' in error"

            print(f"[OK] Whitespace-only string rejected correctly")
            print(f"     Error: {result['error']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 10: List with empty string
        test_count += 1
        print(f"\n[Test {test_count}] List with empty string")
        print("-" * 80)
        try:
            result = await get_extension_details(["omni.ui", "", "omni.physics"])
            assert result["success"] is False, "Expected success=False for list with empty string"
            assert "empty" in result["error"].lower(), "Expected 'empty' in error message"

            print(f"[OK] List with empty string rejected correctly")
            print(f"     Error: {result['error']}")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 11: Verify metadata completeness
        test_count += 1
        print(f"\n[Test {test_count}] Verify metadata completeness")
        print("-" * 80)
        try:
            result = await get_extension_details("omni.ui")
            assert result["success"] is True, "Expected success=True"

            result_data = json.loads(result["result"])
            required_fields = [
                "id",
                "name",
                "version",
                "description",
                "long_description",
                "category",
                "keywords",
                "dependencies",
                "has_python_api",
                "has_overview",
                "total_modules",
                "total_classes",
                "total_methods",
                "features",
            ]

            for field in required_fields:
                assert field in result_data, f"Expected field '{field}' in result"

            # Verify data types
            assert isinstance(result_data["keywords"], list), "Expected keywords to be a list"
            assert isinstance(result_data["dependencies"], list), "Expected dependencies to be a list"
            assert isinstance(result_data["features"], list), "Expected features to be a list"
            assert isinstance(result_data["has_python_api"], bool), "Expected has_python_api to be a bool"
            assert isinstance(result_data["total_classes"], int), "Expected total_classes to be an int"

            print(f"[OK] Metadata completeness verified")
            print(f"     All required fields present: {', '.join(required_fields[:5])}...")
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            failed_count += 1

        # Test 12: Service unavailable scenario
        test_count += 1
        print(f"\n[Test {test_count}] Service unavailable scenario")
        print("-" * 80)
        try:
            mock_service.available = False
            result = await get_extension_details("omni.ui")
            assert result["success"] is False, "Expected success=False when service unavailable"
            assert "not available" in result["error"].lower(), "Expected 'not available' in error"

            print(f"[OK] Service unavailable handled correctly")
            print(f"     Error: {result['error']}")
            mock_service.available = True  # Reset
            passed_count += 1
        except Exception as e:
            print(f"[FAIL] {str(e)}")
            mock_service.available = True  # Reset
            failed_count += 1

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {test_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {(passed_count/test_count*100):.1f}%")
    print("=" * 80)

    if failed_count == 0:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print(f"\n[FAIL] {failed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
