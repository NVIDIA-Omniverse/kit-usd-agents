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

"""Test script for get_app_examples function."""

import asyncio
import io
import sys
from pathlib import Path

# Ensure UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kit_fns.functions.get_app_examples import get_app_examples, list_app_examples


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


def print_result(result: dict, max_lines: int = 20):
    """Print test result with truncation."""
    if result.get("success"):
        print("  [OK] Success")
        if result.get("result"):
            lines = result["result"].split("\n")
            print(f"  Result preview (first {max_lines} lines):")
            for line in lines[:max_lines]:
                print(f"    {line}")
            if len(lines) > max_lines:
                print(f"    ... ({len(lines) - max_lines} more lines)")
    else:
        print(f"  [FAIL] Error: {result.get('error')}")

    # Print additional metadata
    if result.get("templates_retrieved"):
        print(f"  Templates retrieved: {result['templates_retrieved']}")
    if result.get("templates"):
        print(f"  Available templates: {len(result['templates'])} templates")


async def test_list_all_templates():
    """Test listing all templates (None input)."""
    print_section("Test 1: List All Templates (None input)")

    result = await get_app_examples(None)

    print_result(result, max_lines=30)

    # Validation
    success = result.get("success", False)
    has_result = result.get("result") is not None
    has_templates = result.get("templates") is not None

    print("\n  Validation:")
    print(f"    - Success: {success} {'[OK]' if success else '[FAIL]'}")
    print(f"    - Has result: {has_result} {'[OK]' if has_result else '[FAIL]'}")
    print(f"    - Has templates list: {has_templates} {'[OK]' if has_templates else '[FAIL]'}")

    if has_templates:
        templates = result["templates"]
        expected_templates = ["kit_base_editor", "usd_composer", "usd_explorer", "usd_viewer", "streaming_configs"]
        all_found = all(t in templates for t in expected_templates)
        print(f"    - All expected templates present: {all_found} {'[OK]' if all_found else '[FAIL]'}")

        if not all_found:
            print(f"      Expected: {expected_templates}")
            print(f"      Found: {templates}")

    return success and has_result and has_templates


async def test_single_template_string():
    """Test retrieving a single template using string input."""
    print_section("Test 2: Single Template (String Input)")

    template_id = "kit_base_editor"
    print(f"  Requesting template: '{template_id}'")

    result = await get_app_examples(template_id)

    print_result(result, max_lines=25)

    # Validation
    success = result.get("success", False)
    has_result = result.get("result") is not None
    retrieved = result.get("templates_retrieved", [])

    print("\n  Validation:")
    print(f"    - Success: {success} {'[OK]' if success else '[FAIL]'}")
    print(f"    - Has result: {has_result} {'[OK]' if has_result else '[FAIL]'}")
    print(
        f"    - Correct template retrieved: {retrieved == [template_id]} {'[OK]' if retrieved == [template_id] else '[FAIL]'}"
    )

    if has_result:
        result_text = result["result"]
        # Check for key content
        checks = {
            "Template name": "Kit Base Editor" in result_text,
            "Template ID": template_id in result_text,
            "Description section": "## Description" in result_text,
            "Use Cases section": "## Use Cases" in result_text,
            "Key Features section": "## Key Features" in result_text,
            "README section": "## README Documentation" in result_text,
            "Kit file section": "## Kit Configuration File" in result_text,
        }

        for check_name, check_result in checks.items():
            status = "[OK]" if check_result else "[FAIL]"
            print(f"    - Contains {check_name}: {check_result} {status}")

    return success and has_result and retrieved == [template_id]


async def test_multiple_templates_list():
    """Test retrieving multiple templates using list input."""
    print_section("Test 3: Multiple Templates (List Input)")

    template_ids = ["usd_viewer", "usd_explorer"]
    print(f"  Requesting templates: {template_ids}")

    result = await get_app_examples(template_ids)

    print_result(result, max_lines=25)

    # Validation
    success = result.get("success", False)
    has_result = result.get("result") is not None
    retrieved = result.get("templates_retrieved", [])

    print("\n  Validation:")
    print(f"    - Success: {success} {'[OK]' if success else '[FAIL]'}")
    print(f"    - Has result: {has_result} {'[OK]' if has_result else '[FAIL]'}")
    print(
        f"    - Correct templates retrieved: {set(retrieved) == set(template_ids)} {'[OK]' if set(retrieved) == set(template_ids) else '[FAIL]'}"
    )

    if has_result:
        result_text = result["result"]
        # Check that both templates are in result
        checks = {
            "USD Viewer present": "USD Viewer" in result_text,
            "USD Explorer present": "USD Explorer" in result_text,
            "Template count header": f"({len(template_ids)} templates)" in result_text,
            "Separator between templates": "=" * 80 in result_text,
        }

        for check_name, check_result in checks.items():
            status = "[OK]" if check_result else "[FAIL]"
            print(f"    - {check_name}: {check_result} {status}")

    return success and has_result and set(retrieved) == set(template_ids)


async def test_streaming_configs():
    """Test retrieving streaming configurations template."""
    print_section("Test 4: Streaming Configurations Template")

    template_id = "streaming_configs"
    print(f"  Requesting template: '{template_id}'")

    result = await get_app_examples(template_id)

    print_result(result, max_lines=25)

    # Validation
    success = result.get("success", False)
    has_result = result.get("result") is not None

    print("\n  Validation:")
    print(f"    - Success: {success} {'[OK]' if success else '[FAIL]'}")
    print(f"    - Has result: {has_result} {'[OK]' if has_result else '[FAIL]'}")

    if has_result:
        result_text = result["result"]
        # Check for streaming config specific content
        checks = {
            "Streaming Configurations title": "Streaming Configuration" in result_text,
            "Default Streaming": "Default Streaming" in result_text,
            "GDN Streaming": "GDN Streaming" in result_text,
            "NVCF Streaming": "NVCF Streaming" in result_text,
            "Configuration files": "default_stream.kit" in result_text or "gdn_stream.kit" in result_text,
        }

        for check_name, check_result in checks.items():
            status = "[OK]" if check_result else "[FAIL]"
            print(f"    - Contains {check_name}: {check_result} {status}")

    return success and has_result


async def test_invalid_template():
    """Test handling of invalid template ID."""
    print_section("Test 5: Invalid Template ID")

    template_id = "nonexistent_template"
    print(f"  Requesting invalid template: '{template_id}'")

    result = await get_app_examples(template_id)

    print_result(result)

    # Validation - should fail gracefully
    success = result.get("success", False)
    has_error = result.get("error") is not None

    print("\n  Validation:")
    print(f"    - Success (should be False): {not success} {'[OK]' if not success else '[FAIL]'}")
    print(f"    - Has error message: {has_error} {'[OK]' if has_error else '[FAIL]'}")

    if has_error:
        error_msg = result["error"]
        contains_unknown = "Unknown template" in error_msg or "unknown" in error_msg.lower()
        print(f"    - Error mentions unknown template: {contains_unknown} {'[OK]' if contains_unknown else '[FAIL]'}")
        print(f"    - Error message: {error_msg}")

    return not success and has_error


async def test_mixed_valid_invalid():
    """Test handling of mixed valid and invalid template IDs."""
    print_section("Test 6: Mixed Valid and Invalid Template IDs")

    template_ids = ["kit_base_editor", "invalid_template", "usd_viewer"]
    print(f"  Requesting templates: {template_ids}")

    result = await get_app_examples(template_ids)

    print_result(result)

    # Validation - should fail with invalid ID
    success = result.get("success", False)
    has_error = result.get("error") is not None

    print("\n  Validation:")
    print(f"    - Success (should be False): {not success} {'[OK]' if not success else '[FAIL]'}")
    print(f"    - Has error message: {has_error} {'[OK]' if has_error else '[FAIL]'}")

    if has_error:
        error_msg = result["error"]
        mentions_invalid = "invalid_template" in error_msg
        print(f"    - Error mentions invalid template: {mentions_invalid} {'[OK]' if mentions_invalid else '[FAIL]'}")
        print(f"    - Error message: {error_msg}")

    return not success and has_error


async def test_invalid_input_type():
    """Test handling of invalid input type."""
    print_section("Test 7: Invalid Input Type")

    invalid_input = 12345  # Integer instead of string or list
    print(f"  Requesting with invalid type: {type(invalid_input).__name__}")

    result = await get_app_examples(invalid_input)

    print_result(result)

    # Validation - should fail with type error
    success = result.get("success", False)
    has_error = result.get("error") is not None

    print("\n  Validation:")
    print(f"    - Success (should be False): {not success} {'[OK]' if not success else '[FAIL]'}")
    print(f"    - Has error message: {has_error} {'[OK]' if has_error else '[FAIL]'}")

    if has_error:
        error_msg = result["error"]
        mentions_type = "Invalid input type" in error_msg or "type" in error_msg.lower()
        print(f"    - Error mentions type issue: {mentions_type} {'[OK]' if mentions_type else '[FAIL]'}")
        print(f"    - Error message: {error_msg}")

    return not success and has_error


async def test_output_format():
    """Test that output format is consistent and well-formatted."""
    print_section("Test 8: Output Format Validation")

    template_id = "usd_composer"
    print(f"  Requesting template: '{template_id}'")

    result = await get_app_examples(template_id)

    if not result.get("success"):
        print(f"  [FAIL] Could not retrieve template: {result.get('error')}")
        return False

    result_text = result["result"]
    lines = result_text.split("\n")

    print(f"  [OK] Retrieved template successfully")
    print(f"  Total lines: {len(lines)}")
    print(f"  Total characters: {len(result_text)}")

    # Check formatting
    checks = {
        "Has markdown headers": any(line.startswith("#") for line in lines),
        "Has code blocks": "```" in result_text,
        "Has bullet points": any(line.strip().startswith("-") for line in lines),
        "Has bold text": "**" in result_text,
        "Non-empty": len(result_text.strip()) > 0,
        "Multiple sections": result_text.count("##") >= 3,
    }

    print("\n  Format Validation:")
    all_passed = True
    for check_name, check_result in checks.items():
        status = "[OK]" if check_result else "[FAIL]"
        print(f"    - {check_name}: {check_result} {status}")
        if not check_result:
            all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("=" * 70)
    print(" Test Suite: get_app_examples Function")
    print("=" * 70)

    # Run tests
    tests = [
        ("List All Templates", test_list_all_templates()),
        ("Single Template (String)", test_single_template_string()),
        ("Multiple Templates (List)", test_multiple_templates_list()),
        ("Streaming Configurations", test_streaming_configs()),
        ("Invalid Template ID", test_invalid_template()),
        ("Mixed Valid/Invalid IDs", test_mixed_valid_invalid()),
        ("Invalid Input Type", test_invalid_input_type()),
        ("Output Format Validation", test_output_format()),
    ]

    results = []
    for test_name, test_coro in tests:
        try:
            result = asyncio.run(test_coro)
            results.append((test_name, result, None))
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' raised exception: {e}")
            results.append((test_name, False, str(e)))

    # Print summary
    print_section("Test Summary")

    passed = 0
    failed = 0

    for test_name, result, error in results:
        if result:
            print(f"  [OK]   {test_name}")
            passed += 1
        else:
            print(f"  [FAIL] {test_name}")
            if error:
                print(f"         Exception: {error}")
            failed += 1

    print(f"\n  Total: {len(results)} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed == 0:
        print("\n  [OK] All tests passed!")
        return 0
    else:
        print(f"\n  [FAIL] {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
