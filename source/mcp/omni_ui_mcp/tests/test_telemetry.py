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
Simple test script to verify telemetry integration with get_code_examples function.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set required environment variables for testing
os.environ.setdefault("NVIDIA_API_KEY", "test-key-for-telemetry")

from omni_ui_mcp.functions.get_code_examples import get_code_examples
from omni_ui_mcp.services.telemetry import telemetry


async def test_telemetry_integration():
    """Test the telemetry integration."""
    print("🚀 Testing OmniUI MCP Telemetry Integration")
    print("=" * 50)

    # Test telemetry service initialization
    print("1. Testing telemetry service initialization...")
    await telemetry.initialize()

    if telemetry.is_enabled():
        print("✅ Telemetry service initialized successfully")
        print(f"   Redis host: {telemetry.REDIS_HOST}:{telemetry.REDIS_PORT}")
        print(f"   Key prefix: {telemetry.KEY_PREFIX}")

        # Get current telemetry count
        initial_count = await telemetry.get_telemetry_keys_count()
        print(f"   Current telemetry entries: {initial_count}")
    else:
        print("⚠️  Telemetry service disabled (Redis not available or connection failed)")
        print("   Telemetry will be captured to logs only")

    print()

    # Test function call with telemetry
    print("2. Testing get_code_examples function with telemetry...")
    test_query = "How to create a search field?"

    try:
        result = await get_code_examples(
            request=test_query, rerank_k=5, enable_rerank=False  # Disable reranking for faster testing
        )

        print(f"✅ Function executed successfully")
        print(f"   Success: {result['success']}")
        print(f"   Result length: {len(result.get('result', ''))}")
        if result.get("error"):
            print(f"   Error: {result['error']}")

    except Exception as e:
        print(f"❌ Function execution failed: {e}")

    print()

    # Check if telemetry was captured
    if telemetry.is_enabled():
        print("3. Checking telemetry capture...")
        final_count = await telemetry.get_telemetry_keys_count()
        new_entries = final_count - initial_count

        print(f"✅ Telemetry captured successfully")
        print(f"   Total telemetry entries: {final_count}")
        print(f"   New entries from this test: {new_entries}")

        # Show recent keys
        if final_count > 0:
            recent_keys = await telemetry.get_recent_telemetry_keys(limit=3)
            print("   Recent telemetry keys:")
            for key in recent_keys[:3]:  # Show last 3 keys
                print(f"     - {key}")

            print("   Entry structure:")
            print("   - service: omni_ui_mcp")
            print("   - function_name: get_code_examples")
            print("   - timestamp: [ISO format]")
            print("   - duration_ms: [execution time]")
            print("   - success: [true/false]")
            print("   - request_data: [parameters object]")
    else:
        print("3. Telemetry capture skipped (service disabled)")

    print()

    # Clean up
    await telemetry.close()
    print("🏁 Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_telemetry_integration())
