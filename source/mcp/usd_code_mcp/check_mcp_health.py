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
Simple MCP health check script that tests actual MCP functionality.
This script attempts to connect to the MCP server and perform a basic operation.
Exit code 0 = healthy, non-zero = unhealthy.

Updated for NAT 1.3+ which uses streamable-http at /mcp/ instead of SSE at /sse.
"""

import asyncio
import logging
import os
import sys

import aiohttp

# Suppress unnecessary logging
logging.basicConfig(level=logging.ERROR)


async def check_mcp_endpoint(port: int = 9903, timeout: int = 5) -> bool:
    """
    Test if MCP server's streamable-http endpoint is accessible.
    NAT 1.3+ uses /mcp/ as the primary endpoint for MCP protocol.
    """
    try:
        url = f"http://localhost:{port}/mcp/"

        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            # POST to /mcp/ with MCP initialize message
            # This tests if the streamable-http endpoint is working
            mcp_init = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "health-check", "version": "1.0.0"},
                },
            }
            async with session.post(url, json=mcp_init) as response:
                if response.status == 200:
                    # Check if we get a valid JSON-RPC response
                    try:
                        data = await response.json()
                        if isinstance(data, dict) and ("result" in data or "error" in data):
                            return True
                    except Exception:
                        # Even if parsing fails, 200 status means endpoint is up
                        pass
                    return True
                elif response.status == 202:
                    # 202 Accepted is also valid for async operations
                    return True
                else:
                    print(f"MCP endpoint returned status {response.status}", file=sys.stderr)
                    return False

    except asyncio.TimeoutError:
        print(f"MCP endpoint request timed out after {timeout} seconds", file=sys.stderr)
        return False
    except aiohttp.ClientError as e:
        print(f"MCP connection error: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error checking MCP: {e}", file=sys.stderr)
        return False


async def check_mcp_describe(port: int = 9903, timeout: int = 5) -> bool:
    """
    Test if MCP server can respond to describe endpoint (if available).
    This is a fallback check for older NAT versions.
    """
    try:
        url = f"http://localhost:{port}/mcp/describe"

        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Check that we got a valid MCP describe response
                    if isinstance(data, dict) and ("name" in data or "tools" in data):
                        return True
                    print("Invalid MCP describe response format", file=sys.stderr)
                    return False
                else:
                    # describe endpoint may not exist in newer NAT versions
                    return False

    except asyncio.TimeoutError:
        print(f"MCP describe request timed out after {timeout} seconds", file=sys.stderr)
        return False
    except aiohttp.ClientError as e:
        print(f"MCP connection error: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error checking MCP: {e}", file=sys.stderr)
        return False


async def main() -> int:
    """
    Main health check function.
    Returns 0 if healthy, 1 if unhealthy.
    """
    port = int(os.environ.get("MCP_PORT", "9903"))

    # Try primary /mcp/ endpoint first (NAT 1.3+)
    mcp_ok = await check_mcp_endpoint(port)
    if mcp_ok:
        return 0

    # Fallback to describe endpoint for older versions
    describe_ok = await check_mcp_describe(port)
    if describe_ok:
        return 0

    print("MCP server not responding properly", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Health check failed with error: {e}", file=sys.stderr)
        sys.exit(1)
