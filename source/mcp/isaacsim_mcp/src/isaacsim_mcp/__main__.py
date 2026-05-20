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
Main entry point for the Isaac Sim MCP package.
This allows the package to be run as a module with: python -m isaacsim_mcp
"""

import os
import subprocess
import sys
from pathlib import Path

from . import _get_version


def _print_backend_config_banner(uses_reranker: bool = True) -> None:
    """Echo the resolved embedder/reranker backend config at startup.

    Background: the LocalEmbedder/LocalReranker are constructed lazily on
    the first MCP tool call, so the factory's own "Creating embedder..." and
    "Using local embedder at..." log lines don't appear in ``docker logs``
    until a tool fires. That left QA unable to tell whether the .env was
    forwarded correctly without first invoking a tool from Cursor. This banner
    prints the resolved env-var snapshot eagerly so operators can verify
    backend selection immediately after ``docker compose up``.
    """
    embedder_backend = os.environ.get("KIT_EMBEDDER_BACKEND", "nvidia_api")
    embedder_url = os.environ.get("KIT_LOCAL_EMBEDDER_URL", "")
    reranker_backend = os.environ.get("KIT_RERANKER_BACKEND", "nvidia_api")
    reranker_url = os.environ.get("KIT_LOCAL_RERANKER_URL", "")

    sep = "=" * 74
    print(sep)
    print(f"[mcp-startup] Embedder backend: {embedder_backend}")
    if embedder_backend == "local" and embedder_url:
        print(f"[mcp-startup] Using local embedder at {embedder_url}")
    if uses_reranker:
        print(f"[mcp-startup] Reranker backend: {reranker_backend}")
        if reranker_backend == "local" and reranker_url:
            print(f"[mcp-startup] Using local reranker at {reranker_url}")
    else:
        print("[mcp-startup] NOTE: isaacsim-mcp uses the embedder only — the reranker")
        print("[mcp-startup]       env vars are accepted but not exercised by current tools.")
    print(sep)


def main():
    """Main entry point for the Isaac Sim MCP package - starts the MCP server."""
    print(f"Isaac Sim MCP Tools v{_get_version()}")
    print("Starting MCP server with NAT (NeMo Agent Toolkit)...")

    # Check if a config file was provided as an argument
    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
        if not config_file.exists():
            print(f"ERROR: Specified config file does not exist: {config_file}")
            return 1
    else:
        # Find the config file in default locations
        # Get the package root directory (where pyproject.toml is located)
        package_root = Path(__file__).parent.parent.parent.parent

        config_paths = [
            Path("workflows/local_config.yaml"),  # Current directory - local development
            Path("workflows/config.yaml"),  # Current directory - production
            package_root / "workflows" / "local_config.yaml",  # Package root - local development
            package_root / "workflows" / "config.yaml",  # Package root - production
            Path(__file__).parent.parent.parent / "workflows" / "local_config.yaml",  # Relative to src
            Path(__file__).parent.parent.parent / "workflows" / "config.yaml",  # Relative to src
            Path("/app/workflows/config.yaml"),  # Docker path
        ]

        config_file = None
        for path in config_paths:
            if path.exists():
                config_file = path
                break

        if not config_file:
            print("ERROR: Could not find config file")
            print("Searched in:")
            for path in config_paths:
                print(f"  - {path}")
            print("\nUsage: isaacsim-mcp [config_file]")
            print("\nFor local development, ensure workflows/local_config.yaml exists")
            return 1

    print(f"Using config file: {config_file}")

    # Detect development mode based on config file name
    is_dev_mode = "local_config" in str(config_file)
    if is_dev_mode:
        print("Running in DEVELOPMENT mode")

    # Run the MCP server using NAT mcp serve command
    # Use 'nat mcp serve' which enables the MCP protocol frontend
    cmd = ["nat", "mcp", "serve", "--config_file", str(config_file)]

    # Host binding configuration:
    # - For production/security: use 127.0.0.1 (localhost only)
    # - For Docker/development: use 0.0.0.0 (all interfaces)
    # Configurable via MCP_HOST environment variable
    # Default to localhost for security; set MCP_HOST=0.0.0.0 for Docker
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    cmd.extend(["--host", host])

    # Check for PORT environment variable
    port = os.environ.get("MCP_PORT", "9904")
    cmd.extend(["--port", port])

    # Only expose function tools, not the react_agent workflow
    cmd.extend(
        [
            "--tool_names",
            "get_isaac_sim_instructions",
            "--tool_names",
            "search_isaac_sim_extensions",
            "--tool_names",
            "get_isaac_sim_extension_details",
            "--tool_names",
            "search_isaac_sim_code_examples",
            "--tool_names",
            "search_isaac_sim_settings",
        ]
    )

    print(f"Starting MCP server on port {port}...")
    _print_backend_config_banner(uses_reranker=False)
    if is_dev_mode:
        print("Development server will use verbose logging and localhost binding")

    # Show the full command for debugging in dev mode
    if is_dev_mode:
        print(f"Command: {' '.join(cmd)}")

    try:
        # Use subprocess.run to execute the command and wait for it
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("ERROR: 'nat' command not found. Make sure nvidia-nat is installed.")
        print("Run the setup script first: setup-dev.bat (Windows) or ./setup-dev.sh (Unix)")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to start MCP server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
