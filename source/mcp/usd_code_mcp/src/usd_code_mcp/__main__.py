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
Main entry point for the USD Code MCP server.
This allows the package to be run as a module with: python -m usd_code_mcp
"""

import os
import subprocess
import sys
from pathlib import Path

from . import __version__


def main():
    """Main entry point for the USD Code MCP server."""
    print(f"USD Code MCP v{__version__}")
    print("Starting MCP server with NAT (NeMo Agent Toolkit)...")

    # Check if a config file was provided as an argument
    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
        if not config_file.exists():
            print(f"ERROR: Specified config file does not exist: {config_file}")
            return 1
    else:
        # Find the config file in default locations
        config_paths = [
            Path("workflow/config.yaml"),
            Path(__file__).parent.parent / "workflow" / "config.yaml",
            Path("/app/workflow/config.yaml"),  # Docker path
        ]

        config_file = None
        for path in config_paths:
            if path.exists():
                config_file = path
                break

        if not config_file:
            print("ERROR: Could not find workflow/config.yaml")
            print("Searched in:")
            for path in config_paths:
                print(f"  - {path}")
            print("\nUsage: usd-code-mcp [config_file]")
            return 1

    print(f"Using config file: {config_file}")

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
    port = os.environ.get("MCP_PORT", "9903")
    cmd.extend(["--port", port])

    # Only expose function tools, not the react_agent workflow
    cmd.extend(
        [
            "--tool_names",
            "search_usd_code_examples",
            "--tool_names",
            "search_usd_knowledge",
            "--tool_names",
            "list_usd_modules",
            "--tool_names",
            "list_usd_classes",
            "--tool_names",
            "get_usd_module_detail",
            "--tool_names",
            "get_usd_class_detail",
            "--tool_names",
            "get_usd_method_detail",
        ]
    )

    print(f"Starting MCP server on port {port}...")

    try:
        # Use subprocess.run to execute the command and wait for it
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("ERROR: 'nat' command not found. Make sure nvidia-nat is installed.")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to start MCP server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
