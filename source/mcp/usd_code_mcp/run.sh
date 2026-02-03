#!/bin/bash
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

# Unix script to run USD Code MCP Server locally
# This script starts the server for local development
set -e  # Exit on any error

echo "========================================"
echo "USD Code MCP Server - Local Development"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run ./setup-dev.sh first to set up the development environment"
    exit 1
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "ERROR: Poetry not found"
    echo "Please install Poetry or run ./setup-dev.sh to set up the environment"
    exit 1
fi

# Set environment variables for development
export USD_CODE_MCP_DISABLE_USAGE_LOGGING=false
export MCP_PORT=9903

# Display startup information
echo "Starting USD Code MCP Server..."
echo "Config: workflow/config.yaml"
echo "Port: $MCP_PORT"
echo "Development mode: ENABLED"
echo
echo "Server will be available at: http://localhost:$MCP_PORT"
echo
echo "Press Ctrl+C to stop the server"
echo

# Function to handle cleanup on exit
cleanup() {
    echo
    echo "Server stopped."
}

# Set up cleanup on script exit
trap cleanup EXIT

# Start the server using Poetry
# Check if local_config.yaml exists, otherwise use config.yaml
if [ -f "workflow/local_config.yaml" ]; then
    poetry run usd-code-mcp workflow/local_config.yaml
else
    poetry run usd-code-mcp workflow/config.yaml
fi
