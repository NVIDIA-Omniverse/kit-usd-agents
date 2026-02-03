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

# Build script for USD Code MCP Docker container
#
# This script builds both the AIQ functions and MCP server wheels,
# then builds the Docker image.
#
# Prerequisites:
#   - Python 3.11+
#   - Poetry
#   - Docker
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
AIQ_DIR="$ROOT_DIR/aiq/usd_code_fns"

echo "Building USD Code MCP Docker container..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/
rm -rf build/
mkdir -p dist/

# Build the AIQ functions wheel first
echo "Building usd_code_aiq wheel..."
if [ -d "$AIQ_DIR" ]; then
    cd "$AIQ_DIR"
    poetry build
    cp dist/*.whl "$SCRIPT_DIR/dist/"
    cd "$SCRIPT_DIR"
    echo "usd_code_aiq wheel copied to dist/"
else
    echo "ERROR: AIQ directory not found at $AIQ_DIR"
    echo "Please ensure the usd_code_fns package exists."
    exit 1
fi

# Build the MCP package
echo "Building usd_code_mcp wheel..."
poetry build

# Check if both wheels exist
if ! ls dist/usd_code_aiq-*.whl 1>/dev/null 2>&1; then
    echo "ERROR: usd_code_aiq wheel not found in dist/"
    exit 1
fi

if ! ls dist/usd_code_mcp-*.whl 1>/dev/null 2>&1; then
    echo "ERROR: usd_code_mcp wheel not found in dist/"
    exit 1
fi

echo "Wheels ready:"
ls -la dist/*.whl

# Build Docker image
echo ""
echo "Building Docker image..."
DOCKER_TAG="usd-code-mcp:latest"
docker build -t "$DOCKER_TAG" .

echo ""
echo "Docker build complete!"
echo ""
echo "To run the container:"
echo "  docker run --rm -p 9903:9903 -e NVIDIA_API_KEY=\$NVIDIA_API_KEY $DOCKER_TAG"
echo ""
echo "To run with custom port:"
echo "  docker run --rm -e MCP_PORT=8080 -e NVIDIA_API_KEY=\$NVIDIA_API_KEY -p 8080:8080 $DOCKER_TAG"
