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

# Build script for creating Docker image locally

# Clean the dist directory first
echo "Cleaning dist directory..."
rm -rf dist/

# Build the poetry package first
echo "Building poetry package..."
poetry build

# Build the Docker image
echo "Building Docker image..."
docker build -t usd-code-aiq:latest .

echo "Docker image built successfully!"
echo ""
echo "To run the MCP server container:"
echo "  docker run --rm -p 9901:9901 usd-code-aiq:latest"
echo ""
echo "To run with custom port:"
echo "  docker run --rm -p 8080:8080 -e MCP_PORT=8080 usd-code-aiq:latest"
echo ""
echo "To run in background (detached):"
echo "  docker run -d --name usd-code-aiq -p 9901:9901 usd-code-aiq:latest" 