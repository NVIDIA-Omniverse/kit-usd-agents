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

# Build Docker image for Isaac Sim MCP Server.
#
# Thin wrapper: delegates wheel construction to the shared
# ``../build-wheels.sh`` (which runs the Git LFS pointer-stub check and
# auto-recovery), then runs ``docker build`` in this dir.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_TAG="isaacsim-mcp:latest"

echo "Building Isaac Sim MCP Docker container..."

# Build the AIQ + MCP wheels via the shared script (LFS-aware).
bash "$SCRIPT_DIR/../build-wheels.sh" isaac

# Build Docker image
echo ""
echo "Building Docker image..."
cd "$SCRIPT_DIR"
docker build -t "$DOCKER_TAG" .

echo ""
echo "Docker build complete!"
echo ""
echo "To run the container:"
echo "  docker run --rm -p 9904:9904 -e NVIDIA_API_KEY=\$NVIDIA_API_KEY $DOCKER_TAG"
echo ""
echo "To run with custom port:"
echo "  docker run --rm -e MCP_PORT=8080 -e NVIDIA_API_KEY=\$NVIDIA_API_KEY -p 8080:8080 $DOCKER_TAG"
