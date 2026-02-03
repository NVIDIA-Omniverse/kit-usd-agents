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

# Build wheels for MCP servers
#
# This script builds the required wheel files for Docker image construction.
# Run this before 'docker compose -f docker-compose.local.yaml up --build'
#
# Prerequisites:
#   - Python 3.11+
#   - Poetry (https://python-poetry.org/docs/#installation)
#
# Usage:
#   ./build-wheels.sh        # Build all wheels
#   ./build-wheels.sh kit    # Build only kit-mcp wheels
#   ./build-wheels.sh omni   # Build only omni-ui-mcp wheels
#   ./build-wheels.sh usd    # Build only usd-code-mcp wheels
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo_error "Python 3 is required but not installed."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo_info "Python version: $PYTHON_VERSION"

    # Check Poetry
    if ! command -v poetry &> /dev/null; then
        echo_error "Poetry is required but not installed."
        echo_info "Install with: curl -sSL https://install.python-poetry.org | python3 -"
        exit 1
    fi

    POETRY_VERSION=$(poetry --version)
    echo_info "Poetry version: $POETRY_VERSION"
}

# Build a wheel package
build_wheel() {
    local package_dir=$1
    local package_name=$2

    echo_info "Building $package_name wheel..."
    cd "$package_dir"

    # Clean old builds
    rm -rf dist/ build/ *.egg-info

    # Build wheel
    poetry build

    echo_info "$package_name wheel built successfully"
    ls -la dist/*.whl
    cd - > /dev/null
}

# Copy wheel to destination
copy_wheel() {
    local src_dir=$1
    local dest_dir=$2
    local package_name=$3

    echo_info "Copying $package_name wheel to $dest_dir..."
    mkdir -p "$dest_dir"
    cp "$src_dir"/dist/*.whl "$dest_dir/"
}

# Build kit-mcp wheels
build_kit_mcp() {
    echo_info "=== Building Kit MCP wheels ==="

    # Build kit_fns
    build_wheel "$ROOT_DIR/source/aiq/kit_fns" "kit_fns"

    # Build kit_mcp (cleans dist/, so must be before copying kit_fns)
    build_wheel "$SCRIPT_DIR/kit_mcp" "kit_mcp"

    # Copy kit_fns to kit_mcp dist (AFTER kit_mcp build to avoid deletion)
    copy_wheel "$ROOT_DIR/source/aiq/kit_fns" "$SCRIPT_DIR/kit_mcp/dist" "kit_fns"

    echo_info "Kit MCP wheels ready in: $SCRIPT_DIR/kit_mcp/dist/"
}

# Build omni-ui-mcp wheels
build_omni_ui_mcp() {
    echo_info "=== Building Omni UI MCP wheels ==="

    # Build omni_ui_fns
    build_wheel "$ROOT_DIR/source/aiq/omni_ui_fns" "omni_ui_fns"

    # Build omni_ui_mcp (cleans dist/, so must be before copying omni_ui_fns)
    build_wheel "$SCRIPT_DIR/omni_ui_mcp" "omni_ui_mcp"

    # Copy omni_ui_fns to omni_ui_mcp dist (AFTER omni_ui_mcp build to avoid deletion)
    copy_wheel "$ROOT_DIR/source/aiq/omni_ui_fns" "$SCRIPT_DIR/omni_ui_mcp/dist" "omni_ui_fns"

    echo_info "Omni UI MCP wheels ready in: $SCRIPT_DIR/omni_ui_mcp/dist/"
}

# Build usd-code-mcp wheels
build_usd_code_mcp() {
    echo_info "=== Building USD Code MCP wheels ==="

    # Build usd_code_fns
    build_wheel "$ROOT_DIR/source/aiq/usd_code_fns" "usd_code_fns"

    # Build usd_code_mcp (cleans dist/, so must be before copying usd_code_fns)
    build_wheel "$SCRIPT_DIR/usd_code_mcp" "usd_code_mcp"

    # Copy usd_code_fns to usd_code_mcp dist (AFTER usd_code_mcp build to avoid deletion)
    copy_wheel "$ROOT_DIR/source/aiq/usd_code_fns" "$SCRIPT_DIR/usd_code_mcp/dist" "usd_code_fns"

    echo_info "USD Code MCP wheels ready in: $SCRIPT_DIR/usd_code_mcp/dist/"
}

# Main
main() {
    check_prerequisites

    case "${1:-all}" in
        kit)
            build_kit_mcp
            ;;
        omni)
            build_omni_ui_mcp
            ;;
        usd)
            build_usd_code_mcp
            ;;
        all)
            build_kit_mcp
            build_omni_ui_mcp
            build_usd_code_mcp
            ;;
        *)
            echo "Usage: $0 [kit|omni|usd|all]"
            exit 1
            ;;
    esac

    echo ""
    echo_info "=== Build Complete ==="
    echo_info "You can now run: docker compose -f docker-compose.local.yaml up --build"
}

main "$@"
