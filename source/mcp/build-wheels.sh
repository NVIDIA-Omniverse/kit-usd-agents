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
#   ./build-wheels.sh         # Build all wheels
#   ./build-wheels.sh kit     # Build only kit-mcp wheels
#   ./build-wheels.sh omni    # Build only omni-ui-mcp wheels
#   ./build-wheels.sh usd     # Build only usd-code-mcp wheels
#   ./build-wheels.sh isaac   # Build only isaacsim-mcp wheels
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

    # Check git LFS — without this, *_fns data/ files are pointer stubs and the
    # built wheel will be ~13x smaller than expected. Container will start, but
    # search tools fail at first invocation with "Extension data is not available".
    # Fail fast here so the
    # operator gets an actionable error at build time, not a silent runtime fault.
    check_git_lfs
}

# Probe a few canonical data files. Returns 0 if any are LFS pointer stubs
# (size < 1 KB AND first line matches the LFS pointer signature), non-zero
# otherwise.
_lfs_probe_has_pointers() {
    local pointer_marker='version https://git-lfs.github.com/spec/v1'
    # One sentinel per *_fns package — partial LFS resolution (e.g. sparse
    # checkout, interrupted pull) can leave one package's data stubbed while
    # another's is real. Probe all four so we don't ship a broken OmniUI or
    # USD-code wheel just because the Isaac/Kit probes happen to pass.
    local probe_files=(
        "$ROOT_DIR/source/aiq/isaacsim_fns/src/isaacsim_fns/data/6.0/extensions/extensions_database.json"
        "$ROOT_DIR/source/aiq/isaacsim_fns/src/isaacsim_fns/data/6.0/extensions/extensions_faiss/index.faiss"
        "$ROOT_DIR/source/aiq/kit_fns/src/kit_fns/data/110.0/knowledge/index.json"
        "$ROOT_DIR/source/aiq/omni_ui_fns/src/omni_ui_fns/data/faiss_index_omni_ui/index.faiss"
        "$ROOT_DIR/source/aiq/usd_code_fns/src/omni_aiq_usd_code/data/v25.11/code_rag/index.faiss"
    )
    LFS_POINTER_PATHS=()
    for probe in "${probe_files[@]}"; do
        if [ -f "$probe" ] && [ "$(stat -c%s "$probe" 2>/dev/null || stat -f%z "$probe" 2>/dev/null || echo 9999)" -lt 1024 ]; then
            if head -c 50 "$probe" 2>/dev/null | grep -q "$pointer_marker"; then
                LFS_POINTER_PATHS+=("$probe")
            fi
        fi
    done
    [ ${#LFS_POINTER_PATHS[@]} -gt 0 ]
}

# Detect LFS pointer files in the *_fns/data subtrees. If found, attempt
# auto-recovery via ``git lfs install --local && git lfs pull``. Only fail
# hard if that doesn't resolve them, since building wheels off pointer stubs
# produces a 13x smaller wheel that silently breaks at runtime.
check_git_lfs() {
    if ! command -v git &> /dev/null; then
        echo_warn "git is not installed; skipping LFS pointer check."
        # Explicit 0 — bare ``return`` would propagate the most recent command's
        # exit status, which under ``set -e`` could abort the script even
        # though this skip is intentional.
        return 0
    fi
    if ! command -v git-lfs &> /dev/null && ! git lfs version &> /dev/null; then
        echo_error "Git LFS is required but not installed."
        echo_info  "Install with one of:"
        echo_info  "  Ubuntu/Debian: sudo apt-get install git-lfs"
        echo_info  "  macOS (brew):  brew install git-lfs"
        echo_info  "Then re-run this script — LFS objects will be auto-pulled."
        exit 1
    fi

    # First pass: are pointer stubs present?
    if ! _lfs_probe_has_pointers; then
        echo_info "Git LFS check: data files resolved (no pointer stubs)."
        return
    fi

    # Pointers present. Decide whether we can auto-fix.
    local in_git_repo=0
    if (cd "$ROOT_DIR" && git rev-parse --git-dir &> /dev/null); then
        in_git_repo=1
    fi

    if [ "$in_git_repo" -eq 0 ]; then
        echo_error "LFS pointer stubs detected but $ROOT_DIR is not a git working tree."
        echo_error "Detected pointer files:"
        for p in "${LFS_POINTER_PATHS[@]}"; do echo_error "  $p"; done
        echo_error "Re-clone with git (not a tarball download), then re-run this script."
        exit 1
    fi

    # Auto-recovery path.
    echo_warn "Detected LFS pointer stubs in ${#LFS_POINTER_PATHS[@]} probed file(s):"
    for p in "${LFS_POINTER_PATHS[@]}"; do echo_warn "  $p"; done
    echo_info "Attempting auto-recovery (git lfs install --local && git lfs pull) ..."

    if ! (cd "$ROOT_DIR" && git lfs install --local); then
        echo_error "git lfs install --local failed (permissions? .git/config writeable?)."
        echo_info  "Run manually from the repo root and retry:"
        echo_info  "  git lfs install"
        echo_info  "  git lfs pull"
        exit 1
    fi

    if ! (cd "$ROOT_DIR" && git lfs pull); then
        echo_error "git lfs pull failed (network? LFS auth? remote not configured for LFS?)."
        echo_info  "Common causes:"
        echo_info  "  - No network access to the LFS server (corporate proxy / VPN)"
        echo_info  "  - 'origin' remote points at a fork that doesn't have LFS objects pushed"
        echo_info  "  - LFS auth not configured for the remote (try: git lfs env)"
        exit 1
    fi

    # Verify the recovery actually replaced the stubs.
    if _lfs_probe_has_pointers; then
        echo_error "git lfs pull ran but pointer stubs still present in:"
        for p in "${LFS_POINTER_PATHS[@]}"; do echo_error "  $p"; done
        echo_error "The LFS objects may not exist on the remote. Investigate with:"
        echo_info  "  git lfs ls-files --debug | head"
        echo_info  "  git lfs fetch --all"
        exit 1
    fi

    echo_info "Git LFS auto-recovery succeeded — data files now resolved."
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

# Build isaacsim-mcp wheels
build_isaacsim_mcp() {
    echo_info "=== Building Isaac Sim MCP wheels ==="

    # Build isaacsim_fns
    build_wheel "$ROOT_DIR/source/aiq/isaacsim_fns" "isaacsim_fns"

    # Build isaacsim_mcp (cleans dist/, so must be before copying isaacsim_fns)
    build_wheel "$SCRIPT_DIR/isaacsim_mcp" "isaacsim_mcp"

    # Copy isaacsim_fns to isaacsim_mcp dist (AFTER isaacsim_mcp build to avoid deletion)
    copy_wheel "$ROOT_DIR/source/aiq/isaacsim_fns" "$SCRIPT_DIR/isaacsim_mcp/dist" "isaacsim_fns"

    echo_info "Isaac Sim MCP wheels ready in: $SCRIPT_DIR/isaacsim_mcp/dist/"
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
        isaac)
            build_isaacsim_mcp
            ;;
        all)
            build_kit_mcp
            build_omni_ui_mcp
            build_usd_code_mcp
            build_isaacsim_mcp
            ;;
        *)
            echo "Usage: $0 [kit|omni|usd|isaac|all]"
            exit 1
            ;;
    esac

    echo ""
    echo_info "=== Build Complete ==="
    echo_info "You can now run: docker compose -f docker-compose.local.yaml up --build"
}

main "$@"
