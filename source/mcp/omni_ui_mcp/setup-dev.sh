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

#!/bin/bash
# Unix development setup script for OmniUI MCP Server
# This script sets up the local development environment
set -e  # Exit on any error

readonly DIVIDER="========================================"

echo "$DIVIDER"
echo "OmniUI MCP Server - Development Setup"
echo "$DIVIDER"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.11+ and try again"
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "Python version:"
$PYTHON_CMD --version

# Check Python version is 3.11+
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.11"

if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "ERROR: Python $REQUIRED_VERSION or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo
    echo "Poetry not found. Installing Poetry..."
    echo
    curl --proto '=https' --tlsv1.2 -sSL https://install.python-poetry.org | $PYTHON_CMD -

    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    # Check if poetry is now available
    if ! command -v poetry &> /dev/null; then
        echo "ERROR: Failed to install Poetry or Poetry not in PATH"
        echo "Please install Poetry manually from https://python-poetry.org/docs/#installation"
        echo "Or add ~/.local/bin to your PATH and restart this script"
        exit 1
    fi

    echo
    echo "Poetry installed successfully!"
fi

echo "Poetry version:"
poetry --version

# Configure Poetry to create virtual environment in project directory
echo
echo "Configuring Poetry to use local virtual environment..."
poetry config virtualenvs.in-project true

# These MCP servers resolve every dependency from public PyPI, so Poetry never
# needs stored credentials. Poetry still queries the system keyring for each
# package URL, and on Windows sessions without a logon-session credential store
# that raises an uncaught "(1312, 'CredRead', ...)" error which aborts both
# ``poetry lock`` and ``poetry install``. Disable the keyring lookup for this
# script unless the caller has already chosen a setting.
export POETRY_KEYRING_ENABLED="${POETRY_KEYRING_ENABLED:-false}"

# If the shipped poetry.lock no longer matches pyproject.toml, ``poetry install``
# refuses with a "pyproject.toml changed significantly..." error. Detect that up
# front with ``poetry check --lock`` and refresh the lock only in that case, so a
# later install failure is reported as what it really is.
echo
echo "Checking that poetry.lock matches pyproject.toml..."
if ! poetry check --lock > /dev/null 2>&1; then
    echo "poetry.lock is out of date. Refreshing it with 'poetry lock'..."
    if ! poetry lock; then
        echo
        echo "ERROR: 'poetry lock' failed. See the output above for the cause."
        exit 1
    fi
fi

# Keep a copy of the install output so a failure can be classified below.
echo
echo "Installing dependencies..."
INSTALL_LOG="$(mktemp "${TMPDIR:-/tmp}/mcp-poetry-install.XXXXXX")"
poetry install 2>&1 | tee "$INSTALL_LOG"
INSTALL_STATUS=${PIPESTATUS[0]}
if [ "$INSTALL_STATUS" -ne 0 ]; then
    echo
    # Poetry treats only "true" or "1" (any case) as enabled; with the keyring
    # lookup disabled the failure cannot be a keyring error.
    case "$POETRY_KEYRING_ENABLED" in
        [Tt][Rr][Uu][Ee]|1) KEYRING_ON=1 ;;
        *) KEYRING_ON=0 ;;
    esac
    if [ "$KEYRING_ON" -eq 1 ] && grep -qE "CredRead|logon session|KeyringError|keyring\.(errors|backends)" "$INSTALL_LOG"; then
        echo "ERROR: poetry install failed while querying the system keyring for credentials."
        echo "These MCP servers only use public PyPI and need no credentials. Disable"
        echo "Poetry's keyring lookup and re-run this script:"
        echo "    export POETRY_KEYRING_ENABLED=false"
    else
        echo "ERROR: poetry install failed (exit code $INSTALL_STATUS). See the output above for the cause."
    fi
    echo "Full install output: $INSTALL_LOG"
    exit 1
fi
rm -f "$INSTALL_LOG"

# Create local directories if they don't exist
mkdir -p logs

echo
echo "$DIVIDER"
echo "Setup completed successfully!"
echo "$DIVIDER"
echo
echo "Next steps:"
echo "1. Run './run.sh' to start the MCP server"
echo "2. The server will be available at http://localhost:9901"
echo
echo "Optional: Run 'poetry run python validate-setup.py' to verify setup"
echo
echo "For development:"
echo "- Use 'poetry shell' to activate the virtual environment"
echo "- Use 'poetry run omni-ui-aiq' to run the server manually"
echo "- Edit workflow/local_config.yaml to customize configuration"
echo
