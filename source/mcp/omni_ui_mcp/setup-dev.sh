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

echo "========================================"
echo "OmniUI MCP Server - Development Setup"
echo "========================================"
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
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    
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

# Install dependencies
echo
echo "Installing dependencies..."
poetry install

# Create local directories if they don't exist
mkdir -p logs

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
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
echo "- Edit examples/local_config.yaml to customize configuration"
echo