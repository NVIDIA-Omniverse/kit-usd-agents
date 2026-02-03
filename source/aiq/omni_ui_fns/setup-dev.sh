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

# Development setup script for OmniUI AIQ Functions
# This script sets up the local development environment

# Development setup script for OmniUI AIQ Functions
# This script sets up the local development environment
set -e  # Exit on any error

echo "========================================"
echo "OmniUI AIQ Functions - Development Setup"
echo "========================================"
echo

# Check for Python 3.12 or 3.11 specifically
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "Found Python 3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "Found Python 3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "ERROR: Python 3.11 or 3.12 is not installed or not in PATH"
    echo "Please install Python 3.11 or 3.12 and try again"
    echo "Visit: https://www.python.org/downloads/"
    exit 1
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

# Remove any existing virtual environment
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Set Poetry to use the correct Python version
echo
echo "Setting Poetry to use $PYTHON_CMD..."
poetry env use $PYTHON_CMD

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
echo "1. Use 'poetry shell' to activate the virtual environment"
echo "2. Functions are now available for import in other projects"
echo
echo "For development:"
echo "- Use 'poetry shell' to activate the virtual environment"
echo "- Use 'poetry add <package>' to add new dependencies"
echo