================================================================================
OMNI_UI_MCP ROOT FILES CONCATENATION
================================================================================
SUMMARY STATISTICS:
----------------------------------------
Total files processed: 13
Total size: 32.44 KB
Total tokens: 8,008

File-by-file breakdown:
  - README.md                        13.08 KB      3,079 tokens
  - pyproject.toml                    3.13 KB      1,017 tokens
  - check_mcp_health.py               4.19 KB        876 tokens
  - setup-dev.sh                      2.78 KB        687 tokens
  - setup-dev.bat                     2.39 KB        534 tokens
  - catalog-info.yaml                 1.53 KB        474 tokens
  - run-local.bat                     1.52 KB        353 tokens
  - run-local.sh                      1.28 KB        314 tokens
  - build-docker.bat                 868.00 B        228 tokens
  - build-docker.sh                  738.00 B        203 tokens
  - Dockerfile                       641.00 B        163 tokens
  - start-server.bat                 341.00 B         74 tokens
  - VERSION.md                         7.00 B          6 tokens

================================================================================
================================================================================

DIRECTORY STRUCTURE:
----------------------------------------

📄 .dockerignore
📄 .gitattributes
📄 .gitignore
📄 Dockerfile
📄 README.md
📄 VERSION.md
📄 build-docker.bat
📄 build-docker.sh
📄 catalog-info.yaml
📄 check_mcp_health.py
📁 dist/
📁 docs/
📁 examples/
📄 poetry.lock
📄 pyproject.toml
📄 run-local.bat
📄 run-local.sh
📄 setup-dev.bat
📄 setup-dev.sh
📁 src/
📄 start-server.bat
📁 tests/
📁 utils/
📁 workflow/

================================================================================


================================================================================
FILE: Dockerfile
Size: 641.00 B | Tokens: 163
================================================================================

FROM python:3.11-slim

# Copy the built distribution files from the Poetry build
COPY dist/*.whl /app/
# Copy the workflow folder recursively from the source directory
COPY workflow/ /app/workflow/
# Copy health check script
COPY check_mcp_health.py /app/

WORKDIR /app

# Install the package from the wheel file
RUN pip install --no-cache-dir *.whl

# Set environment variables
ENV PYTHONPATH=/app
ENV OMNI_UI_DISABLE_USAGE_LOGGING=false
ENV MCP_PORT=9901

EXPOSE 9901
# Create Docker environment marker
RUN touch /.dockerenv

# Set the command to run the application
# Using the console script defined in pyproject.toml
CMD ["omni-ui-aiq"]

================================================================================
FILE: README.md
Size: 13.08 KB | Tokens: 3,079
================================================================================

# OmniUI MCP Server

A Model Context Protocol (MCP) server implementation providing comprehensive OmniUI API information and development tools, built using the AIQ toolkit following the same patterns as the USD Code MCP server.

## Overview

This MCP server provides intelligent assistance for Omniverse UI developers by offering complete access to the OmniUI Atlas database, which contains detailed information about all OmniUI classes, modules, and methods. The server exposes 7 specialized tools for comprehensive OmniUI development support.

## Features

### Core API Functions

- **list_ui_classes**: Returns all OmniUI class names from the Atlas database
- **list_ui_modules**: Lists all OmniUI modules with extension information
- **get_ui_class_detail**: Provides detailed information about specific classes including methods and documentation
- **get_ui_module_detail**: Shows comprehensive module information with associated classes and functions
- **get_ui_method_detail**: Returns detailed method signatures, parameters, and documentation
- **get_omni_ui_code_example**: Retrieves relevant OmniUI code examples using semantic search with reranking
- **get_instructions**: Retrieves OmniUI system instructions and documentation for code generation

### Additional Features

- **Atlas Database**: Complete OmniUI API coverage with class hierarchy and documentation
- **Code Examples**: Semantic search through OmniUI code examples with FAISS indexing
- **System Instructions**: Bundled documentation for omni.ui framework, widgets, scene system, and styling
- **Fuzzy Matching**: Intelligent name matching for partial queries
- **AIQ Integration**: Built using the AIQ toolkit for robust MCP server functionality
- **Docker Support**: Complete containerization setup for easy deployment
- **Usage Logging**: Built-in analytics and usage tracking
- **Error Handling**: Comprehensive error handling and logging
- **Reranking Support**: NVIDIA AI reranking for improved search results

## Quick Start

### TL;DR - Local Development

**Windows:**
```cmd
setup-dev.bat    # Run once
run-local.bat    # Run to start server
```

**Unix/Linux/macOS:**
```bash
./setup-dev.sh  # Run once
./run-local.sh  # Run to start server
```

The server will be available at `http://localhost:9901`

### Prerequisites

- Python 3.11+
- Poetry (for development)
- Docker (for containerized deployment)

### Docker Deployment (Recommended)

1. Build the Docker image:
```bash
./build-docker.sh  # Linux/macOS
# or
build-docker.bat   # Windows
```

2. Run the container:
```bash
docker run --rm -p 9901:9901 omni-ui-mcp:latest
```

The server will start on port 9901.

### Local Development (Recommended)

The easiest way to set up local development is using the provided setup scripts:

#### Quick Setup

**Windows:**
```cmd
setup-dev.bat
run-local.bat
```

**Unix/Linux/macOS:**
```bash
./setup-dev.sh
./run-local.sh
```

#### What the setup does:
- Checks for Python 3.11+ and Poetry installation
- Installs Poetry if not available
- Configures Poetry to use local virtual environment
- Installs all project dependencies
- Creates necessary directories

#### Manual Setup (Advanced)

If you prefer manual setup:

1. Install dependencies:
```bash
poetry install
```

2. Run the server with local config:
```bash
poetry run omni-ui-aiq examples/local_config.yaml
```

#### Development Features

The local development setup includes:
- **Auto-detection**: Uses `examples/local_config.yaml` for development-specific settings
- **Verbose logging**: Enabled by default in local config for debugging
- **Localhost binding**: Server binds to 127.0.0.1 for local development
- **Development mode**: Enhanced error messages and debugging information

#### Virtual Environment

The setup scripts configure Poetry to create the virtual environment in the project directory (`.venv/`). This makes it easy to:
- Activate manually: `poetry shell`
- Run commands: `poetry run <command>`
- Install new dependencies: `poetry add <package>`

## Configuration

The server uses `examples/config.yaml` for configuration:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 16384

functions:
  get_omni_ui_code_example:
    _type: omni_aiq_omni_ui/get_omni_ui_code_example
    verbose: false
    enable_rerank: true
    rerank_k: 10
  
  list_ui_classes:
    _type: omni_aiq_omni_ui/list_ui_classes
    verbose: false
  
  list_ui_modules:
    _type: omni_aiq_omni_ui/list_ui_modules
    verbose: false
  
  get_ui_class_detail:
    _type: omni_aiq_omni_ui/get_ui_class_detail
    verbose: false
  
  get_ui_module_detail:
    _type: omni_aiq_omni_ui/get_ui_module_detail
    verbose: false
  
  get_ui_method_detail:
    _type: omni_aiq_omni_ui/get_ui_method_detail
    verbose: false
  
  get_instructions:
    _type: omni_aiq_omni_ui/get_instructions
    verbose: false

workflow:
  _type: react_agent
  llm_name: nim_llm
  tool_names:
    - get_omni_ui_code_example
    - list_ui_classes
    - list_ui_modules
    - get_ui_class_detail
    - get_ui_module_detail
    - get_ui_method_detail
    - get_instructions
```

## Available Tools

### list_ui_classes

Returns all OmniUI class names from the Atlas database.

**Parameters**: None

**Returns**: 
```json
{
  "class_full_names": ["FilterButton", "OptionsMenu", "RadioModel", ...],
  "total_count": 150,
  "description": "OmniUI classes from Atlas data"
}
```

### list_ui_modules

Lists all OmniUI modules with extension information.

**Parameters**: None

**Returns**:
```json
{
  "module_names": ["omni.kit.widget.filter", "omni.kit.widget.options_menu", ...],
  "total_count": 50,
  "description": "OmniUI modules from Atlas data"
}
```

### get_ui_class_detail

Provides detailed information about a specific class.

**Parameters**: 
- `class_name` (string): Name of the class (supports partial matching)

**Returns**: Detailed class information including methods, parent classes, and documentation

### get_ui_module_detail

Shows comprehensive module information.

**Parameters**:
- `module_name` (string): Name of the module (supports partial matching)

**Returns**: Module details including classes, functions, and extension information

### get_ui_method_detail

Returns detailed method information.

**Parameters**:
- `method_name` (string): Name of the method (supports partial matching)

**Returns**: Method signature, parameters, return type, and documentation

### get_omni_ui_code_example

Retrieves relevant OmniUI code examples using semantic search with optional reranking.

**Parameters**:
- `query` (string): Natural language search query describing what you're looking for
- `top_k` (integer, optional): Number of results to return (default: 90)
- `length` (integer, optional): Maximum cumulative length of examples (default: 30000)

**Returns**: Formatted markdown with relevant code examples and explanations

**Example**:
```
get_omni_ui_code_example query="create a button with a callback function"
get_omni_ui_code_example query="how to use sliders" top_k=5
```

### get_instructions

Retrieves OmniUI system instructions and documentation for code generation.

**Parameters**:
- `name` (string, optional): Specific instruction set to retrieve. If not provided, lists all available instructions.

**Available Instruction Sets**:
- `agent_system`: Core system prompt with omni.ui framework basics
- `classes`: Comprehensive class API reference and model patterns
- `omni_ui_scene_system`: Complete 3D UI system documentation
- `omni_ui_system`: Core widgets, containers, layouts and styling

**Returns**: Formatted documentation with metadata, descriptions, and use cases

**Examples**:
```
get_instructions                          # Lists all available instructions
get_instructions name="agent_system"      # Get core system prompt
get_instructions name="omni_ui_system"    # Get widgets and layouts documentation
get_instructions name="omni_ui_scene_system"  # Get 3D UI documentation
```

## Usage Examples

```
# Get all available classes
list_ui_classes

# Search for specific modules
list_ui_modules

# Get detailed information about a class
get_ui_class_detail class_name="FilterButton"

# Find information about a module
get_ui_module_detail module_name="filter"

# Look up method details
get_ui_method_detail method_name="create_button"

# Search for code examples
get_omni_ui_code_example query="create a window with buttons"

# Load system instructions for better code generation
get_instructions name="omni_ui_system"
```

## Architecture

The project follows the AIQ toolkit patterns:

```
src/omni_aiq_omni_ui/
├── __init__.py                          # Package initialization with AIQ patching
├── __main__.py                          # Entry point for running the server
├── config.py                            # Configuration constants
├── data/
│   ├── instructions/                    # System instruction documents
│   │   ├── agent_system.md
│   │   ├── classes.md
│   │   ├── omni_ui_scene_system.md
│   │   └── omni_ui_system.md
│   ├── faiss_index_omni_ui/            # FAISS index for code examples
│   ├── omni_ui_rag_collection.json     # RAG collection data
│   └── ui_atlas.json                   # Atlas database
├── functions/
│   ├── __init__.py
│   ├── list_ui_classes.py                  # Returns all class names
│   ├── list_ui_modules.py                  # Lists all modules
│   ├── get_ui_class_detail.py             # Detailed class information
│   ├── get_ui_module_detail.py            # Module details
│   ├── get_ui_method_detail.py            # Method signatures and docs
│   ├── get_omni_ui_code_example.py     # Semantic code search
│   └── get_instructions.py             # System instructions
├── register_*.py                        # AIQ function registrations
├── services/
│   ├── __init__.py
│   ├── omni_ui_atlas.py               # Atlas database service
│   ├── retrieval.py                   # FAISS retrieval service
│   └── reranking.py                   # Reranking service
├── utils/
│   ├── __init__.py
│   ├── fuzzy_matching.py              # Fuzzy string matching
│   ├── usage_logging.py               # Usage logging utilities
│   └── usage_logging_decorator.py     # Logging decorators
└── models/
    └── __init__.py
```

## Environment Variables

- `MCP_PORT`: Server port (default: 9901)
- `OMNI_UI_DISABLE_USAGE_LOGGING`: Disable usage analytics (default: false)

## Usage Analytics

The server includes built-in usage analytics that log:
- Tool calls and parameters
- Success/failure status
- Execution times
- Error messages

Analytics can be disabled by setting `OMNI_UI_DISABLE_USAGE_LOGGING=true`.

## Integration with Cursor IDE

Add to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "omni-ui-mcp": {
      "type": "mcp",
      "url": "http://localhost:9901/sse"
    }
  }
}
```

## Troubleshooting

### Common Issues

**"Poetry not found" error:**
- Make sure Poetry is installed: https://python-poetry.org/docs/#installation
- On Unix systems, you may need to restart your terminal or add `~/.local/bin` to your PATH
- Run the setup script again after installing Poetry

**"aiq command not found" error:**
- Make sure all dependencies are installed: `poetry install`
- Activate the virtual environment: `poetry shell`
- Check that aiqtoolkit is properly installed: `poetry show aiqtoolkit`

**Port already in use:**
- Check if another service is using port 9901: `netstat -an | grep 9901`
- Set a different port: `set MCP_PORT=9902` (Windows) or `export MCP_PORT=9902` (Unix)
- Or modify the port in `examples/local_config.yaml`

**Virtual environment issues:**
- Delete `.venv` directory and run setup script again
- Make sure Poetry is configured correctly: `poetry config --list`

### Development Tips

- Use `poetry shell` to activate the virtual environment for manual testing
- Check logs in the `logs/` directory if created
- Modify `examples/local_config.yaml` to adjust development settings
- Use `poetry show` to list installed dependencies
- Use `poetry add --group dev <package>` to add development dependencies

## Development

### Adding New Tools

When adding new tools to the MCP server, follow this checklist:

1. **Create function implementation** in `src/omni_aiq_omni_ui/functions/`
   - Implement async function with proper error handling
   - Include logging and documentation

2. **Create registration wrapper** `register_[tool_name].py`
   - Define Pydantic input schema
   - Write comprehensive tool description
   - Include usage logging

3. **Update pyproject.toml**
   - Add entry point in `[tool.poetry.plugins."aiq.components"]`
   - Increment version number

4. **Update configuration** in `examples/config.yaml`
   - Add function definition in `functions:` section
   - Add tool to `workflow.tool_names` list

5. **Update package version** in `src/omni_aiq_omni_ui/__init__.py`

6. **Reinstall package**: `poetry install`

7. **Update documentation** in README.md

### Testing

```bash
poetry run pytest
```

### Building

```bash
poetry build
```

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE file in the repository root for details.

================================================================================
FILE: VERSION.md
Size: 7.00 B | Tokens: 6
================================================================================

0.10.0


================================================================================
FILE: build-docker.bat
Size: 868.00 B | Tokens: 228
================================================================================

@echo off
REM Build script for OmniUI MCP Docker container

echo Building OmniUI MCP Docker container...

REM Clean previous builds
echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM Build the package using Poetry
echo Building Python package...
poetry build

REM Check if wheel was created
if not exist "dist\*.whl" (
    echo ERROR: No wheel file found in dist/
    exit /b 1
)

REM Build Docker image
echo Building Docker image...
set DOCKER_TAG=omni-ui-mcp:latest
docker build -t %DOCKER_TAG% .

if %ERRORLEVEL% neq 0 (
    echo ERROR: Docker build failed
    exit /b 1
)

echo Docker build complete!
echo To run the container:
echo   docker run --rm -p 9901:9901 %DOCKER_TAG%
echo.
echo To run with custom port:
echo   docker run --rm -e MCP_PORT=8080 -p 8080:8080 %DOCKER_TAG%

================================================================================
FILE: build-docker.sh
Size: 738.00 B | Tokens: 203
================================================================================

#!/bin/bash

# Build script for OmniUI MCP Docker container

set -e

echo "Building OmniUI MCP Docker container..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/
rm -rf build/

# Build the package using Poetry
echo "Building Python package..."
poetry build

# Check if wheel was created
if [ ! -f dist/*.whl ]; then
    echo "ERROR: No wheel file found in dist/"
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
DOCKER_TAG="omni-ui-mcp:latest"
docker build -t "$DOCKER_TAG" .

echo "Docker build complete!"
echo "To run the container:"
echo "  docker run --rm -p 9901:9901 $DOCKER_TAG"
echo ""
echo "To run with custom port:"
echo "  docker run --rm -e MCP_PORT=8080 -p 8080:8080 $DOCKER_TAG"

================================================================================
FILE: catalog-info.yaml
Size: 1.53 KB | Tokens: 474
================================================================================

apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: omni-ui-mcp
  description: extract the teamplte for an exiting project
  annotations:
    gitlab.com/project-slug: omniverse/gen-ai/kit-lc-agent
    backstage.io/techdocs-ref: dir:.
    backstage.io/kubernetes-id: omni-ui-mcp
    backstage.io/kubernetes-label-selector: app.kubernetes.io/instance=omni-ui-mcp
    backstage.io/source-template: default:template/kaizen-java-microservice-nvb-template
    nvb.nvidia.com/project-expiration-timestamp: '0'
  links:
    - url: >-
        https://ov.data.nvidiagrid.net/app/discover?sg_tenant=Global#/?_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:now-15h,to:now))&_a=(columns:!(),filters:!(),index:omniverse-devplat.logs.omni-devinfra-svc,interval:auto,query:(language:kuery,query:%22omni-ui-mcp%22),sort:!(!(time,desc)))
      title: Logs on Kratos Kibana Dashboard
    - url: >-
        https://omni-devinfra-prometheus-nvks-blue.service.odp.nvidia.com/graph?g0.expr=jvm_memory_usage_bytes%7Bkubernetes_namespace%3D%22omni-ui-mcp%22%7D&g0.tab=1&g0.display_mode=lines&g0.show_exemplars=0&g0.range_input=1h
      title: Metrics on Prometheus
    - url: >-
        https://app.lightstep.com/nvidia-staging/service-directory/omni-ui-mcp/deployments?time_window=minutes_60
      title: Trace on Lightstep
    - url: https://omni-ui-mcp.service.odp.nvidia.com/health
      title: Health endpoint
  tags:
    - usd-agent
    - kit-mcp
spec:
  type: service
  lifecycle: production
  owner: group:default/dfagnou-org
  pic: user:default/dfagnou


================================================================================
FILE: check_mcp_health.py
Size: 4.19 KB | Tokens: 876
================================================================================

#!/usr/bin/env python3
"""
Simple MCP health check script that tests actual MCP functionality.
This script attempts to connect to the MCP server and perform a basic operation.
Exit code 0 = healthy, non-zero = unhealthy.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Optional

import aiohttp

# Suppress unnecessary logging
logging.basicConfig(level=logging.ERROR)


async def check_mcp_describe(port: int = 9901, timeout: int = 5) -> bool:
    """
    Test if MCP server can respond to describe endpoint.
    This is a basic MCP operation that should always work if the server is healthy.
    """
    try:
        url = f"http://localhost:{port}/mcp/describe"

        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Check that we got a valid MCP describe response
                    if isinstance(data, dict) and ("name" in data or "tools" in data):
                        return True
                    print(f"Invalid MCP describe response format", file=sys.stderr)
                    return False
                else:
                    print(f"MCP describe returned status {response.status}", file=sys.stderr)
                    return False

    except asyncio.TimeoutError:
        print(f"MCP describe request timed out after {timeout} seconds", file=sys.stderr)
        return False
    except aiohttp.ClientError as e:
        print(f"MCP connection error: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error checking MCP: {e}", file=sys.stderr)
        return False


async def check_mcp_sse_connection(port: int = 9901, timeout: int = 5) -> bool:
    """
    Test if MCP SSE endpoint is accessible.
    SSE connections are what MCP clients actually use.
    """
    try:
        url = f"http://localhost:{port}/sse"

        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            # Try to establish SSE connection
            async with session.get(url, headers={"Accept": "text/event-stream"}) as response:
                if response.status == 200:
                    # Read just the first bit to confirm SSE is working
                    # We don't need to fully establish the protocol
                    chunk = await response.content.read(1)
                    if chunk:
                        return True
                    return True  # Even empty response is OK if status is 200
                else:
                    print(f"MCP SSE endpoint returned status {response.status}", file=sys.stderr)
                    return False

    except asyncio.TimeoutError:
        print(f"MCP SSE connection timed out after {timeout} seconds", file=sys.stderr)
        return False
    except aiohttp.ClientError as e:
        print(f"MCP SSE connection error: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error checking MCP SSE: {e}", file=sys.stderr)
        return False


async def main() -> int:
    """
    Main health check function.
    Returns 0 if healthy, 1 if unhealthy.
    """
    port = int(os.environ.get("MCP_PORT", "9901"))

    # Check both describe endpoint and SSE connection
    describe_ok = await check_mcp_describe(port)
    sse_ok = await check_mcp_sse_connection(port)

    if describe_ok and sse_ok:
        # MCP is healthy
        return 0
    else:
        # MCP is unhealthy
        if not describe_ok:
            print(f"MCP describe endpoint not responding properly", file=sys.stderr)
        if not sse_ok:
            print(f"MCP SSE endpoint not responding properly", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Health check failed with error: {e}", file=sys.stderr)
        sys.exit(1)


================================================================================
FILE: pyproject.toml
Size: 3.13 KB | Tokens: 1,017
================================================================================

[tool.poetry]
name = "omni-ui-aiq"
version = "0.10.0"
description = "OmniUI tools for Nemo Agent Toolkit - providing OmniUI class information and documentation"
authors = ["Omniverse GenAI Team <doyopk-org@exchange.nvidia.com>"]
readme = "README.md"
homepage = "https://github.com/NVIDIA-Omniverse/kit-usd-agents"
repository = "https://github.com/NVIDIA-Omniverse/kit-usd-agents"
documentation = "https://github.com/NVIDIA-Omniverse/kit-usd-agents"
keywords = ["omniui", "ai", "tools", "code-generation"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Multimedia :: Graphics :: 3D Modeling",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
packages = [{include = "omni_aiq_omni_ui", from = "src"}]
include = [
    "src/omni_aiq_omni_ui/data/**/*",
    "workflow/config.yaml",
]

[tool.poetry.dependencies]
python = ">=3.11,<3.13"
aiqtoolkit = "1.1.0"
aiqtoolkit-langchain = "1.1.0"
# LangChain dependencies
langchain-community = ">=0.0.10"
langchain-core = ">=0.1.0"
langchain-nvidia-ai-endpoints = ">=0.0.4"
# Vector search and HTTP dependencies
faiss-cpu = ">=1.7.0"
requests = ">=2.25.0"
httpx = ">=0.24.0"
# Configuration support
pyyaml = ">=6.0"
aiohttp-sse = "^2.2.0"
# Telemetry support
redis = {extras = ["hiredis"], version = ">=4.5.0"}

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
pytest-cov = "^4.0.0"
pytest-asyncio = "^0.21.0"
black = "^23.0.0"
flake8 = "^6.0.0"
mypy = "^1.0.0"

# Entry points for OmniUI functions
[tool.poetry.plugins."aiq.components"]
omni_aiq_omni_ui_search_ui_code_examples = "omni_aiq_omni_ui.register_search_ui_code_examples"
omni_aiq_omni_ui_search_ui_window_examples = "omni_aiq_omni_ui.register_search_ui_window_examples"
omni_aiq_omni_ui_get_classes = "omni_aiq_omni_ui.register_get_classes"
omni_aiq_omni_ui_get_modules = "omni_aiq_omni_ui.register_get_modules"
omni_aiq_omni_ui_get_class_detail = "omni_aiq_omni_ui.register_get_class_detail"
omni_aiq_omni_ui_get_module_detail = "omni_aiq_omni_ui.register_get_module_detail"
omni_aiq_omni_ui_get_method_detail = "omni_aiq_omni_ui.register_get_method_detail"
omni_aiq_omni_ui_get_instructions = "omni_aiq_omni_ui.register_get_instructions"
omni_aiq_omni_ui_get_class_instructions = "omni_aiq_omni_ui.register_get_class_instructions"
omni_aiq_omni_ui_get_style_docs = "omni_aiq_omni_ui.register_get_style_docs"

[tool.poetry.scripts]
omni-ui-aiq = "omni_aiq_omni_ui.__main__:main"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 120
target-version = ["py311"]
include = '\.pyi?$'

[tool.flake8]
max-line-length = 120
ignore = ["E501", "W503"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=omni_aiq_omni_ui --cov-report=term-missing"

================================================================================
FILE: run-local.bat
Size: 1.52 KB | Tokens: 353
================================================================================

@echo off
REM Windows script to run OmniUI MCP Server locally
REM This script starts the server for local development

echo ========================================
echo OmniUI MCP Server - Local Development
echo ========================================
echo.

REM Check if Poetry environment exists
poetry env info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Poetry environment not found
    echo Please run setup-dev.bat first to set up the development environment
    pause
    exit /b 1
)

REM Check if Poetry is available
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Poetry not found
    echo Please install Poetry or run setup-dev.bat to set up the environment
    pause
    exit /b 1
)

REM Set environment variables for development
set OMNI_UI_DISABLE_USAGE_LOGGING=false
set MCP_PORT=9901

REM Display startup information
echo Starting OmniUI MCP Server...
echo Config: examples/local_config.yaml
echo Port: %MCP_PORT%
echo Development mode: ENABLED
echo.
echo Server will be available at: http://localhost:%MCP_PORT%
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server using Poetry
REM First check if local_config.yaml exists, otherwise use config.yaml
if exist "examples\local_config.yaml" (
    poetry run omni-ui-aiq examples/local_config.yaml
) else (
    echo Note: Using config.yaml as local_config.yaml not found
    poetry run omni-ui-aiq examples/config.yaml
)

REM If we get here, the server has stopped
echo.
echo Server stopped.
pause

================================================================================
FILE: run-local.sh
Size: 1.28 KB | Tokens: 314
================================================================================

#!/bin/bash
# Unix script to run OmniUI MCP Server locally
# This script starts the server for local development

set -e  # Exit on any error

echo "========================================"
echo "OmniUI MCP Server - Local Development"
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
export OMNI_UI_DISABLE_USAGE_LOGGING=false
export MCP_PORT=9901

# Display startup information
echo "Starting OmniUI MCP Server..."
echo "Config: examples/local_config.yaml"
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
poetry run omni-ui-aiq examples/local_config.yaml

================================================================================
FILE: setup-dev.bat
Size: 2.39 KB | Tokens: 534
================================================================================

@echo off
REM Windows development setup script for OmniUI MCP Server
REM This script sets up the local development environment

echo ========================================
echo OmniUI MCP Server - Development Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python version:
python --version

REM Check if Poetry is available
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Poetry not found. Installing Poetry...
    echo.
    curl -sSL https://install.python-poetry.org | python -
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Poetry
        echo Please install Poetry manually from https://python-poetry.org/docs/#installation
        pause
        exit /b 1
    )
    echo.
    echo Poetry installed successfully!
    echo Please restart this script or refresh your PATH to use Poetry
    echo You may need to restart your command prompt
    pause
    exit /b 0
)

echo Poetry version:
poetry --version

REM Configure Poetry to create virtual environment in project directory
echo.
echo Configuring Poetry to use local virtual environment...
poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true

REM Install dependencies
echo.
echo Installing dependencies...
poetry install

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

REM Create local directories if they don't exist
if not exist "logs" mkdir logs

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run 'run-local.bat' to start the MCP server
echo 2. The server will be available at http://localhost:9901
echo.
echo Optional: Run 'poetry run python validate-setup.py' to verify setup
echo.
echo For development:
echo - Use 'poetry shell' to activate the virtual environment
echo - Use 'poetry run omni-ui-aiq' to run the server manually
echo - Edit examples/local_config.yaml to customize configuration
echo.
pause

================================================================================
FILE: setup-dev.sh
Size: 2.78 KB | Tokens: 687
================================================================================

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
poetry config virtualenvs.prefer-active-python true

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
echo "1. Run './run-local.sh' to start the MCP server"
echo "2. The server will be available at http://localhost:9901"
echo
echo "Optional: Run 'poetry run python validate-setup.py' to verify setup"
echo
echo "For development:"
echo "- Use 'poetry shell' to activate the virtual environment"
echo "- Use 'poetry run omni-ui-aiq' to run the server manually"
echo "- Edit examples/local_config.yaml to customize configuration"
echo

================================================================================
FILE: start-server.bat
Size: 341.00 B | Tokens: 74
================================================================================

@echo off
REM Simple script to start the OmniUI MCP Server

echo ============================================
echo Starting OmniUI MCP Server
echo ============================================
echo.
echo Server will run on: http://localhost:9901
echo Press Ctrl+C to stop
echo.

cd /d "%~dp0"
poetry run python -m omni_aiq_omni_ui
