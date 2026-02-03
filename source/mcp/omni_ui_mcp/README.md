# OmniUI MCP Server

A Model Context Protocol (MCP) server implementation providing comprehensive OmniUI API information and development tools, built using NAT (NeMo Agent Toolkit) 1.3+.

## Overview

This MCP server provides intelligent assistance for Omniverse UI developers by offering complete access to the OmniUI Atlas database, which contains detailed information about all OmniUI classes, modules, and methods. The server exposes 10 specialized tools for comprehensive OmniUI development support.

## Features

### Core API Functions

- **list_ui_classes**: Returns all OmniUI class names from the Atlas database
- **list_ui_modules**: Lists all OmniUI modules with extension information
- **get_ui_class_detail**: Provides detailed information about specific classes including methods and documentation
- **get_ui_module_detail**: Shows comprehensive module information with associated classes and functions
- **get_ui_method_detail**: Returns detailed method signatures, parameters, and documentation
- **get_omni_ui_code_example**: Retrieves relevant OmniUI code examples using semantic search with reranking
- **get_ui_instructions**: Retrieves OmniUI system instructions and documentation for code generation

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
setup-dev.bat    # Run once to set up environment
run.bat    # Run to start server
```

**Linux/macOS:**
```bash
./setup-dev.sh  # Run once to set up environment
./run.sh  # Run to start server
```

The server will be available at `http://localhost:9901`

### Prerequisites

- Python 3.11+
- Poetry (for development)
- Docker (for containerized deployment)
- NVIDIA API Key (for embeddings and reranking via NVIDIA API)
- OR: GPUs with NIM containers (for local embeddings and reranking)

---

## Deployment Options

There are **two main deployment options** for embedding and reranking services:

| Option | GPU Required | Best For |
|--------|--------------|----------|
| **NVIDIA API** | No | Getting started, cloud deployment |
| **Local NIMs** | Yes (1-2 GPUs) | Production, data privacy, no rate limits |

---

## Option 1: NVIDIA API Deployment (Easiest - No GPU Required)

Uses NVIDIA's cloud endpoints for embeddings and reranking. Best for getting started quickly.

### Step-by-Step Setup

#### Step 1: Configure Environment

1. Go to [build.nvidia.com](https://build.nvidia.com) and generate an API key
2. Copy the environment template and add your key:

```bash
cd source/mcp
cp .env.example .env
# Edit .env and set NVIDIA_API_KEY=your_api_key_here
```

> **Note**: For local development (not Docker), you can alternatively export the variable directly:
> ```bash
> export NVIDIA_API_KEY=your_api_key_here
> ```

#### Step 2: Set Up Development Environment

**Windows:**
```cmd
cd source\mcp\omni_ui_mcp
setup-dev.bat
```

**Linux/macOS:**
```bash
cd source/mcp/omni_ui_mcp
./setup-dev.sh
```

This will:
- Check for Python 3.11+ and Poetry installation
- Install Poetry if not available
- Configure Poetry to use local virtual environment
- Install all project dependencies

#### Step 3: Run the Server

**Windows:**
```cmd
run.bat
```

**Linux/macOS:**
```bash
./run.sh
```

The server will be available at: `http://localhost:9901/mcp`

### Docker Deployment (NVIDIA API)

#### Step 1: Configure Environment

Ensure your `.env` file is set up in the `source/mcp` directory:

```bash
cd source/mcp
cp .env.example .env
# Edit .env and set NVIDIA_API_KEY=nvapi-xxxxx
```

#### Step 2: Run with Docker Compose (Recommended)

```bash
cd source/mcp
docker compose -f docker-compose.ngc.yaml up omni-ui-mcp --build
```

#### Alternative: Build and Run Individual Container

```bash
cd source/mcp/omni_ui_mcp
./build-docker.sh    # Linux/macOS
# or
build-docker.bat     # Windows

# Run container (ensure .env is configured)
docker run --rm -p 9901:9901 --env-file ../.env omni-ui-mcp:latest
```

The server will start on port 9901.

---

## Option 2: Local Deployment with Local Embedder/Reranker (GPU Required)

Run embedder and reranker models locally using NVIDIA NIM containers. Better for production, data privacy, and avoiding API rate limits.

### Prerequisites for Local Deployment

- **1-2 NVIDIA GPUs** (or 1 GPU with sufficient VRAM for both NIMs)
- **NGC API Key** for pulling NIM images
- **Docker with NVIDIA Container Toolkit**
- **NVIDIA API Key** (still needed for the LLM)

### Step-by-Step Setup

#### Step 1: Get Required API Keys

1. **NVIDIA API Key**: Go to [build.nvidia.com](https://build.nvidia.com) and generate a key
2. **NGC API Key**: Go to [ngc.nvidia.com](https://ngc.nvidia.com) and generate a key

#### Step 2: Configure Environment

```bash
cd source/mcp
cp .env.example .env
# Edit .env and set both keys:
# NVIDIA_API_KEY=nvapi-xxxxx
# NGC_API_KEY=your_ngc_key
```

#### Step 3: Login to NGC Docker Registry

```bash
source .env  # Load environment variables
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
```

#### Step 4: Build Wheels (Required for Docker)

From the `source/mcp` directory:

**Linux/macOS:**
```bash
cd source/mcp
./build-wheels.sh omni
```

**Windows:**
```cmd
cd source\mcp
build-wheels.bat omni
```

#### Step 5: Start All Services with Docker Compose

```bash
cd source/mcp
docker compose -f docker-compose.local.yaml up --build
```

This starts:
- **Embedder NIM** on port 8001
- **Reranker NIM** on port 8002
- **OmniUI MCP Server** on port 9901

#### Step 6: Verify Services

Wait 1-2 minutes for NIM containers to load models, then:

```bash
# Check embedder health
curl http://localhost:8001/v1/health

# Check reranker health
curl http://localhost:8002/v1/health

# Check MCP server
curl http://localhost:9901/health
```

### Using Existing External Embedder/Reranker

If you already have embedder and reranker services running elsewhere:

```bash
export KIT_EMBEDDER_BACKEND=local
export KIT_LOCAL_EMBEDDER_URL=http://your-embedder-host:8000
export KIT_RERANKER_BACKEND=local
export KIT_LOCAL_RERANKER_URL=http://your-reranker-host:8000
export NVIDIA_API_KEY=your_key  # Still needed for LLM

# Then run the MCP server
./run.sh
# or
docker run -p 9901:9901 \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e KIT_EMBEDDER_BACKEND=local \
  -e KIT_LOCAL_EMBEDDER_URL=$KIT_LOCAL_EMBEDDER_URL \
  -e KIT_RERANKER_BACKEND=local \
  -e KIT_LOCAL_RERANKER_URL=$KIT_LOCAL_RERANKER_URL \
  omni-ui-mcp:latest
```

See the **[Local Deployment Guide](../LOCAL_DEPLOYMENT.md)** for more details.

---

## Configuration

The server uses `workflow/config.yaml` for configuration:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 16384

functions:
  get_omni_ui_code_example:
    _type: omni_ui_mcp/get_omni_ui_code_example
    verbose: false
    enable_rerank: true
    rerank_k: 10

  list_ui_classes:
    _type: omni_ui_mcp/list_ui_classes
    verbose: false

  list_ui_modules:
    _type: omni_ui_mcp/list_ui_modules
    verbose: false

  get_ui_class_detail:
    _type: omni_ui_mcp/get_ui_class_detail
    verbose: false

  get_ui_module_detail:
    _type: omni_ui_mcp/get_ui_module_detail
    verbose: false

  get_ui_method_detail:
    _type: omni_ui_mcp/get_ui_method_detail
    verbose: false

  get_ui_instructions:
    _type: omni_ui_mcp/get_ui_instructions
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
    - get_ui_instructions
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

### get_ui_instructions

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
get_ui_instructions                          # Lists all available instructions
get_ui_instructions name="agent_system"      # Get core system prompt
get_ui_instructions name="omni_ui_system"    # Get widgets and layouts documentation
get_ui_instructions name="omni_ui_scene_system"  # Get 3D UI documentation
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
get_ui_instructions name="omni_ui_system"
```

## Architecture

The project follows the AIQ toolkit patterns:

```
source/mcp/omni_ui_mcp/
├── VERSION.md                           # Version file
├── README.md                            # This file
├── pyproject.toml                       # Poetry configuration
├── Dockerfile                           # Docker configuration
├── check_mcp_health.py                  # Health check script
├── setup-dev.sh / setup-dev.bat         # Development setup scripts
├── run.sh / run.bat         # Local run scripts
├── build-docker.sh / build-docker.bat   # Docker build scripts
├── workflow/
│   └── config.yaml                      # Workflow configuration
└── src/
    └── omni_ui_mcp/
        ├── __init__.py                  # Package initialization with AIQ patching
        ├── __main__.py                  # Entry point for running the server
        ├── config.py                    # Configuration constants
        ├── data/
        │   ├── instructions/            # System instruction documents
        │   ├── faiss_index_omni_ui/     # FAISS index for code examples
        │   ├── omni_ui_rag_collection.json  # RAG collection data
        │   └── ui_atlas.json            # Atlas database
        ├── functions/                   # Function implementations
        ├── register_*.py                # AIQ function registrations
        ├── services/                    # Core services
        ├── utils/                       # Utility functions
        └── models/
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_PORT` | Server port | 9901 |
| `NVIDIA_API_KEY` | Required for LLM and NVIDIA API embeddings/reranking | - |
| `NGC_API_KEY` | Required for pulling NIM images (local deployment only) | - |
| `KIT_EMBEDDER_BACKEND` | Embedder backend: `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_EMBEDDER_URL` | Local embedder URL (when backend=local) | - |
| `KIT_RERANKER_BACKEND` | Reranker backend: `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_RERANKER_URL` | Local reranker URL (when backend=local) | - |
| `OMNI_UI_DISABLE_USAGE_LOGGING` | Disable usage analytics | false |

## Usage Analytics

The server includes built-in usage analytics that log:
- Tool calls and parameters
- Success/failure status
- Execution times
- Error messages

Analytics can be disabled by setting `OMNI_UI_DISABLE_USAGE_LOGGING=true`.

## Port Allocation

To avoid conflicts when running multiple MCP servers:
- **omni-ui-mcp**: Port 9901
- **kit-mcp**: Port 9902
- **usd-code-mcp**: Port 9903
- **isaacsim-mcp**: Port 9904

## Integration with Cursor IDE

Create a `.cursor/mcp.json` file in your **project/workspace root**:

```bash
# Create the .cursor directory if it doesn't exist
mkdir -p .cursor

# Create the MCP configuration
cat > .cursor/mcp.json << 'EOF'
{
  "mcpServers": {
    "omni-ui-mcp": {
      "url": "http://localhost:9901/mcp"
    }
  }
}
EOF
```

After creating the file, **reload Cursor** (Cmd/Ctrl+Shift+P → "Developer: Reload Window").

> **Note:** NAT 1.3+ uses streamable-http at `/mcp` instead of SSE at `/sse`.

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

**"NVIDIA_API_KEY not set" error:**
- Set your NVIDIA API key: `export NVIDIA_API_KEY=your_key`
- Or add it to your environment configuration

**Port already in use:**
- Check if another service is using port 9901: `netstat -an | grep 9901`
- Set a different port: `set MCP_PORT=9902` (Windows) or `export MCP_PORT=9902` (Unix)
- Or modify the port in `workflow/config.yaml`

**NIM containers fail to start (Local Deployment):**
- Install NVIDIA Container Toolkit: `sudo apt-get install -y nvidia-container-toolkit`
- Restart Docker: `sudo systemctl restart docker`

**NIM containers show unhealthy status:**
- NIMs take 1-2 minutes to load models
- Check logs: `docker logs mcp-embedder` or `docker logs mcp-reranker`

**GPU memory issues:**
- Check GPU usage: `nvidia-smi`
- Try using 2 GPUs (one per NIM) instead of single GPU mode

**Virtual environment issues:**
- Delete `.venv` directory and run setup script again
- Make sure Poetry is configured correctly: `poetry config --list`

### Development Tips

- Use `poetry shell` to activate the virtual environment for manual testing
- Check logs in the `logs/` directory if created
- Modify `workflow/config.yaml` to adjust development settings
- Use `poetry show` to list installed dependencies
- Use `poetry add --group dev <package>` to add development dependencies

## Development

### Adding New Tools

When adding new tools to the MCP server, follow this checklist:

1. **Create function implementation** in `src/omni_ui_mcp/functions/`
   - Implement async function with proper error handling
   - Include logging and documentation

2. **Create registration wrapper** `register_[tool_name].py`
   - Define Pydantic input schema
   - Write comprehensive tool description
   - Include usage logging

3. **Update pyproject.toml**
   - Add entry point in `[tool.poetry.plugins."aiq.components"]`
   - Increment version number

4. **Update configuration** in `workflow/config.yaml`
   - Add function definition in `functions:` section
   - Add tool to `workflow.tool_names` list

5. **Update package version** in `src/omni_ui_mcp/__init__.py`

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
