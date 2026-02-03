

# USD Code MCP Server

A Model Context Protocol (MCP) server implementation providing comprehensive USD/OpenUSD API information and development tools, built using NAT (NeMo Agent Toolkit) 1.3+.

## Overview

This MCP server provides intelligent assistance for USD/OpenUSD developers by offering access to the USD Atlas database, code examples, and knowledge retrieval. The server exposes 7 specialized tools for comprehensive USD development support.

## Features

### Core API Functions

- **list_usd_modules**: Lists all USD modules from the Atlas database
- **list_usd_classes**: Returns all USD class names with optional module filtering
- **get_usd_module_detail**: Provides detailed information about specific modules
- **get_usd_class_detail**: Shows comprehensive class information including methods and documentation
- **get_usd_method_detail**: Returns detailed method signatures, parameters, and documentation
- **search_usd_code_examples**: Retrieves relevant USD code examples using semantic search with reranking
- **search_usd_knowledge**: Searches USD documentation and knowledge base using semantic search

### Additional Features

- **Atlas Database**: Complete USD API coverage with class hierarchy and documentation
- **Code Examples**: Semantic search through USD code examples with FAISS indexing
- **Knowledge Base**: Comprehensive USD documentation with semantic search
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

The server will be available at `http://localhost:9903`

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
cd source\mcp\usd_code_mcp
setup-dev.bat
```

**Linux/macOS:**
```bash
cd source/mcp/usd_code_mcp
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

The server will be available at: `http://localhost:9903/mcp`

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
docker compose -f docker-compose.ngc.yaml up usd-code-mcp --build
```

#### Alternative: Build and Run Individual Container

```bash
cd source/mcp/usd_code_mcp
./build-docker.sh    # Linux/macOS
# or
build-docker.bat     # Windows

# Run container (ensure .env is configured)
docker run --rm -p 9903:9903 --env-file ../.env usd-code-mcp:latest
```

The server will start on port 9903.

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
./build-wheels.sh usd
```

**Windows:**
```cmd
cd source\mcp
build-wheels.bat usd
```

#### Step 5: Start All Services with Docker Compose

```bash
cd source/mcp
docker compose -f docker-compose.local.yaml up --build
```

This starts:
- **Embedder NIM** on port 8001
- **Reranker NIM** on port 8002
- **USD Code MCP Server** on port 9903

#### Step 6: Verify Services

Wait 1-2 minutes for NIM containers to load models, then:

```bash
# Check embedder health
curl http://localhost:8001/v1/health

# Check reranker health
curl http://localhost:8002/v1/health

# Check MCP server
curl http://localhost:9903/health
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
docker run -p 9903:9903 \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e KIT_EMBEDDER_BACKEND=local \
  -e KIT_LOCAL_EMBEDDER_URL=$KIT_LOCAL_EMBEDDER_URL \
  -e KIT_RERANKER_BACKEND=local \
  -e KIT_LOCAL_RERANKER_URL=$KIT_LOCAL_RERANKER_URL \
  usd-code-mcp:latest
```

See the **[Local Deployment Guide](../LOCAL_DEPLOYMENT.md)** for more details.

---

## Configuration

The server uses `workflow/config.yaml` for production and `workflow/local_config.yaml` for development:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 16384

functions:
  search_usd_code_examples:
    _type: omni_aiq_usd_code/search_usd_code_examples
    verbose: false
    enable_rerank: true
    rerank_k: 10

  search_usd_knowledge:
    _type: omni_aiq_usd_code/search_usd_knowledge
    verbose: false
    enable_rerank: true
    rerank_k: 10

  list_usd_modules:
    _type: omni_aiq_usd_code/list_usd_modules
    verbose: false

  list_usd_classes:
    _type: omni_aiq_usd_code/list_usd_classes
    verbose: false

  get_usd_module_detail:
    _type: omni_aiq_usd_code/get_usd_module_detail
    verbose: false

  get_usd_class_detail:
    _type: omni_aiq_usd_code/get_usd_class_detail
    verbose: false

  get_usd_method_detail:
    _type: omni_aiq_usd_code/get_usd_method_detail
    verbose: false
```

## Available Tools

### list_usd_modules

Lists all USD modules from the Atlas database.

**Parameters**: None

**Returns**:
```json
{
  "module_names": ["Usd", "UsdGeom", "UsdShade", ...],
  "total_count": 30,
  "description": "USD modules from Atlas data"
}
```

### list_usd_classes

Returns all USD class names, optionally filtered by module.

**Parameters**:
- `module_name` (string, optional): Filter classes by module name

**Returns**:
```json
{
  "class_full_names": ["UsdStage", "UsdPrim", "UsdAttribute", ...],
  "total_count": 200,
  "description": "USD classes from Atlas data"
}
```

### get_usd_module_detail

Provides detailed information about a specific USD module.

**Parameters**:
- `module_name` (string): Name of the module (supports partial matching)

**Returns**: Module details including classes, functions, and documentation

### get_usd_class_detail

Shows comprehensive class information.

**Parameters**:
- `class_name` (string): Name of the class (supports partial matching)

**Returns**: Detailed class information including methods, parent classes, and documentation

### get_usd_method_detail

Returns detailed method information.

**Parameters**:
- `method_name` (string): Name of the method (supports partial matching)

**Returns**: Method signature, parameters, return type, and documentation

### search_usd_code_examples

Retrieves relevant USD code examples using semantic search with optional reranking.

**Parameters**:
- `query` (string): Natural language search query describing what you're looking for
- `top_k` (integer, optional): Number of results to return (default: 5)

**Returns**: Formatted markdown with relevant code examples and explanations

**Example**:
```
search_usd_code_examples query="create a sphere primitive"
search_usd_code_examples query="how to apply materials" top_k=3
```

### search_usd_knowledge

Searches USD documentation and knowledge base using semantic search.

**Parameters**:
- `query` (string): Natural language search query
- `top_k` (integer, optional): Number of results to return (default: 5)

**Returns**: Formatted markdown with relevant documentation

**Example**:
```
search_usd_knowledge query="what are variants in USD"
search_usd_knowledge query="composition arcs explained"
```

## Usage Examples

```
# Get all available modules
list_usd_modules

# Get all USD classes
list_usd_classes

# Get classes from a specific module
list_usd_classes module_name="UsdGeom"

# Get detailed information about a module
get_usd_module_detail module_name="UsdGeom"

# Get detailed information about a class
get_usd_class_detail class_name="UsdGeomMesh"

# Look up method details
get_usd_method_detail method_name="GetPointsAttr"

# Search for code examples
search_usd_code_examples query="create a stage and add primitives"

# Search knowledge base
search_usd_knowledge query="how do I use layer offsets"
```

## Architecture

The project follows the AIQ toolkit patterns:

```
source/mcp/usd_code_mcp/
├── VERSION.md                           # Version file
├── README.md                            # This file
├── pyproject.toml                       # Poetry configuration
├── Dockerfile                           # Docker configuration
├── check_mcp_health.py                  # Health check script
├── setup-dev.sh / setup-dev.bat         # Development setup scripts
├── run.sh / run.bat         # Local run scripts
├── build-docker.sh / build-docker.bat   # Docker build scripts
├── workflow/
│   ├── config.yaml                      # Production configuration
│   └── local_config.yaml                # Development configuration
└── src/
    └── usd_code_mcp/
        ├── __init__.py                  # Package initialization
        └── __main__.py                  # Entry point for running the server
```

The actual USD functions are in `source/aiq/usd_code_fns/`:

```
source/aiq/usd_code_fns/
└── src/omni_aiq_usd_code/
    ├── functions/                       # Function implementations
    │   ├── list_usd_modules.py
    │   ├── list_usd_classes.py
    │   ├── get_usd_module_detail.py
    │   ├── get_usd_class_detail.py
    │   ├── get_usd_method_detail.py
    │   ├── search_usd_code_examples.py
    │   └── search_usd_knowledge.py
    ├── register_*.py                    # AIQ function registrations
    ├── services/                        # Core services
    │   ├── usd_atlas.py                # Atlas database service
    │   ├── retrieval.py                # FAISS retrieval service
    │   ├── reranking.py                # Reranking service
    │   └── feedback.py                 # Feedback service
    └── utils/                          # Utility functions
        ├── fuzzy_matching.py
        └── usage_logging_decorator.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_PORT` | Server port | 9903 |
| `NVIDIA_API_KEY` | Required for LLM and NVIDIA API embeddings/reranking | - |
| `NGC_API_KEY` | Required for pulling NIM images (local deployment only) | - |
| `KIT_EMBEDDER_BACKEND` | Embedder backend: `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_EMBEDDER_URL` | Local embedder URL (when backend=local) | - |
| `KIT_RERANKER_BACKEND` | Reranker backend: `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_RERANKER_URL` | Local reranker URL (when backend=local) | - |

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
    "usd-code-mcp": {
      "url": "http://localhost:9903/mcp"
    }
  }
}
EOF
```

After creating the file, **reload Cursor** (Cmd/Ctrl+Shift+P → "Developer: Reload Window").

> **Note:** NAT 1.3+ uses streamable-http at `/mcp` instead of SSE at `/sse`.

## Troubleshooting

### Common Issues

**"aiq command not found" error:**
- Make sure all dependencies are installed: `poetry install`
- Activate the virtual environment: `poetry shell`
- Check that aiqtoolkit is properly installed: `poetry show aiqtoolkit`

**"NVIDIA_API_KEY not set" error:**
- Set your NVIDIA API key: `export NVIDIA_API_KEY=your_key`
- Or add it to your environment configuration

**Port already in use:**
- Check if another service is using port 9903: `netstat -an | grep 9903`
- Set a different port: `export MCP_PORT=9905`

**NIM containers fail to start (Local Deployment):**
- Install NVIDIA Container Toolkit: `sudo apt-get install -y nvidia-container-toolkit`
- Restart Docker: `sudo systemctl restart docker`

**NIM containers show unhealthy status:**
- NIMs take 1-2 minutes to load models
- Check logs: `docker logs mcp-embedder` or `docker logs mcp-reranker`

**GPU memory issues:**
- Check GPU usage: `nvidia-smi`
- Try using 2 GPUs (one per NIM) instead of single GPU mode

## Development

### Building

```bash
poetry build
```

### Testing

```bash
poetry run pytest
```

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE file in the repository root for details.
