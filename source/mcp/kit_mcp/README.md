# Kit MCP Server

A Model Context Protocol (MCP) server implementation providing comprehensive NVIDIA Omniverse Kit development tools, built using NAT (NeMo Agent Toolkit) 1.3+.

## Overview

This MCP server provides intelligent assistance for Kit developers by offering complete access to Kit extensions, APIs, documentation, and code examples. The server exposes 12 specialized tools for comprehensive Kit development support.

## Features

### Core Documentation Functions

- **get_kit_instructions**: Returns Kit system documentation and best practices
- **search_kit_extensions**: Semantic search across 400+ Kit extensions using RAG techniques
- **get_kit_extension_details**: Provides detailed information about specific extensions
- **get_kit_extension_dependencies**: Analyzes and visualizes extension dependency graphs
- **get_kit_extension_apis**: Lists all APIs provided by extensions
- **get_kit_api_details**: Gets detailed documentation for specific APIs
- **search_kit_code_examples**: Finds relevant code examples using semantic search
- **search_kit_test_examples**: Finds test implementations and patterns
- **search_kit_settings**: Searches Kit settings and configuration options
- **search_kit_app_templates**: Finds Kit application templates
- **get_kit_app_template_details**: Gets detailed information about specific app templates
- **search_kit_knowledge**: Searches Kit documentation and knowledge base using semantic search

### Additional Features

- **Extension Discovery**: Complete Kit extension coverage with hierarchical information retrieval
- **Code Examples**: Semantic search through Kit code examples with FAISS indexing
- **System Instructions**: Bundled documentation for Kit framework, extensions, testing, USD, and UI
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

The server will be available at `http://localhost:9902`

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
cd source\mcp\kit_mcp
setup-dev.bat
```

**Linux/macOS:**
```bash
cd source/mcp/kit_mcp
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

The server will be available at: `http://localhost:9902/mcp`

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
docker compose -f docker-compose.ngc.yaml up kit-mcp --build
```

#### Alternative: Build and Run Individual Container

```bash
cd source/mcp/kit_mcp
./build-docker.sh    # Linux/macOS
# or
build-docker.bat     # Windows

# Run container (ensure .env is configured)
docker run --rm -p 9902:9902 --env-file ../.env kit-mcp:latest
```

The server will start on port 9902.

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
./build-wheels.sh kit
```

**Windows:**
```cmd
cd source\mcp
build-wheels.bat kit
```

#### Step 5: Start All Services with Docker Compose

```bash
cd source/mcp
docker compose -f docker-compose.local.yaml up --build
```

This starts:
- **Embedder NIM** on port 8001
- **Reranker NIM** on port 8002
- **Kit MCP Server** on port 9902

#### Step 6: Verify Services

Wait 1-2 minutes for NIM containers to load models, then:

```bash
# Check embedder health
curl http://localhost:8001/v1/health

# Check reranker health
curl http://localhost:8002/v1/health

# Check MCP server
curl http://localhost:9902/health
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
docker run -p 9902:9902 \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e KIT_EMBEDDER_BACKEND=local \
  -e KIT_LOCAL_EMBEDDER_URL=$KIT_LOCAL_EMBEDDER_URL \
  -e KIT_RERANKER_BACKEND=local \
  -e KIT_LOCAL_RERANKER_URL=$KIT_LOCAL_RERANKER_URL \
  kit-mcp:latest
```

See the **[Local Deployment Guide](../LOCAL_DEPLOYMENT.md)** for more details.

---

## Available Tools

### get_kit_instructions

Returns Kit system documentation and development best practices.

**Parameters**: 
- `instruction_sets` (optional): Specific instruction sets to retrieve

**Returns**: 
```json
{
  "kit_system": "Core Kit framework fundamentals and architecture",
  "extensions": "Extension development guidelines and patterns",
  "testing": "Test writing best practices and framework usage",
  "usd": "USD integration and scene description patterns",
  "ui": "UI development with Kit widgets and layouts"
}
```

### search_kit_extensions

Semantic search across 400+ Kit extensions using RAG techniques.

**Parameters**:
- `query` (string): Search query for finding relevant extensions
- `top_k` (integer, optional): Number of results to return (default: 10)
- `categories` (array, optional): Filter by extension categories

**Returns**: Ranked list of relevant extensions with scores and descriptions

### get_kit_extension_details

Provides detailed information about specific extensions.

**Parameters**:
- `extension_ids` (string/array): Extension IDs to retrieve (supports flexible input)

**Returns**: Detailed extension information including features, dependencies, configuration

### get_kit_extension_dependencies

Analyzes and visualizes extension dependency graphs.

**Parameters**:
- `extension_id` (string): Extension ID to analyze dependencies for
- `depth` (integer, optional): Dependency tree depth (default: 2)
- `include_optional` (boolean, optional): Include optional dependencies

**Returns**: Dependency tree with version requirements and conflict analysis

### get_kit_extension_apis

Lists all APIs provided by extensions.

**Parameters**:
- `extension_ids` (string/array): Extension IDs to get APIs for

**Returns**: Structured API listing with classes, methods, and functions

### get_kit_api_details

Gets detailed documentation for specific APIs.

**Parameters**:
- `api_references` (string/array): API references in format 'extension_id@symbol'

**Returns**: Complete API documentation with docstrings, parameters, return types

### search_kit_code_examples

Finds relevant code examples using semantic search.

**Parameters**:
- `query` (string): Description of desired code functionality

**Returns**: Formatted code examples with file paths and implementation details

### search_kit_test_examples

Finds test implementations and patterns.

**Parameters**:
- `query` (string): Test scenario or functionality to find examples for

**Returns**: Test examples with setup, execution, and validation patterns

### search_kit_settings

Searches Kit settings and configuration options.

**Parameters**:
- `query` (string): Setting search query

**Returns**: Settings information with paths, types, defaults, and documentation

### search_kit_app_templates

Finds Kit application templates.

**Parameters**:
- `query` (string): Template search query

**Returns**: Matching application templates with descriptions

### get_kit_app_template_details

Gets detailed information about specific app templates.

**Parameters**:
- `template_id` (string): Template ID to retrieve details for

**Returns**: Complete template information including structure and configuration

### search_kit_knowledge

Searches Kit documentation and knowledge base using semantic search.

**Parameters**:
- `query` (string): Natural language search query

**Returns**: Formatted markdown with relevant documentation

## Configuration

The server uses `workflows/config.yaml` for production and `workflows/local_config.yaml` for development:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 16384

functions:
  get_kit_instructions:
    _type: kit_fns/get_kit_instructions
    verbose: false
  
  search_kit_extensions:
    _type: kit_fns/search_kit_extensions
    verbose: false
    enable_rerank: true
    rerank_k: 10
  
  # ... additional function configurations

workflow:
  _type: react_agent
  llm_name: nim_llm
  tool_names:
    - get_kit_instructions
    - search_kit_extensions
    - get_kit_extension_details
    - get_kit_extension_dependencies
    - get_kit_extension_apis
    - get_kit_api_details
    - search_kit_code_examples
    - search_kit_test_examples
    - search_kit_settings
    - search_kit_app_templates
    - get_kit_app_template_details
    - search_kit_knowledge
```

## Architecture

The project follows the AIQ toolkit patterns:

```
source/mcp/kit_mcp/
├── VERSION.md                           # Version file
├── README.md                            # This file
├── pyproject.toml                       # Poetry configuration
├── Dockerfile                           # Docker configuration
├── check_mcp_health.py                  # Health check script
├── setup-dev.sh / setup-dev.bat         # Development setup scripts
├── run.sh / run.bat         # Local run scripts
├── build-docker.sh / build-docker.bat   # Docker build scripts
├── workflows/
│   ├── config.yaml                      # Production configuration
│   └── local_config.yaml                # Development configuration
└── src/
    └── kit_mcp/
        ├── __init__.py                  # Package initialization with AIQ patching
        ├── __main__.py                  # Entry point for running the server
        ├── config.py                    # Configuration constants
        ├── data/
        │   ├── instructions/            # System instruction documents
        │   ├── extensions_index/        # Extension metadata and search index
        │   └── code_examples_index/     # Code examples FAISS index
        ├── functions/                   # Function implementations
        ├── register_*.py                # AIQ function registrations
        ├── services/                    # Core services
        ├── utils/                       # Utility functions
        └── models/
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_PORT` | Server port | 9902 |
| `NVIDIA_API_KEY` | Required for LLM and NVIDIA API embeddings/reranking | - |
| `NGC_API_KEY` | Required for pulling NIM images (local deployment only) | - |
| `KIT_EMBEDDER_BACKEND` | Embedder backend: `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_EMBEDDER_URL` | Local embedder URL (when backend=local) | - |
| `KIT_RERANKER_BACKEND` | Reranker backend: `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_RERANKER_URL` | Local reranker URL (when backend=local) | - |
| `KIT_MCP_DISABLE_USAGE_LOGGING` | Disable usage analytics | false |

## Usage Analytics

The server includes built-in usage analytics that log:
- Tool calls and parameters
- Success/failure status
- Execution times
- Error messages

Analytics can be disabled by setting `KIT_MCP_DISABLE_USAGE_LOGGING=true`.

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
    "kit-mcp": {
      "url": "http://localhost:9902/mcp"
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
- Check if another service is using port 9902: `netstat -an | grep 9902`
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

### Adding New Tools

When adding new tools to the MCP server, follow this checklist:

1. **Create function implementation** in `src/kit_mcp/functions/`
2. **Create registration wrapper** `register_[tool_name].py`
3. **Update pyproject.toml** entry points
4. **Update configuration** in `workflows/config.yaml`
5. **Update package version** in `src/kit_mcp/__init__.py`
6. **Reinstall package**: `poetry install`

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
