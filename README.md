# Kit USD Agents

AI-powered development tools for NVIDIA Omniverse, including Chat USD and Model Context Protocol (MCP) servers for Kit, USD, and OmniUI development.

## What's Included

This repository provides two types of AI development tools:

### 1. Chat USD Extension

Chat USD is an AI assistant that runs inside NVIDIA Omniverse Kit, enabling natural language interaction with USD scenes.

**Capabilities:**
- **USD Code Generation**: Generate and execute USD Python code from natural language
- **Asset Search**: Find USD assets using conversational queries
- **Scene Analysis**: Get information about your current USD scene
- **Real-time Editing**: Modify scenes interactively through conversation
- **Custom Agents**: Extensible architecture for adding specialized capabilities

### 2. MCP Servers

Three standalone [Model Context Protocol](https://modelcontextprotocol.io/) servers that integrate with AI coding assistants like Claude, Cursor, and other MCP-compatible tools.

#### USD Code MCP Server
Provides USD/OpenUSD API assistance with 8 specialized tools:
- Browse USD modules and classes
- Get detailed API documentation
- Search code examples with semantic search
- Access USD knowledge base

#### Kit MCP Server  
Comprehensive Kit development assistance with 12 specialized tools:
- Search 400+ Kit extensions
- Explore extension APIs and dependencies
- Find code examples and test patterns
- Access Kit documentation and settings

#### OmniUI MCP Server
OmniUI development assistance with 7 specialized tools:
- Browse UI classes and modules
- Get widget documentation and examples
- Search styling and scene system APIs

## Getting Started

### Chat USD Extension

Build and run in Omniverse Kit:
```bash
# Windows
build.bat -r
_build\windows-x86_64\release\omni.app.chat_usd.bat

# Linux
./build.sh -r
./_build/linux-x86_64/release/omni.app.chat_usd.sh
```

### MCP Servers

Each MCP server can be run locally with Docker or Python. See the individual server documentation:
- [USD Code MCP](source/mcp/usd_code_mcp/README.md)
- [Kit MCP](source/mcp/kit_mcp/README.md)
- [OmniUI MCP](source/mcp/omni_ui_mcp/README.md)

**Quick Start (Docker - Recommended):**
```bash
# Set your NVIDIA API key first
export NVIDIA_API_KEY=your_api_key_here  # Linux/macOS
# or
set NVIDIA_API_KEY=your_api_key_here     # Windows

# Build wheels and start all MCP servers
cd source/mcp
./build-wheels.sh all  # or build-wheels.bat all on Windows
docker compose -f docker-compose.ngc.yaml up --build
```

This uses NVIDIA's cloud-hosted embedder and reranker services via your API key. No local GPUs required.

**Local GPU Deployment (Advanced):**

For running embedder/reranker models locally on your own GPUs (requires 2 NVIDIA GPUs):
```bash
export NGC_API_KEY=your_ngc_api_key      # For pulling NIM container images
export NVIDIA_API_KEY=your_nvidia_api_key

cd source/mcp
./build-wheels.sh all
docker compose -f docker-compose.local.yaml up --build
```

See [Local Deployment Guide](source/mcp/LOCAL_DEPLOYMENT.md) for details.

**Requirements:**
- [NVIDIA API Key](https://build.nvidia.com/) (for embeddings, reranking, and LLM access)
- Docker (recommended for deployment)

## Documentation

- [Chat USD Architecture](source/extensions/omni.ai.chat_usd.bundle/docs/README.md)
- [MCP Local Deployment Guide](source/mcp/LOCAL_DEPLOYMENT.md)

## Contributing

This project is currently not accepting contributions.
