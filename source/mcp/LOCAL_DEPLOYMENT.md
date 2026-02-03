# MCP Servers Deployment Guide

This guide explains the different deployment options for running embedder and reranker services with the MCP servers.

## Deployment Options Overview

| Option | GPUs Required | Setup Complexity | Best For |
|--------|---------------|------------------|----------|
| **NVIDIA API (Default)** | None | Easiest | Getting started |
| **Local NIMs** | 1-2 GPUs | Medium | Production, data privacy |
| **External Services** | Varies | Flexible | Existing infrastructure |

---

## Prerequisites

### Environment Setup

1. **Copy the environment template:**
```bash
cd source/mcp
cp .env.example .env
```

2. **Edit `.env` and add your API keys:**
```bash
# Required for all deployment options
NVIDIA_API_KEY=nvapi-xxxxx

# Required only for Local NIMs deployment (Option 2)
NGC_API_KEY=your_ngc_key
```

> **Important**: Never commit your `.env` file with real API keys. The `.env` file is already in `.gitignore`.

---

## Option 1: NVIDIA API (Default - Recommended for Getting Started)

The simplest option - uses NVIDIA's cloud endpoints for embeddings and reranking. No GPU required.

### Quick Start

1. Ensure your `.env` file contains a valid `NVIDIA_API_KEY`

2. Run with Docker Compose:
```bash
cd source/mcp
docker compose -f docker-compose.ngc.yaml up --build
```

That's it! The servers will automatically use NVIDIA API endpoints for embeddings and reranking.

### Run a Single Server

```bash
docker compose -f docker-compose.ngc.yaml up usd-code-mcp --build
```

### Pros & Cons

✅ No GPU required  
✅ Simplest setup  
✅ Always up-to-date models  
❌ Requires internet connection  
❌ API rate limits apply  
❌ Queries sent to cloud  

---

## Option 2: Local NIMs (Recommended for Production)

Run NVIDIA NIM containers locally on your own GPUs. Better latency, no rate limits, data stays local.

### Prerequisites

- **2 NVIDIA GPUs** (or 1 GPU with sufficient VRAM for both NIMs)
- **NGC API Key** for pulling NIM images
- **Docker with NVIDIA Container Toolkit**
- **Python 3.11+** and **Poetry** (for building wheels from source)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Local NIM Deployment                         │
├─────────────────────────────────────────────────────────────────┤
│  GPU 0                    GPU 1                                  │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │  Embedder NIM   │     │  Reranker NIM   │                    │
│  │  Port: 8001     │     │  Port: 8002     │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       │  (shared)                                │
│  ┌────────────────────┼────────────────────┐                    │
│  ▼                    ▼                    ▼                    │
│ ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│ │ OmniUI   │    │ Kit MCP  │    │ USD Code │                   │
│ │ :9901    │    │ :9902    │    │ :9903    │                   │
│ └──────────┘    └──────────┘    └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start

1. **Ensure your `.env` file contains both keys:**
```bash
NVIDIA_API_KEY=nvapi-xxxxx
NGC_API_KEY=your_ngc_key
```

2. **Login to NGC registry:**
```bash
source .env  # Load environment variables
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
```

3. **Build wheels (required when building from source):**
```bash
cd source/mcp
./build-wheels.sh    # Linux/macOS
build-wheels.bat     # Windows
```

4. **Start all services:**
```bash
docker compose -f docker-compose.local.yaml up --build
```

5. **Services will be available at:**

| Service | Port | Description |
|---------|------|-------------|
| Embedder NIM | 8001 | `nvidia/nv-embedqa-e5-v5` |
| Reranker NIM | 8002 | `nvidia/llama-3.2-nv-rerankqa-1b-v2` |
| OmniUI MCP | 9901 | http://localhost:9901/mcp |
| Kit MCP | 9902 | http://localhost:9902/mcp |
| USD Code MCP | 9903 | http://localhost:9903/mcp |

### Run Specific Servers Only

```bash
# Just USD Code MCP with NIMs
docker compose -f docker-compose.local.yaml up embedder reranker usd-code-mcp

# Just Kit MCP with NIMs
docker compose -f docker-compose.local.yaml up embedder reranker kit-mcp
```

### Single GPU Configuration

If you only have one GPU:

1. Edit `docker-compose.local.yaml`
2. Change both services to use `device_ids: ['0']`

### Pros & Cons

✅ No API rate limits  
✅ Data stays local  
✅ Cost-effective for high volume  
❌ Requires GPU(s)  
❌ More setup complexity  

---

## Option 3: External Embedder/Reranker

Connect to your own existing embedder and reranker services (e.g., running on a different server or managed service).

### Setup

Add to your `.env` file:

```bash
# Required
NVIDIA_API_KEY=nvapi-xxxxx

# External service configuration
KIT_EMBEDDER_BACKEND=local
KIT_LOCAL_EMBEDDER_URL=http://your-embedder-host:8000
KIT_RERANKER_BACKEND=local
KIT_LOCAL_RERANKER_URL=http://your-reranker-host:8000
```

Then run the MCP server:

```bash
docker compose -f docker-compose.ngc.yaml up usd-code-mcp --build
```

### API Compatibility

Your embedder must expose `/v1/embeddings` endpoint (OpenAI-compatible):
```json
POST /v1/embeddings
{
  "input": ["text to embed"],
  "model": "nvidia/nv-embedqa-e5-v5",
  "input_type": "query"
}
```

Your reranker must expose `/v1/ranking` endpoint:
```json
POST /v1/ranking
{
  "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
  "query": {"text": "query"},
  "passages": [{"text": "passage1"}, {"text": "passage2"}]
}
```

---

## Health Checks

After starting the servers, verify they are healthy:

```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}"

# Test MCP endpoint (should return initialization response)
curl -X POST http://localhost:9903/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | NVIDIA API key (always required for LLM) | Required |
| `NGC_API_KEY` | NGC key for pulling NIM images | Required for local NIMs |
| `KIT_EMBEDDER_BACKEND` | `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_EMBEDDER_URL` | URL when backend=local | - |
| `KIT_RERANKER_BACKEND` | `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_RERANKER_URL` | URL when backend=local | - |

---

## Cursor IDE Integration

Add to your Cursor MCP settings (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "omni-ui-mcp": {
      "url": "http://localhost:9901/mcp"
    },
    "kit-mcp": {
      "url": "http://localhost:9902/mcp"
    },
    "usd-code-mcp": {
      "url": "http://localhost:9903/mcp"
    }
  }
}
```

---

## Building from Source

When cloning the repository, you need to build Python wheels before the Docker images can be built. The build scripts automate this process.

### Install Poetry

If you don't have Poetry installed:

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows
pip install poetry
```

### Build All Wheels

```bash
cd source/mcp

# Linux/macOS
./build-wheels.sh

# Windows
build-wheels.bat
```

### Build Specific Server Wheels

```bash
# Linux/macOS
./build-wheels.sh kit    # Build only Kit MCP wheels
./build-wheels.sh omni   # Build only Omni UI MCP wheels
./build-wheels.sh usd    # Build only USD Code MCP wheels

# Windows
build-wheels.bat kit
build-wheels.bat omni
build-wheels.bat usd
```

### What the Script Does

The build script:
1. Builds the `*_fns` function packages (e.g., `kit_fns`, `omni_ui_fns`, `usd_code_fns`)
2. Copies the wheels to each MCP server's `dist/` directory
3. Builds the MCP server wheels

This mirrors the CI/CD pipeline build process and ensures all dependencies are available for Docker builds.

---

## Troubleshooting

### NIM containers fail to start

```
nvidia-container-cli: initialization error
```

Install NVIDIA Container Toolkit:
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Containers show unhealthy status

NIMs take 1-2 minutes to load models. Check logs:
```bash
docker logs mcp-embedder
docker logs mcp-reranker
```

### GPU memory issues

```
CUDA out of memory
```

- Check no other processes using GPU: `nvidia-smi`
- Try using 2 GPUs (one per NIM) instead of single GPU mode

### Connection refused

MCP server can't connect to embedder/reranker:
- Ensure NIMs are healthy before MCP starts
- Check services are on same Docker network
- Use container names in URLs (e.g., `http://embedder:8000`)

### Environment variables not loaded

If you see warnings about missing environment variables:
```
The "NVIDIA_API_KEY" variable is not set. Defaulting to a blank string.
```

Ensure you:
1. Have a `.env` file in the `source/mcp` directory
2. Are running `docker compose` from the `source/mcp` directory
3. The `.env` file contains valid values (no quotes around values needed)

---

## Stopping Services

```bash
docker compose -f docker-compose.local.yaml down
# or
docker compose -f docker-compose.ngc.yaml down
```
