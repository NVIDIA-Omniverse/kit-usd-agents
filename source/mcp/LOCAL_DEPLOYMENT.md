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

### Git LFS

The `*_fns/data/` subtrees (FAISS indices, knowledge bundles, extension metadata) are tracked via Git LFS. Without LFS resolved, your checkout will contain ~135-byte pointer stubs in place of the real files, and `build-wheels.sh` would otherwise build a wheel that's roughly 13× smaller than expected — the MCP container would start fine, but every search tool would fail at first invocation with `Extension data is not available`.

Only one thing is required from you: **install the `git-lfs` binary**:

```bash
sudo apt-get install git-lfs   # Ubuntu/Debian
# or: brew install git-lfs     # macOS
```

`./build-wheels.sh` detects LFS pointer stubs in the data subtrees on every run and **auto-recovers** by invoking `git lfs install --local && git lfs pull` for you. You'll see a one-time `Detected LFS pointer stubs … Attempting auto-recovery` message during the first build; subsequent builds say `Git LFS check: data files resolved` and proceed. Auto-recovery only fails hard if the `git-lfs` binary is missing, the remote LFS objects are unreachable, or the working tree isn't a git checkout (e.g. a tarball download).

### Environment Setup

1. **Copy the environment template:**
```bash
cd source/mcp
cp .env.example .env
```

2. **Edit `.env` and add your API keys:**
```bash
# Required for all deployment options
NVIDIA_API_KEY=REPLACE_WITH_NVIDIA_API_KEY

# Required only for Local NIMs deployment (Option 2)
NGC_API_KEY=REPLACE_WITH_NGC_API_KEY
```

> **Important**: Never commit your `.env` file with real API keys. The `.env` file is already in `.gitignore`.

---

## Option 1: NVIDIA API (Default - Recommended for Getting Started)

The simplest option - uses NVIDIA's cloud endpoints for embeddings and reranking. No GPU required.

### Quick Start

1. Ensure your `.env` file contains a valid `NVIDIA_API_KEY`

2. **Build Python wheels** (required — Dockerfiles need pre-built `.whl` files in each server's `dist/` directory):
```bash
cd source/mcp
./build-wheels.sh    # Linux/macOS
build-wheels.bat     # Windows
```

3. Run with Docker Compose:
```bash
docker compose -f docker-compose.ngc.yaml up --build
```

That's it! The servers will automatically use NVIDIA API endpoints for embeddings and reranking.

> **Note**: Skipping step 2 causes `COPY dist/*.whl` to fail for every service, even if you only intend to run one of them — Docker Compose builds all services in parallel by default.

### Run a Single Server

Use the service name as defined in `docker-compose.ngc.yaml`:

| Service | Name |
|---------|------|
| OmniUI MCP | `omni-ui-mcp` |
| Kit MCP | `kit-mcp` |
| USD Code MCP | `usd-code-mcp` |
| Isaac Sim MCP | `isaacsim-mcp` |

```bash
# Build wheels for one server only (faster)
./build-wheels.sh isaac   # or: kit, omni, usd

# Then start just that server
docker compose -f docker-compose.ngc.yaml up isaacsim-mcp --build
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
│                     Local NIM Deployment                        │
├─────────────────────────────────────────────────────────────────┤
│  GPU 0                    GPU 1                                 │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │  Embedder NIM   │     │  Reranker NIM   │                    │
│  │  Port: 8001     │     │  Port: 8002     │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                             |
│           └───────────┬───────────┘                             │
│                       │  (shared)                               │
│  ┌─────────────────────┼─────────────────────┐                  │
│  ▼                     ▼                     ▼                  │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│ │ OmniUI   │ │ Kit MCP  │ │ USD Code │ │ Isaac Sim│             │
│ │ :9901    │ │ :9902    │ │ :9903    │ │ :9904    │             │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start

1. **Ensure your `.env` file contains both keys:**
```bash
NVIDIA_API_KEY=REPLACE_WITH_NVIDIA_API_KEY
NGC_API_KEY=REPLACE_WITH_NGC_API_KEY
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
| Embedder NIM | 8001 | `nvidia/nemotron-3-embed-1b` |
| Reranker NIM | 8002 | `nvidia/llama-nemotron-rerank-vl-1b-v2` |
| OmniUI MCP | 9901 | http://localhost:9901/mcp |
| Kit MCP | 9902 | http://localhost:9902/mcp |
| USD Code MCP | 9903 | http://localhost:9903/mcp |
| Isaac Sim MCP | 9904 | http://localhost:9904/mcp |

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

Connect to embedder and reranker services that are already running somewhere — either on the same machine (in a separate Docker Compose project) or on a different host in your network.

> **When to use**: You already have NIM containers running from a previous Option 2 setup, or your team has shared embedder/reranker instances. This avoids starting duplicate GPU workloads.

### How the URLs work

The MCP servers run **inside Docker containers**, so `localhost` inside a container refers to itself — not your host machine. Use one of these depending on where your NIMs are running:

| NIMs location | Use this URL |
|---|---|
| Same host, started via `docker-compose.local.yaml` | `http://host.docker.internal:8001` (Mac/Windows) or `http://172.17.0.1:8001` (Linux) |
| Different machine on the network | `http://<that-machine-ip>:8001` |
| Inside the same Docker Compose project | `http://embedder:8000` (Docker service name) |

The NIM containers from Option 2 expose port **8001** for the embedder and **8002** for the reranker on the host (they listen on 8000 internally).

### Scenario A: NIMs already running on the same machine (Linux)

First, verify the NIMs are up and reachable from the host:
```bash
curl http://localhost:8001/v1/health/ready   # embedder
curl http://localhost:8002/v1/health/ready   # reranker
```

Find your host's Docker bridge IP (the address containers use to reach the host):
```bash
ip route show default | awk '/default/ {print $3}'
# Typically outputs: 172.17.0.1
```

Add to your `.env`:
```bash
NVIDIA_API_KEY=REPLACE_WITH_NVIDIA_API_KEY

KIT_EMBEDDER_BACKEND=local
KIT_LOCAL_EMBEDDER_URL=http://172.17.0.1:8001

KIT_RERANKER_BACKEND=local
KIT_LOCAL_RERANKER_URL=http://172.17.0.1:8002
```

### Scenario B: NIMs running on a different machine in the network

Add to your `.env`, substituting the actual IP of the machine running the NIMs:
```bash
NVIDIA_API_KEY=REPLACE_WITH_NVIDIA_API_KEY

KIT_EMBEDDER_BACKEND=local
KIT_LOCAL_EMBEDDER_URL=http://192.168.1.50:8001

KIT_RERANKER_BACKEND=local
KIT_LOCAL_RERANKER_URL=http://192.168.1.50:8002
```

Verify reachability before starting:
```bash
curl http://192.168.1.50:8001/v1/health/ready
curl http://192.168.1.50:8002/v1/health/ready
```

### Start the MCP server

After setting the URLs in `.env`, build wheels if not already done, then start any MCP server:

```bash
./build-wheels.sh isaac   # or whichever server you're testing
docker compose -f docker-compose.ngc.yaml up isaacsim-mcp --build
```

### Verify the connection

Once the MCP container is running, confirm it picked up the external NIM configuration:
```bash
# Check the eager startup banner (printed BEFORE any tool call)
docker logs isaacsim-mcp 2>&1 | grep "\[mcp-startup\]"

# Check the server started cleanly (last few lines should show port listening)
docker logs isaacsim-mcp 2>&1 | tail -20
```

**What success looks like** — every MCP prints a `[mcp-startup]` banner block when
the container boots, so you can verify backend selection IMMEDIATELY after
`docker compose up` without needing to invoke a tool first:

```
==========================================================================
[mcp-startup] Embedder backend: local
[mcp-startup] Using local embedder at http://<system_1_ip>:8001
[mcp-startup] Reranker backend: local
[mcp-startup] Using local reranker at http://<system_1_ip>:8002
==========================================================================
```

For `isaacsim-mcp` the banner notes that only the embedder is exercised today:
```
==========================================================================
[mcp-startup] Embedder backend: local
[mcp-startup] Using local embedder at http://<system_1_ip>:8001
[mcp-startup] NOTE: isaacsim-mcp uses the embedder only — the reranker
[mcp-startup]       env vars are accepted but not exercised by current tools.
==========================================================================
```

After Cursor (or any MCP client) invokes its first tool, the actual factory
also logs `Creating embedder with backend: local` / `Using local embedder at ...`
— but those lines are lazy and won't appear until a tool fires. The
`[mcp-startup]` banner is the canonical "did my .env get through?" signal.
For local-NIM reranking, the hybrid retrieval shim may instead log `Rerank routed to local NIM at ...`.

**Red flags:**
- `[mcp-startup] Embedder backend: nvidia_api` when you set `KIT_EMBEDDER_BACKEND=local` in `.env` — the `.env` wasn't picked up; the container is silently using the NVIDIA cloud. Confirm with `docker exec isaacsim-mcp env | grep KIT_`. Most common cause: `.env` placed in the wrong directory. Compose looks for `.env` in the directory containing `docker-compose.ngc.yaml` (i.e., `source/mcp/.env`), not in `source/mcp/isaacsim_mcp/.env`.
- `[mcp-startup] Embedder backend: local` shown but no URL line ("Using local embedder at ...") — `KIT_EMBEDDER_BACKEND` was set but `KIT_LOCAL_EMBEDDER_URL` was not. Tools will fail at first call with `Local embedder URL must be provided via KIT_LOCAL_EMBEDDER_URL environment variable`. Set the URL in `.env`.
- `Failed to embed documents via local API` / `Failed to embed query via local API` (appears later, on first tool call) — backend was selected correctly but the NIM is unreachable from inside the container. Re-test reachability with `docker exec isaacsim-mcp curl -f http://<system_1_ip>:8001/v1/health/ready`.

> `ModuleNotFoundError: No module named 'nat.plugins.opentelemetry'` is benign and unrelated to embedder/reranker selection.

### API Compatibility

Your embedder must expose `/v1/embeddings` (OpenAI-compatible):
```json
POST /v1/embeddings
{
  "input": ["text to embed"],
  "model": "nvidia/nemotron-3-embed-1b",
  "input_type": "query"
}
```

Your reranker must expose `/v1/ranking`:
```json
POST /v1/ranking
{
  "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
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
curl -X POST http://localhost:9903/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

---

## Environment Variables Reference

### Embedder / reranker NIM selection

These govern which embedding / reranking service the MCP servers talk to. Used by
both the legacy semantic-only path and the hybrid retrieval path.

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | NVIDIA API key (always required for LLM) | Required |
| `NGC_API_KEY` | NGC key for pulling NIM images | Required for local NIMs |
| `KIT_EMBEDDER_BACKEND` | `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_EMBEDDER_URL` | URL when backend=local | — |
| `KIT_RERANKER_BACKEND` | `nvidia_api` or `local` | `nvidia_api` |
| `KIT_LOCAL_RERANKER_URL` | URL when backend=local | — |

### Hybrid retrieval (ovgenai-retrieval) tuning

Hybrid mode is the default starting with the `ovgenai-retrieval` migration. These
variables route through `ovgenai_retrieval.shim.maybe_load_hybrid` so a change here
takes effect across all four MCPs without a per-package code edit.

| Variable | Description | Default |
|----------|-------------|---------|
| `OVAI_RETRIEVAL_MODE` | `hybrid` or `semantic` (fall back to legacy LangChain FAISS) | `hybrid` |
| `OVAI_FUSION` | `rrf` \| `semantic_only` \| `lexical_only` | `rrf` |
| `OVAI_BM25_BACKEND` | `rank_bm25` \| `tantivy` \| `whoosh` | `rank_bm25` |
| `OVAI_RERANK` | Enable a second-stage reranker on top of fusion (`true`/`false`). When unset, the legacy `*_RERANKER_BACKEND` env vars below act as a fallback signal — set one of them (e.g. `KIT_RERANKER_BACKEND=local`) and reranking is on. Set `OVAI_RERANK=false` explicitly to force-off even when a legacy var is set. | `false`, **but defaults to `true` if any legacy `*_RERANKER_BACKEND` is non-empty** |
| `OVAI_RERANK_BACKEND` | `cross_encoder` (local sentence-transformers) or `nvidia` (hosted endpoint or local NIM via `*_LOCAL_RERANKER_URL`, uses `langchain-nvidia-ai-endpoints`). When unset, legacy `*_RERANKER_BACKEND=local` plus a local reranker URL routes to `nvidia` with that base URL; legacy `=local` without a URL falls back to `cross_encoder`; legacy `=nvidia_api` maps to `nvidia`. | `cross_encoder` |
| `OVAI_RERANK_MODEL` | Reranker model. `NvidiaReranker` ignores values that don't start with `nvidia/` and uses its own default (`nvidia/llama-nemotron-rerank-vl-1b-v2`). Set a current `nvidia/...` value here when pinning a hosted model or matching a local NIM image. | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `OVAI_QE_DOMAINS` | Comma-separated glossary categories that scope query expansion. Overrides per-MCP defaults (kit_mcp: `kit,ui,rendering,general`; isaacsim_mcp: `physics,sensors,ai,general`; usd_code_mcp: `usd_modules,geometry,general`; omni_ui_mcp: `ui,kit,general`). Set empty to disable. | per-package |

To exercise the NVIDIA hosted reranker on any MCP, set a current rerank model
in your `.env`:
```bash
OVAI_RERANK=true
OVAI_RERANK_BACKEND=nvidia
OVAI_RERANK_MODEL=nvidia/llama-nemotron-rerank-vl-1b-v2
```

> **NVIDIA hosted rerank model EOL (2026-05-18)**
>
> NVIDIA retired `nvidia/llama-3.2-nv-rerankqa-1b-v2` on 2026-05-18. Older
> configs/images that still pin that model with `OVAI_RERANK=true` +
> `OVAI_RERANK_BACKEND=nvidia` will hit:
> ```
> Exception: [410] Gone
> This endpoint has reached its end of life on 2026-05-18T00:00:00Z and is
> no longer available.
> ```
> Mitigations (pick one):
> 1. **Pin a replacement model** in `.env`. Check
>    [build.nvidia.com/explore/retrieval](https://build.nvidia.com/explore/retrieval)
>    for a current `nvidia/*rerank*` model,
>    then set:
>    ```bash
>    OVAI_RERANK_MODEL=nvidia/llama-nemotron-rerank-vl-1b-v2
>    ```
> 2. **Disable rerank** until the replacement is wired up — embedder-only hybrid
>    (RRF fusion of BM25 + dense) still produces good results:
>    ```bash
>    OVAI_RERANK=false
>    ```
> 3. **Run the rerank NIM locally** (`docker-compose.local.yaml` Option 2) and
>    point at it via `KIT_LOCAL_RERANKER_URL=http://<nim-host>:8002` — the shim
>    routes that through `NvidiaReranker(base_url=...)`, so the EOL of the
>    hosted endpoint doesn't affect on-prem deployments.
>
> Verified on 2026-05-18: with `OVAI_RERANK=false`, the live MCP smoke
> (`source/mcp/mcp_tool_smoke.py`) returns 102 PASS / 0 EMPTY / 0 ERROR across
> all four non-MaaS MCPs.

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
    },
    "isaacsim-mcp": {
      "url": "http://localhost:9904/mcp"
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
./build-wheels.sh isaac  # Build only Isaac Sim MCP wheels

# Windows
build-wheels.bat kit
build-wheels.bat omni
build-wheels.bat usd
build-wheels.bat isaac
```

### What the Script Does

The build script:
1. Builds the `*_fns` function packages (e.g., `kit_fns`, `omni_ui_fns`, `usd_code_fns`, `isaacsim_fns`)
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

### Container process dies mid-build (`runc run failed: container process is already dead`)

This typically means the container was killed by the OS (OOM) during `pip`/`uv` install. The MCP servers have large dependency sets (litellm, ragas, faiss-cpu, etc.).

- **Increase Docker memory**: Docker Desktop → Settings → Resources → Memory (≥8 GB recommended)
- **Retry**: `docker compose -f docker-compose.ngc.yaml up isaacsim-mcp --build --no-cache`
- **Check available RAM**: `free -h` — ensure at least 4 GB free before building

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
