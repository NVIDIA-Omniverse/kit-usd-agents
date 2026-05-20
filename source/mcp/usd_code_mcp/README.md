# USD Code MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives AI coding assistants deep knowledge of the USD/OpenUSD API — modules, classes, methods, and code examples — powered by semantic search and NVIDIA AI reranking.

Built on the [NVIDIA AIQ Toolkit](https://github.com/NVIDIA/GenerativeAIExamples) (formerly NeMo Agent Toolkit / NAT 1.3+).

---

## 5-Minute Quickstart

Get from zero to working USD tools in your IDE. Follow every step in order.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- An NVIDIA API key (see Step 1 below)
- [Git LFS](https://git-lfs.com/) installed — the FAISS indices and USD knowledge data under `source/aiq/*/data/` are LFS-tracked. `build-wheels.sh` auto-runs `git lfs install --local && git lfs pull` for you on the first build, so you only need the binary on PATH (`sudo apt-get install git-lfs` or `brew install git-lfs`). Without LFS, the wheel ends up ~13× smaller and the container silently fails at first tool call with `Extension data is not available` — auto-recovery in `build-wheels.sh` catches this.
- The repo cloned: `git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git`

### Step 1: Get Your API Key

You need **one** key to use the cloud deployment (recommended for getting started).

| Key | What It's For | Where to Get It |
|-----|---------------|-----------------|
| `NVIDIA_API_KEY` | Authenticates calls to NVIDIA's cloud endpoints for embeddings, reranking, and LLM inference | [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) — sign in, click **Generate API Key**, copy the `nvapi-...` value |

> **Note:** A second key (`NGC_API_KEY` from [org.ngc.nvidia.com/setup/api-key](https://org.ngc.nvidia.com/setup/api-key)) is only required if you plan to run embedder/reranker models locally via NVIDIA NIM containers — see [Deployment Options](#deployment-options) below. The cloud quickstart needs only `NVIDIA_API_KEY`.

### Step 2: Configure Your `.env`

```bash
cd kit-usd-agents/source/mcp

# Create your .env file from the template
cp .env.example .env
```

Open `.env` and set:

```env
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
```

### Step 3: Build and Run the Docker Container

```bash
cd usd_code_mcp

# Build the image
./build-docker.sh        # Linux/macOS
# build-docker.bat       # Windows

# Run the server (note --env-file points to ../.env, one level up)
docker run --rm -p 9903:9903 --env-file ../.env usd-code-mcp:latest
```

> **`.env` location matters.** The `--env-file ../.env` flag is relative to your current directory. Run the `docker run` command from `source/mcp/usd_code_mcp/` (where you `cd`-ed in this step) so `../.env` resolves to `source/mcp/.env`. If you `cd` somewhere else, use an absolute path: `--env-file "$(git rev-parse --show-toplevel)/source/mcp/.env"`.

### Step 4: Verify the Server is Running

The server speaks Streamable HTTP at `/mcp` (the canonical endpoint in NAT 1.25). There is **no separate `/health` GET endpoint**; the canonical liveness probe is an MCP `initialize` POST.

> **Trailing slash:** older NAT 1.3 builds required `/mcp/`. NAT 1.25 returns `307 Temporary Redirect` from `/mcp/` to `/mcp`, so both work — but using `/mcp` directly avoids a redirect on every call. If you keep the trailing slash, pass `-L` to curl.

In a new terminal:

```bash
# Easiest: use the included Python health check (already in the repo)
python check_mcp_health.py

# Or curl the MCP endpoint directly. The Accept header is required —
# NAT 1.4 streams the response as text/event-stream and returns 406
# without it.
curl -s -X POST http://localhost:9903/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"health","version":"1.0"}}}'
```

A healthy server returns a JSON-RPC `result` payload listing the server's name and capabilities.

> **Troubleshooting:** If you get `connection refused`, make sure the Docker container is still running (`docker ps`). If the response has a JSON-RPC `error` mentioning auth, double-check `NVIDIA_API_KEY` in `.env`.

### Step 5: Connect Your IDE

Pick your IDE; configs all point at the same URL: `http://localhost:9903/mcp`.

> **MCP scoping note (Claude Code):** `claude mcp add ... -t http <url>` writes to `.claude.json` in your **current working directory**. The MCP is then only active when you launch Claude CLI from that directory. To register globally, pass `--scope user`. This same caveat applies to all the MCPs in this repo.

<details>
<summary><strong>Cursor</strong></summary>

Create `.cursor/mcp.json` in your project root (or `~/.cursor/mcp.json` for global access):

```json
{
  "mcpServers": {
    "usd-code-mcp": {
      "url": "http://localhost:9903/mcp"
    }
  }
}
```

Reload Cursor: `Cmd/Ctrl+Shift+P` → **Developer: Reload Window**

</details>

<details>
<summary><strong>Claude Code</strong></summary>

Add the MCP server via the CLI (project scope):

```bash
claude mcp add usd-code-mcp -t http http://localhost:9903/mcp
```

For user (global) scope:

```bash
claude mcp add usd-code-mcp --scope user -t http http://localhost:9903/mcp
```

Or add it directly to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "usd-code-mcp": {
      "type": "http",
      "url": "http://localhost:9903/mcp"
    }
  }
}
```

</details>

<details>
<summary><strong>Windsurf</strong></summary>

Create `~/.windsurf/mcp.json` (or in your project's `.windsurf/` folder):

```json
{
  "mcpServers": {
    "usd-code-mcp": {
      "url": "http://localhost:9903/mcp"
    }
  }
}
```

Restart Windsurf to pick up the new server.

</details>

<details>
<summary><strong>VS Code (Copilot)</strong></summary>

Add to your `.vscode/mcp.json`:

```json
{
  "servers": {
    "usd-code-mcp": {
      "type": "http",
      "url": "http://localhost:9903/mcp"
    }
  }
}
```

</details>

### Step 6: Verify Tools Appear

In your IDE's AI chat, you should see **7 USD tools** available:

| Tool | Description |
|------|-------------|
| `list_usd_modules` | Lists all available USD module names |
| `list_usd_classes` | Returns USD class names, optionally filtered by module |
| `get_usd_module_detail` | Provides detailed information about a specific USD module |
| `get_usd_class_detail` | Returns comprehensive details about a specific USD class |
| `get_usd_method_detail` | Returns method signatures and documentation |
| `search_usd_code_examples` | Retrieves relevant USD code examples via semantic search |
| `search_usd_knowledge` | Searches the USD knowledge base for conceptual information |

Try asking your AI assistant: *"List all USD modules related to geometry"* — if you get results, you're all set.

---

## Architecture Overview

```
┌──────────────────────┐
│  Your IDE             │
│  (Cursor / Claude     │
│   Code / Windsurf /   │
│   VS Code Copilot)    │
└────────┬─────────────┘
         │ MCP (Streamable HTTP, POST /mcp/)
         ▼
┌──────────────────────┐
│  USD Code MCP Server  │ ← This package
│  (port 9903)          │
└────────┬─────────────┘
         │ AIQ Workflow
         ▼
┌──────────────────────┐
│  RAG Pipeline         │
│  ┌────────────────┐  │
│  │ Embedder       │  │  ← NVIDIA cloud or local NIM
│  │ Reranker       │  │  ← NVIDIA cloud or local NIM
│  └────────────────┘  │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  USD Atlas Database   │  ← Modules, classes, methods,
│                       │    code examples
└──────────────────────┘
```

The server is built on **NVIDIA AIQ Toolkit** (formerly NeMo Agent Toolkit / NAT), which handles workflow orchestration and the plugin system. As an MCP consumer, you don't need to understand AIQ internals — it manages the RAG (Retrieval-Augmented Generation) pipeline that powers the 7 USD tools. Workflow YAMLs (`workflow/config.yaml`, `workflow/local_config.yaml`) wire the embedder, reranker, and USD Atlas database into the MCP tool surface.

---

## Deployment Options

### Option A: Cloud Endpoints (Recommended)

Uses NVIDIA's hosted cloud endpoints for embeddings and reranking. This is what the quickstart above uses — best for getting started quickly.

**Required environment variables:**

```env
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
```

### Option B: Local NIM Containers (Advanced, GPU required)

Runs embedder and reranker models locally using NVIDIA NIM containers. Better for production, data privacy, and avoiding cloud rate limits.

**Required environment variables:**

```env
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
NGC_API_KEY=your_ngc_key_here
KIT_EMBEDDER_BACKEND=local
KIT_LOCAL_EMBEDDER_URL=http://localhost:8080
KIT_RERANKER_BACKEND=local
KIT_LOCAL_RERANKER_URL=http://localhost:8081
```

> **Where to get your NGC API key:** Sign in at [org.ngc.nvidia.com/setup/api-key](https://org.ngc.nvidia.com/setup/api-key) → **Generate API Key**. This key authenticates access to NVIDIA's NGC container registry for pulling NIM model images.

For the full local-NIM setup including `docker login nvcr.io`, wheel building, and the `docker-compose` flow that brings up MCP + embedder + reranker together, see [`source/mcp/LOCAL_DEPLOYMENT.md`](../LOCAL_DEPLOYMENT.md). Linked NIM docs: [docs.nvidia.com/nim](https://docs.nvidia.com/nim/).

---

## Project Structure

```
source/mcp/usd_code_mcp/
├── VERSION.md                 # Version information
├── README.md                  # This file
├── pyproject.toml             # Poetry configuration and dependencies
├── Dockerfile                 # Docker image configuration
├── check_mcp_health.py        # MCP-initialize-based health probe
├── setup-dev.sh / .bat        # Development environment setup
├── run.sh / run.bat           # Local run scripts (non-Docker)
├── build-docker.sh / .bat     # Docker image build scripts
├── workflow/
│   ├── config.yaml            # AIQ workflow config (cloud endpoints)
│   └── local_config.yaml      # AIQ workflow config (local NIM)
└── src/
    └── usd_code_mcp/          # Server source code
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `connection refused` on port 9903 | Docker container not running | `docker ps` to check; restart the container if needed |
| `404` on `GET /health` | No `/health` GET endpoint exists in this server | Use `python check_mcp_health.py` or POST an MCP `initialize` to `/mcp` (see Step 4) |
| `307 Temporary Redirect` on POST `/mcp/` | NAT 1.25 canonicalises to `/mcp`. `curl -f` (without `-L`) treats 307 as success, so a healthcheck never exercises the endpoint. | Drop the trailing slash, or pass `-L` to curl. The repo's Dockerfile and compose healthchecks both use `curl -fL ... /mcp`. |
| `401 Unauthorized` / auth error from cloud calls | Invalid or expired `NVIDIA_API_KEY` | Regenerate at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) and update `.env` |
| `--env-file: file not found` | Wrong cwd when invoking `docker run` | Run from `source/mcp/usd_code_mcp/`, or use absolute path: `--env-file "$(git rev-parse --show-toplevel)/source/mcp/.env"` |
| Docker build fails pulling base image | Network/proxy issues | Check connectivity; configure Docker proxy if behind corporate firewall |
| Port 9903 already in use | Another process on that port | `lsof -i :9903` (Linux/macOS) or `netstat -aon \| findstr 9903` (Windows); stop the process or remap: `-p 9904:9903` |
| Tools not appearing in IDE | MCP config not loaded or wrong URL | Verify with `check_mcp_health.py`, check IDE's MCP config URL (use `/mcp` — trailing slash works too via 307 redirect), reload the IDE |
| Claude CLI: `usd-code-mcp` missing from `claude mcp list` | `-t http` registered the MCP at project scope (writes `.claude.json` in cwd) | Re-add with `--scope user` for global, or always launch Claude CLI from the project root where you registered |
| Rate limiting errors from cloud endpoints | Heavy usage | Wait and retry; for sustained load, switch to local NIM (Option B) |

---

## Development

### Running Locally (Without Docker)

```bash
# Set up the development environment
./setup-dev.sh        # Linux/macOS
# setup-dev.bat       # Windows

# Run the server
./run.sh              # Linux/macOS
# run.bat             # Windows
```

### Configuration Files

- `workflow/config.yaml` — Cloud-endpoint workflow
- `workflow/local_config.yaml` — Local-NIM workflow

### `.kit`-config-file path for headless dev

Working over SSH or in a CI environment without a desktop session? The MCP itself is purely an HTTP server — `./run.sh` works headlessly. For Kit-app-side extension installation that this MCP helps users discover, the GUI flow (Window → Extensions → search → install → AUTOLOAD) is not the only option: extensions can be added directly to your `.kit` config file under `[dependencies]`. Example:

```toml
[dependencies]
"omni.physxclashdetection.bundle" = { version = "110.1.7" }
```

The next `./repo.sh build` will pull the bundle into `extscache` automatically. This is the documented developer-path alternative for headless / SSH workflows.

---

## License

See the [LICENSE](../../../LICENSE) file in the root of this repository.
