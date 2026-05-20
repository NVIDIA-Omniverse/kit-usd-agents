# Isaac Sim MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives AI coding assistants deep knowledge of NVIDIA Isaac Sim — extensions, code examples, settings, and developer instructions — powered by semantic search and NVIDIA AI reranking.

Built on the [NVIDIA AIQ Toolkit](https://github.com/NVIDIA/GenerativeAIExamples) (formerly NeMo Agent Toolkit / NAT 1.3+).

---

## 5-Minute Quickstart

Get from zero to working Isaac Sim tools in your IDE. Follow every step in order.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Poetry](https://python-poetry.org/docs/#installation) — `build-docker.sh` builds the AIQ + MCP wheels with poetry before `docker build`
- Python **3.11 – 3.13** on the host. `pyproject.toml` requires `>=3.11,<3.14`; `poetry build` aborts on Python 3.10 with *"Current Python version (3.10.x) is not allowed by the project"*. If your default `python3` is 3.10, point poetry at a newer interpreter in **both** project dirs:
  ```bash
  cd source/aiq/isaacsim_fns      && poetry env use python3.12
  cd source/mcp/isaacsim_mcp      && poetry env use python3.12
  ```
- An NVIDIA API key (see Step 1 below)
- [Git LFS](https://git-lfs.com/) installed — the Isaac Sim extension metadata and FAISS indices under `source/aiq/isaacsim_fns/src/isaacsim_fns/data/` are LFS-tracked. `build-wheels.sh` auto-runs `git lfs install --local && git lfs pull` for you on the first build, so you only need the binary on PATH (`sudo apt-get install git-lfs` or `brew install git-lfs`). Without LFS, the wheel ends up ~13× smaller and the container silently fails at first tool call with `Extension data is not available` — auto-recovery in `build-wheels.sh` catches this.
- The repo cloned: `git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git` — `build-docker.sh` expects `source/aiq/isaacsim_fns/` to exist next to `source/mcp/isaacsim_mcp/`, so a sparse-checkout of just the MCP subtree won't build.

### Step 1: Get Your API Key

| Key | What It's For | Where to Get It |
|-----|---------------|-----------------|
| `NVIDIA_API_KEY` | Authenticates calls to NVIDIA's cloud endpoints for embeddings, reranking, and LLM inference | [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) — sign in, click **Generate API Key**, copy the `nvapi-...` value |

> **Note:** A second key (`NGC_API_KEY` from [org.ngc.nvidia.com/setup/api-key](https://org.ngc.nvidia.com/setup/api-key)) is only required for local NIM deployment — see [Deployment Options](#deployment-options).

### Step 2: Configure Your `.env`

```bash
cd kit-usd-agents/source/mcp
cp .env.example .env
```

Open `.env` and set:

```env
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
```

### Step 3: Build and Run the Docker Container

```bash
cd isaacsim_mcp

# Build the image (~10–15 min cold, produces a ~1.35 GB image)
./build-docker.sh        # Linux/macOS
# build-docker.bat       # Windows

# Run the server (foreground, auto-removed on Ctrl-C)
docker run --rm -p 9904:9904 --env-file ../.env isaacsim-mcp:latest

# Or run as a managed service so you can stop/restart later:
# docker run -d --name isaacsim-mcp -p 9904:9904 --env-file ../.env isaacsim-mcp:latest
# docker logs -f isaacsim-mcp
# docker stop isaacsim-mcp && docker rm isaacsim-mcp
```

> **`.env` location matters.** `--env-file ../.env` resolves relative to your current directory. Run `docker run` from `source/mcp/isaacsim_mcp/`, or use the absolute path: `--env-file "$(git rev-parse --show-toplevel)/source/mcp/.env"`.

### Step 4: Verify the Server is Running

The server speaks Streamable HTTP at `/mcp` (the canonical endpoint in NAT 1.25). There is **no separate `/health` GET endpoint**; the canonical liveness probe is an MCP `initialize` POST.

> **Trailing slash:** older NAT 1.3 builds required `/mcp/`. NAT 1.25 returns `307 Temporary Redirect` from `/mcp/` to `/mcp`, so both work — but using `/mcp` directly avoids a redirect on every call. If you keep the trailing slash, pass `-L` to curl.

```bash
# Easiest: use the included Python health check.
# Needs aiohttp on the host (`pip install aiohttp`); or run it inside the container:
#   docker exec isaacsim-mcp python /app/check_mcp_health.py
python check_mcp_health.py

# Or curl the MCP endpoint directly. The Accept header is required —
# the server replies with text/event-stream:
curl -s -X POST http://localhost:9904/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"health","version":"1.0"}}}'
```

A healthy server returns a JSON-RPC `result` payload listing the server's name and capabilities.

### Step 5: Connect Your IDE

All IDE configs point at the same URL: `http://localhost:9904/mcp`.

> **Name collision:** if you previously had an `isaac-sim-mcp` registered against a hosted endpoint (e.g. `http://isaac-sim-mcp.nvidia.com/mcp`), `claude mcp add isaac-sim-mcp` will refuse / clash. List with `claude mcp list | grep isaac-sim-mcp`, then remove the prior entry: `claude mcp remove isaac-sim-mcp -s <local|user|project>`.

> **MCP scoping note (Claude Code):** `claude mcp add ... -t http <url>` writes to `.claude.json` in your **current working directory**. The MCP is then only active when you launch Claude CLI from that directory. To register globally, pass `--scope user`.

<details>
<summary><strong>Cursor</strong></summary>

```json
{
  "mcpServers": {
    "isaac-sim-mcp": {
      "url": "http://localhost:9904/mcp"
    }
  }
}
```

Reload Cursor: `Cmd/Ctrl+Shift+P` → **Developer: Reload Window**

</details>

<details>
<summary><strong>Claude Code</strong></summary>

```bash
# Project scope
claude mcp add isaac-sim-mcp -t http http://localhost:9904/mcp

# User (global) scope
claude mcp add isaac-sim-mcp --scope user -t http http://localhost:9904/mcp
```

Or add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "isaac-sim-mcp": {
      "type": "http",
      "url": "http://localhost:9904/mcp"
    }
  }
}
```

</details>

<details>
<summary><strong>Windsurf</strong></summary>

```json
{
  "mcpServers": {
    "isaac-sim-mcp": {
      "url": "http://localhost:9904/mcp"
    }
  }
}
```

</details>

<details>
<summary><strong>VS Code (Copilot)</strong></summary>

```json
{
  "servers": {
    "isaac-sim-mcp": {
      "type": "http",
      "url": "http://localhost:9904/mcp"
    }
  }
}
```

</details>

### Step 6: Verify Tools Appear

In your IDE's AI chat, you should see **5 Isaac Sim tools**:

| Tool | Description |
|------|-------------|
| `get_isaac_sim_instructions` | Top-level guidance on Isaac Sim development |
| `search_isaac_sim_extensions` | Semantic search across the indexed Isaac Sim extension catalog |
| `get_isaac_sim_extension_details` | Full details for one or more extensions |
| `search_isaac_sim_code_examples` | Find Isaac Sim code patterns by description |
| `search_isaac_sim_settings` | Find an Isaac Sim setting by name or purpose |

Try asking: *"Find me an Isaac Sim extension for cameras"* — if you get the `isaacsim.sensors.camera` family back, the index is intact.

---

## Architecture Overview

```
┌──────────────────────┐
│  Your IDE             │
│  (Cursor / Claude     │
│   Code / Windsurf /   │
│   VS Code Copilot)    │
└────────┬─────────────┘
         │ MCP (Streamable HTTP, POST /mcp)
         ▼
┌──────────────────────┐
│  Isaac Sim MCP Server │ ← This package
│  (port 9904)          │
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
│  Isaac Sim Atlas DB   │  ← Extensions, code examples,
│                       │    settings, instructions
└──────────────────────┘
```

---

## Deployment Options

### Option A: Cloud Endpoints (Recommended)

```env
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
```

### Option B: Local NIM Containers (Advanced, GPU required)

```env
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
NGC_API_KEY=your_ngc_key_here
KIT_EMBEDDER_BACKEND=local
KIT_LOCAL_EMBEDDER_URL=http://localhost:8080
KIT_RERANKER_BACKEND=local
KIT_LOCAL_RERANKER_URL=http://localhost:8081
```

> NGC API key: [org.ngc.nvidia.com/setup/api-key](https://org.ngc.nvidia.com/setup/api-key). Full local-NIM setup is in [`source/mcp/LOCAL_DEPLOYMENT.md`](../LOCAL_DEPLOYMENT.md).

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `connection refused` on port 9904 | Docker container not running | `docker ps` to check; restart the container |
| `404` on `GET /health` | No `/health` GET endpoint exists | Use `python check_mcp_health.py` or POST an MCP `initialize` to `/mcp` |
| `307 Temporary Redirect` on `/mcp/` | NAT 1.25 canonicalizes to `/mcp` (no trailing slash); NAT 1.3 was the opposite | Use `/mcp` directly in IDE config and curl probes, or pass `-L` to curl to follow the redirect |
| `406 Not Acceptable` from curl probe | Server replies with `text/event-stream`; default curl `Accept: */*` is rejected | Add `-H 'Accept: application/json, text/event-stream'` |
| `401 Unauthorized` from cloud calls | Invalid `NVIDIA_API_KEY` | Regenerate at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys); update `.env` |
| `ModuleNotFoundError: aiohttp` from `check_mcp_health.py` | The host doesn't have `aiohttp` | `pip install aiohttp`, or run inside the container: `docker exec isaacsim-mcp python /app/check_mcp_health.py` |
| `--env-file: file not found` | Wrong cwd when invoking `docker run` | Run from `source/mcp/isaacsim_mcp/`, or use absolute path |
| Port 9904 already in use | Another process on that port | `lsof -i :9904` or `netstat -aon \| findstr 9904`; stop or remap (e.g. `-p 9914:9904`) |
| `Current Python version (3.10.x) is not allowed by the project` during `build-docker.sh` | poetry venv is bound to system Python 3.10 | `poetry env remove --all && poetry env use python3.12` in **both** `source/aiq/isaacsim_fns/` and `source/mcp/isaacsim_mcp/` |
| `command not found: poetry` during `build-docker.sh` | Poetry isn't installed | Install per [python-poetry.org](https://python-poetry.org/docs/#installation); see Prerequisites |
| `pip ResolutionImpossible` mentioning `ragas` and `nvidia-nat` during `docker build` | The Dockerfile pins `ragas` and `nvidia-nat` to incompatible pre-release versions (e.g. `nvidia-nat==1.5.0a20260120` requires `ragas~=0.2.14`, but the Dockerfile may pin `ragas>=0.3.0rc1`) | Pin `ragas` in `Dockerfile` to the version `nvidia-nat` accepts (`"ragas~=0.2.14"`), or upgrade `nvidia-nat` to a build that allows newer `ragas` |
| Tools not appearing in IDE | MCP config not loaded or wrong URL | Verify with `check_mcp_health.py`; ensure URL is `http://localhost:9904/mcp`; reload IDE |
| `isaac-sim-mcp` already registered / `claude mcp add` rejects | Earlier registration (e.g. against the hosted `isaac-sim-mcp.nvidia.com`) is still in your config | Remove first: `claude mcp remove isaac-sim-mcp -s <local|user|project>`, then re-add against the local server |
| `isaac-sim-mcp` missing from `claude mcp list` | `-t http` registered the MCP at project scope | Re-add with `--scope user`, or always launch Claude CLI from where you registered |

---

## Development

### Running Locally (Without Docker)

```bash
./setup-dev.sh        # Linux/macOS
# setup-dev.bat       # Windows

./run.sh              # Linux/macOS
# run.bat             # Windows
```

---

## License

See the [LICENSE](../../../LICENSE) file in the root of this repository.
