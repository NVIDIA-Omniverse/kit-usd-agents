# OmniUI MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives AI coding assistants deep knowledge of NVIDIA Omniverse's UI framework — `omni.ui` modules, classes, methods, code examples, window patterns, and styling — powered by semantic search and NVIDIA AI reranking.

Built on the [NVIDIA AIQ Toolkit](https://github.com/NVIDIA/GenerativeAIExamples) (formerly NeMo Agent Toolkit / NAT 1.3+).

---

## 5-Minute Quickstart

Get from zero to working OmniUI tools in your IDE. Follow every step in order.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- An NVIDIA API key (see Step 1 below)
- [Git LFS](https://git-lfs.com/) installed — the FAISS indices and UI extension metadata under `source/aiq/*/data/` are LFS-tracked. `build-wheels.sh` auto-runs `git lfs install --local && git lfs pull` for you on the first build, so you only need the binary on PATH (`sudo apt-get install git-lfs` or `brew install git-lfs`). Without LFS, the wheel ends up ~13× smaller and the container silently fails at first tool call with `Extension data is not available` — auto-recovery in `build-wheels.sh` catches this.
- The repo cloned: `git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git`

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
cd omni_ui_mcp

# Build the image
./build-docker.sh        # Linux/macOS
# build-docker.bat       # Windows

# Run the server
docker run --rm -p 9901:9901 --env-file ../.env omni-ui-mcp:latest
```

> **`.env` location matters.** `--env-file ../.env` resolves relative to your current directory. Run `docker run` from `source/mcp/omni_ui_mcp/`, or use the absolute path: `--env-file "$(git rev-parse --show-toplevel)/source/mcp/.env"`.

### Step 4: Verify the Server is Running

The server speaks Streamable HTTP at `/mcp` (the canonical endpoint in NAT 1.25). There is **no separate `/health` GET endpoint**; the canonical liveness probe is an MCP `initialize` POST.

> **Trailing slash:** older NAT 1.3 builds required `/mcp/`. NAT 1.25 returns `307 Temporary Redirect` from `/mcp/` to `/mcp`, so both work — but using `/mcp` directly avoids a redirect on every call. If you keep the trailing slash, pass `-L` to curl.

```bash
# Easiest: use the included Python health check
python check_mcp_health.py

# Or curl the MCP endpoint directly. The Accept header is required —
# NAT 1.4 streams the response as text/event-stream and returns 406
# without it.
curl -s -X POST http://localhost:9901/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"health","version":"1.0"}}}'
```

A healthy server returns a JSON-RPC `result` payload listing the server's name and capabilities.

### Step 5: Connect Your IDE

All IDE configs point at the same URL: `http://localhost:9901/mcp`.

> **MCP scoping note (Claude Code):** `claude mcp add ... -t http <url>` writes to `.claude.json` in your **current working directory**. The MCP is then only active when you launch Claude CLI from that directory. To register globally, pass `--scope user`.

<details>
<summary><strong>Cursor</strong></summary>

```json
{
  "mcpServers": {
    "omni-ui-mcp": {
      "url": "http://localhost:9901/mcp"
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
claude mcp add omni-ui-mcp -t http http://localhost:9901/mcp

# User (global) scope
claude mcp add omni-ui-mcp --scope user -t http http://localhost:9901/mcp
```

Or add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "omni-ui-mcp": {
      "type": "http",
      "url": "http://localhost:9901/mcp"
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
    "omni-ui-mcp": {
      "url": "http://localhost:9901/mcp"
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
    "omni-ui-mcp": {
      "type": "http",
      "url": "http://localhost:9901/mcp"
    }
  }
}
```

</details>

### Step 6: Verify Tools Appear

In your IDE's AI chat, you should see **10 OmniUI tools**:

| Tool | Description |
|------|-------------|
| `get_ui_instructions` | Top-level guidance on building UI with `omni.ui` |
| `get_ui_class_instructions` | Class-by-class authoring tips |
| `get_ui_style_docs` | Styling system documentation |
| `list_ui_modules` | Lists all `omni.ui*` module names |
| `list_ui_classes` | Lists UI classes, optionally filtered by module |
| `get_ui_module_detail` | Detailed info on a specific module |
| `get_ui_class_detail` | Comprehensive class info incl. methods |
| `get_ui_method_detail` | Method signatures and documentation |
| `search_ui_code_examples` | Semantic search across UI code patterns |
| `search_ui_window_examples` | Semantic search across window-layout examples |

Try asking your AI assistant: *"Show me a code example for an `omni.ui.Window` with a styled button"* — if you get a code snippet back, you're set.

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
│  OmniUI MCP Server    │ ← This package
│  (port 9901)          │
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
│  OmniUI Corpus        │  ← Modules, classes, methods,
│                       │    code examples, window
│                       │    examples, style docs
└──────────────────────┘
```

The OmniUI corpus is a curated set of `omni.ui*` API metadata, code patterns, and styling docs — built by the data-collection scripts at `source/aiq/omni_ui_fns/`. Unlike Kit-fns this corpus is not driven by an `extscache` walk; the inputs are hand-curated examples and the published `omni.ui` API surface, so the "missing extension family" failure mode that affects the Kit MCP doesn't apply here.

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
| `connection refused` on port 9901 | Docker container not running | `docker ps` to check; restart the container |
| `404` on `GET /health` | No `/health` GET endpoint exists | Use `python check_mcp_health.py` or POST an MCP `initialize` to `/mcp` |
| `307 Temporary Redirect` on POST `/mcp/` | NAT 1.25 canonicalises to `/mcp`. `curl -f` (without `-L`) treats 307 as success, so a healthcheck never exercises the endpoint. | Drop the trailing slash, or pass `-L` to curl. The repo's Dockerfile and compose healthchecks both use `curl -fL ... /mcp`. |
| `401 Unauthorized` from cloud calls | Invalid `NVIDIA_API_KEY` | Regenerate at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys); update `.env` |
| `--env-file: file not found` | Wrong cwd when invoking `docker run` | Run from `source/mcp/omni_ui_mcp/`, or use absolute path |
| Port 9901 already in use | Another process on that port | `lsof -i :9901` or `netstat -aon \| findstr 9901`; stop or remap (e.g. `-p 9911:9901`) |
| Tools not appearing in IDE | MCP config not loaded or wrong URL | Verify with `check_mcp_health.py`; check IDE's MCP config URL (use `/mcp` — trailing slash works too via 307 redirect); reload IDE |
| `omni-ui-mcp` missing from `claude mcp list` | `-t http` registered the MCP at project scope | Re-add with `--scope user`, or always launch Claude CLI from where you registered |

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
