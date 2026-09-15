# Kit MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives AI coding assistants deep knowledge of NVIDIA Omniverse Kit — extensions, app templates, code examples, settings, and developer instructions — powered by semantic search and NVIDIA AI reranking.

Built on the [NVIDIA AIQ Toolkit](https://github.com/NVIDIA/GenerativeAIExamples) (formerly NeMo Agent Toolkit / NAT 1.3+).

---

## What's new (Apr 2026)

- **MCP protocol bumped to `2025-11-25`** (G-8, G-9): NAT 1.4+ accepts the new spec date end-to-end.
- **Hybrid retrieval is now the default** (`OVAI_RETRIEVAL_MODE=hybrid`): RRF fusion of BM25 lexical + dense semantic with optional cross-encoder reranking.
- **Query expansion active on every search** — 215+ term glossary + auto-generated USD v25.02/v25.11 glossary expand abbreviations (SSS, PBR, LIVRPS, Gf, Sdf, ...) before retrieval (REQ-QE-1..6).
- **BM25 sidecar migrated from `bm25.pkl` to `bm25.json`** (G-7, pickle-safety hardening) — new `bm25_safe` schema is JSON-only, pickle code paths are gone.
- **Result guardrails wired into every search tool** (G-1, REQ-RG-1/2/3): `sanitize_query` on input, deterministic "No relevant documentation found for this query." sentinel for empty results, and `search_kit_settings` caps output at 12000 chars with a `prefix_filter`/`type_filter` hint.
- **Tool descriptions refreshed (Phase 14, 34 tools)** — every tool now has PRIMARY / WHEN-TO-USE / ARGUMENTS / RETURNS / USAGE EXAMPLES / WHEN-TO-USE-A-DIFFERENT-TOOL-INSTEAD / abbreviation-tip sections.

## 5-Minute Quickstart

Get from zero to working Kit tools in your IDE. Follow every step in order.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- An NVIDIA API key (see Step 1 below)
- [Git LFS](https://git-lfs.com/) installed — the FAISS indices and extension metadata under `source/aiq/*/data/` are LFS-tracked. `build-wheels.sh` auto-runs `git lfs install --local && git lfs pull` for you on the first build, so you only need the binary on PATH (`sudo apt-get install git-lfs` or `brew install git-lfs`). Without LFS, the wheel ends up ~13× smaller and the container silently fails at first tool call with `Extension data is not available` — auto-recovery in `build-wheels.sh` catches this.
- The repo cloned: `git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git`

### Step 1: Get Your API Key

| Key | What It's For | Where to Get It |
|-----|---------------|-----------------|
| `NVIDIA_API_KEY` | Authenticates calls to NVIDIA's cloud endpoints for embeddings, reranking, and LLM inference | [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) — sign in, click **Generate API Key**, paste it in place of `REPLACE_WITH_NVIDIA_API_KEY` |

> **Note:** A second key (`NGC_API_KEY` from [org.ngc.nvidia.com/setup/api-key](https://org.ngc.nvidia.com/setup/api-key)) is only required if you plan to run embedder/reranker models locally via NVIDIA NIM containers — see [Deployment Options](#deployment-options).

### Step 2: Configure Your `.env`

```bash
cd kit-usd-agents/source/mcp
cp .env.example .env
```

Open `.env` and set:

```env
NVIDIA_API_KEY=REPLACE_WITH_NVIDIA_API_KEY
```

### Step 3: Build and Run the Docker Container

```bash
cd kit_mcp

# Build the image
./build-docker.sh        # Linux/macOS
# build-docker.bat       # Windows

# Run the server (note --env-file points to ../.env, one level up)
docker run --rm -p 9902:9902 --env-file ../.env kit-mcp:latest
```

> **`.env` location matters.** `--env-file ../.env` resolves relative to your current directory. Run `docker run` from `source/mcp/kit_mcp/`. If you need to launch from elsewhere, use the absolute path: `--env-file "$(git rev-parse --show-toplevel)/source/mcp/.env"`.

### Step 4: Verify the Server is Running

The server speaks Streamable HTTP at `/mcp` (the canonical endpoint in NAT 1.25). There is **no separate `/health` GET endpoint**; the canonical liveness probe is an MCP `initialize` POST.

> **Trailing slash:** older NAT 1.3 builds required `/mcp/`. NAT 1.25 returns `307 Temporary Redirect` from `/mcp/` to `/mcp`, so both work — but using `/mcp` directly avoids a redirect on every call. If you keep the trailing slash, pass `-L` to curl.

In a new terminal:

```bash
# Easiest: use the included Python health check
python check_mcp_health.py

# Or curl the MCP endpoint directly. The Accept header is required —
# NAT 1.4 streams the response as text/event-stream and returns 406
# without it.
curl -s -X POST http://localhost:9902/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"health","version":"1.0"}}}'
```

A healthy server returns a JSON-RPC `result` payload listing the server's name and capabilities.

### Step 5: Connect Your IDE

All IDE configs point at the same URL: `http://localhost:9902/mcp`.

> **MCP scoping note (Claude Code):** `claude mcp add ... -t http <url>` writes to `.claude.json` in your **current working directory**. The MCP is then only active when you launch Claude CLI from that directory. To register globally, pass `--scope user`. This caveat has bitten developers in practice — running `claude mcp add` from `kit-app-template/`, then opened a new terminal elsewhere and found `kit-dev-mcp` mysteriously absent from `claude mcp list`.

<details>
<summary><strong>Cursor</strong></summary>

Create `.cursor/mcp.json` in your project root (or `~/.cursor/mcp.json` for global access):

```json
{
  "mcpServers": {
    "kit-dev-mcp": {
      "url": "http://localhost:9902/mcp"
    }
  }
}
```

> **Common docs typo:** some internal docs show `"type": "kit-dev-mcp"` — that is **not** a valid MCP transport. Cursor's `mcp.json` accepts a bare `"url"` (no `"type"` field needed); if you must include one, the correct value is `"type": "http"`.

Reload Cursor: `Cmd/Ctrl+Shift+P` → **Developer: Reload Window**

</details>

<details>
<summary><strong>Claude Code</strong></summary>

Add via the CLI (project scope — registers in `.claude.json` in your cwd):

```bash
claude mcp add kit-dev-mcp -t http http://localhost:9902/mcp
```

For user (global) scope:

```bash
claude mcp add kit-dev-mcp --scope user -t http http://localhost:9902/mcp
```

Or add it directly to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "kit-dev-mcp": {
      "type": "http",
      "url": "http://localhost:9902/mcp"
    }
  }
}
```

</details>

<details>
<summary><strong>Windsurf</strong></summary>

Create `~/.windsurf/mcp.json`:

```json
{
  "mcpServers": {
    "kit-dev-mcp": {
      "url": "http://localhost:9902/mcp"
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
    "kit-dev-mcp": {
      "type": "http",
      "url": "http://localhost:9902/mcp"
    }
  }
}
```

</details>

### Step 6: Verify Tools Appear

In your IDE's AI chat, you should see **12 Kit tools**:

| Tool | Description |
|------|-------------|
| `get_kit_instructions` | Top-level guidance on Kit development concepts |
| `search_kit_app_templates` | Discover app templates (USD Composer, USD Explorer, etc.) |
| `get_kit_app_template_details` | Detailed info on a specific app template |
| `search_kit_extensions` | Semantic search across the indexed Kit extension catalog |
| `get_kit_extension_details` | Full details for one or more extensions (super-flexible input format) |
| `get_kit_extension_dependencies` | Resolve an extension's dependency graph |
| `get_kit_extension_apis` | List the public APIs exposed by an extension |
| `get_kit_api_details` | Full signature + docstring for a Kit API |
| `search_kit_code_examples` | Find Kit code patterns by description |
| `search_kit_test_examples` | Find Kit test patterns by description |
| `search_kit_settings` | Find a Kit setting by name or purpose |
| `search_kit_knowledge` | General Kit-documentation Q&A retrieval |

Try asking: *"Find me the Kit extension that does clash detection"* — if you get the `omni.physxclashdetection.bundle` family back, the index is intact. If you get "no published Kit bundle in the registry", the index is missing that family — see [Index Coverage](#index-coverage) below.

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
│  Kit MCP Server       │ ← This package
│  (port 9902)          │
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
│  Kit Atlas Database   │  ← Extensions, app templates,
│                       │    code examples, settings,
│                       │    instructions
└──────────────────────┘
```

---

## Deployment Options

### Option A: Cloud Endpoints (Recommended)

```env
NVIDIA_API_KEY=REPLACE_WITH_NVIDIA_API_KEY
```

### Option B: Local NIM Containers (Advanced, GPU required)

```env
NVIDIA_API_KEY=REPLACE_WITH_NVIDIA_API_KEY
NGC_API_KEY=REPLACE_WITH_NGC_API_KEY
KIT_EMBEDDER_BACKEND=local
KIT_LOCAL_EMBEDDER_URL=http://localhost:8080
KIT_RERANKER_BACKEND=local
KIT_LOCAL_RERANKER_URL=http://localhost:8081
```

> NGC API key: [org.ngc.nvidia.com/setup/api-key](https://org.ngc.nvidia.com/setup/api-key). Full local-NIM setup including `docker login nvcr.io`, wheel building, and the `docker-compose` flow is in [`source/mcp/LOCAL_DEPLOYMENT.md`](../LOCAL_DEPLOYMENT.md).

---

## Project Structure

```
source/mcp/kit_mcp/
├── VERSION.md                 # Version information
├── README.md                  # This file
├── pyproject.toml             # Poetry configuration and dependencies
├── Dockerfile                 # Docker image configuration
├── check_mcp_health.py        # MCP-initialize-based health probe
├── setup-dev.sh / .bat        # Development environment setup
├── run.sh / run.bat           # Local run scripts (non-Docker)
├── build-docker.sh / .bat     # Docker image build scripts
├── workflows/                 # AIQ workflow configs
│   ├── config.yaml            # Cloud-endpoint workflow
│   └── local_config.yaml      # Local-NIM workflow
└── src/
    └── kit_mcp/               # Server source code
```

The data corpus ships pre-built in the wheel at `source/aiq/kit_fns/src/kit_fns/data/<kit_version>/`.

---

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
| `OVAI_RETRIEVAL_MODE` | Retrieval mode: `hybrid` (default) or `semantic` (legacy fallback) | `hybrid` |
| `OVAI_FUSION` | Fusion strategy: `rrf` (default), `semantic_only`, `lexical_only` | `rrf` |
| `OVAI_RERANK` | Opt into the cross-encoder reranker: `false` (default), `true` | `false` |
| `OVAI_BM25_BACKEND` | BM25 backend: `rank_bm25` (default), `tantivy`, `whoosh` | `rank_bm25` |
| `OVAI_QE_LOG` | When set, INFO-log every query-expansion firing (REQ-QE-6) | - |

## Usage Analytics

The server includes built-in usage analytics that log tool calls, parameters, success/failure status, execution times, and error messages. Disable by setting `KIT_MCP_DISABLE_USAGE_LOGGING=true`.

## Port Allocation

To avoid conflicts when running multiple MCP servers:
- **omni-ui-mcp**: Port 9901
- **kit-mcp**: Port 9902
- **usd-code-mcp**: Port 9903
- **isaacsim-mcp**: Port 9904

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `connection refused` on port 9902 | Docker container not running | `docker ps` to check; restart the container if needed |
| `404` on `GET /health` | No `/health` GET endpoint exists | Use `python check_mcp_health.py` or POST an MCP `initialize` to `/mcp` (Step 4) |
| `307 Temporary Redirect` on POST `/mcp/` | NAT 1.25 canonicalises to `/mcp`. `curl -f` (without `-L`) treats 307 as success, so a healthcheck never exercises the endpoint. | Drop the trailing slash, or pass `-L` to curl. The repo's Dockerfile and compose healthchecks both use `curl -fL ... /mcp`. |
| `401 Unauthorized` from cloud calls | Invalid or expired `NVIDIA_API_KEY` | Regenerate at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys); update `.env` |
| `--env-file: file not found` | Wrong cwd when invoking `docker run` | Run from `source/mcp/kit_mcp/`, or use absolute path: `--env-file "$(git rev-parse --show-toplevel)/source/mcp/.env"` |
| Port 9902 already in use | Another process on that port | `lsof -i :9902` or `netstat -aon \| findstr 9902`; stop or remap to e.g. `-p 9912:9902` |
| Tools not appearing in IDE | MCP config not loaded or wrong URL | Verify with `check_mcp_health.py`; check IDE's MCP config path and URL (use `/mcp` — trailing slash works too via 307 redirect); reload IDE |
| `[410] Gone — This endpoint has reached its end of life on 2026-05-18T00:00:00Z` during hosted rerank | An older config or image still pins retired model `nvidia/llama-3.2-nv-rerankqa-1b-v2` | Set `OVAI_RERANK_MODEL=nvidia/llama-nemotron-rerank-vl-1b-v2` (or another current model from [build.nvidia.com/explore/retrieval](https://build.nvidia.com/explore/retrieval)), or fall back to `OVAI_RERANK=false`. Full details + on-prem NIM workaround in [LOCAL_DEPLOYMENT.md](../LOCAL_DEPLOYMENT.md#hybrid-retrieval-ovgenai-retrieval-tuning) |
| `kit-dev-mcp` missing from `claude mcp list` | `-t http` registered the MCP at project scope (writes `.claude.json` in cwd) | Re-add with `--scope user` for global, or always launch Claude CLI from the project root where you registered |
| Cursor: `"type": "kit-dev-mcp"` in some docs | Docs typo — `kit-dev-mcp` is not a transport | Cursor accepts bare `"url"`; if you must specify a type, use `"type": "http"` |

---

## Headless / SSH developer path

The MCP server itself runs headlessly under `./run.sh` — no GUI required. Some Kit-app workflows that this MCP helps users discover (e.g. installing extensions in a USD Composer / USD Explorer build) are typically demonstrated GUI-first (Window → Extensions → search → install → AUTOLOAD). The supported alternative for headless dev is editing the `.kit` config file directly:

```toml
[dependencies]
"omni.physxclashdetection.bundle" = { version = "110.1.7" }
"omni.kit.viewport.navigation.usd_explorer.bundle" = {}
```

The next `./repo.sh build` of the kit-app-template will resolve and pull these into `extscache`. This is the documented dev-path alternative for SSH / CI environments — call it out when an MCP-assisted developer is working remotely.

---

**Stale or corrupt hybrid BM25 sidecar:**
- As of Apr 2026 the sidecar is `bm25.json` (the old `bm25.pkl` pickle format was removed for pickle-safety). If hybrid search returns empty or mis-ranked results, delete `bm25.json` next to the FAISS index and restart — the server will rebuild it on first query using the `bm25_safe` JSON schema.

## Dependencies

The Docker images install `ovgenai-retrieval` — the shared hybrid-retrieval library bundled under `source/aiq/ovgenai_retrieval/` and built into a wheel by `source/mcp/build-wheels.sh` — backed by `rank_bm25`, so this MCP and the sibling `ovgenai-agent-search` tool share the exact same RRF-fusion + query-expansion + JSON-BM25 code path.

## Related Projects

- **`ovgenai-agent-search`** — the filesystem-search equivalent of this MCP. Same `ovgenai-retrieval` library, same guardrails, same query expansion, but exposed as a shell-first CLI / skill rather than an HTTP MCP. Use `ovgenai-agent-search` from agents that prefer shell access (e.g., Claude Code skills); use this MCP for agents that prefer HTTP tool calls.

## Development

### Running Locally (Without Docker)

```bash
./setup-dev.sh        # Linux/macOS
# setup-dev.bat       # Windows

./run.sh              # Linux/macOS
# run.bat             # Windows
```

### Configuration Files

- `workflows/config.yaml` — Cloud-endpoint workflow
- `workflows/local_config.yaml` — Local-NIM workflow

---

## License

See the [LICENSE](../../../LICENSE) file in the root of this repository.
