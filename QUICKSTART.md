# Quick Start Guide (Pure Python)

Get the MCP servers running locally **without Docker** in minutes. This guide covers Windows, macOS, and Linux.

## Prerequisites

- **Python 3.11+** (Python 3.12 recommended)
- **Poetry** package manager
- **NVIDIA API Key** (get one at [build.nvidia.com](https://build.nvidia.com))

## Step 1: Clone and Setup

**Linux/macOS:**
```bash
git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git
cd kit-usd-agents/source/mcp
```

**Windows CMD:**
```cmd
git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git
cd kit-usd-agents\source\mcp
```

**Windows PowerShell:**
```powershell
git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git
cd kit-usd-agents\source\mcp
```

## Step 2: Configure Environment

**Linux/macOS:**
```bash
cp .env.example .env
# Edit .env and add your NVIDIA API key
```

**Windows CMD:**
```cmd
copy .env.example .env
REM Edit .env and add your NVIDIA API key
```

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
# Edit .env and add your NVIDIA API key
```

Add your NVIDIA API key to the `.env` file:
```
NVIDIA_API_KEY=nvapi-your-key-here
```

## Step 3: Install and Run an MCP Server

Choose the MCP server you want to run:

### USD Code MCP

**Linux/macOS:**
```bash
cd usd_code_mcp
./setup-dev.sh    # Run once to set up environment
./run.sh          # Start the server
```

**Windows:**
```cmd
cd usd_code_mcp
setup-dev.bat
run.bat
```

The server will start at `http://localhost:9903/mcp`

### Kit MCP

**Linux/macOS:**
```bash
cd kit_mcp
./setup-dev.sh
./run.sh
```

**Windows:**
```cmd
cd kit_mcp
setup-dev.bat
run.bat
```

The server will start at `http://localhost:9902/mcp`

### OmniUI MCP

**Linux/macOS:**
```bash
cd omni_ui_mcp
./setup-dev.sh
./run.sh
```

**Windows:**
```cmd
cd omni_ui_mcp
setup-dev.bat
run.bat
```

The server will start at `http://localhost:9901/mcp`

## Step 4: Connect Your AI Client

### Cursor IDE

1. Create or edit `.cursor/mcp.json` in your **project/workspace root**:

```json
{
  "mcpServers": {
    "omni-ui-mcp": {
      "type": "omni-ui-mcp",
      "url": "http://localhost:9901/mcp"
    },
    "kit-mcp": {
      "type": "kit-mcp",
      "url": "http://localhost:9902/mcp"
    },
    "usd-code-mcp": {
      "type": "usd-code-mcp",
      "url": "http://localhost:9903/mcp"
    }
  }
}
```

2. **Enable the tools:** Go to Cursor Settings → Tools and MCPs. You should see the 3 MCPs - toggle to enable them.


### Claude Code CLI

Add the MCP servers using the `claude mcp add` command:

```bash
claude mcp add --transport http omni-ui-mcp http://localhost:9901/mcp
claude mcp add --transport http kit-mcp http://localhost:9902/mcp
claude mcp add --transport http usd-code-mcp http://localhost:9903/mcp
```

Test with Claude Code:
```bash
claude "Using the usd-code-mcp tools, list all USD modules available"
```

## Server Ports

| Server | Default Port | Endpoint |
|--------|--------------|----------|
| OmniUI MCP | 9901 | http://localhost:9901/mcp |
| Kit MCP | 9902 | http://localhost:9902/mcp |
| USD Code MCP | 9903 | http://localhost:9903/mcp |

## Troubleshooting

### "NVIDIA_API_KEY not set"

**Linux/macOS:**
```bash
export NVIDIA_API_KEY=nvapi-your-key-here
```

**Windows CMD:**
```cmd
set NVIDIA_API_KEY=nvapi-your-key-here
```

**Windows PowerShell:**
```powershell
$env:NVIDIA_API_KEY="nvapi-your-key-here"
```

### Poetry not found

**Linux/macOS:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows CMD:**
```cmd
pip install poetry
```

**Windows PowerShell:**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### "aiq command not found"

```bash
poetry install
poetry shell
```

### Port already in use

**Linux/macOS:**
```bash
MCP_PORT=9910 ./run.sh
```

**Windows:**
```cmd
set MCP_PORT=9910
run.bat
```

## Next Steps

- Check out the full [Local Deployment Guide](source/mcp/LOCAL_DEPLOYMENT.md) for Docker-based deployment with local NIMs
- Explore the MCP tool capabilities in your AI client
- Read the individual server READMEs in `source/mcp/*/README.md` for detailed documentation
