# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-05-19

First public release covering the four MCP servers and their supporting packages under a single distribution version. Individual components keep their own SemVer; the table below records which component version is included in this release.

| Component | Version |
| --- | --- |
| `source/mcp/isaacsim_mcp`, `source/mcp/isaacsim_mcp_maas` | 2.0.0 |
| `source/mcp/kit_mcp`, `source/mcp/omni_ui_mcp`, `source/mcp/usd_code_mcp` | 1.0.0 |
| `source/mcp/kit_mcp_maas`, `source/mcp/omni_ui_mcp_maas`, `source/mcp/usd_code_mcp_maas` | 0.1.0 |
| `source/aiq/isaacsim_fns` | 2.0.1 |
| `source/aiq/kit_fns`, `source/aiq/omni_ui_fns` | 0.6.0 |
| `source/aiq/usd_code_fns` | 0.3.0 |

### Added

#### MCP Servers (Model Context Protocol)

Four new MCP servers provide AI-powered assistance for Omniverse development:

- **USD Code MCP** (`source/mcp/usd_code_mcp`)
  - USD API documentation and code examples
  - Semantic search across USD knowledge base
  - Class, module, and method detail lookups
  - Port: 9903

- **Kit MCP** (`source/mcp/kit_mcp`)
  - Omniverse Kit extension documentation
  - Extension dependency analysis
  - Kit API reference and code examples
  - Settings and configuration search
  - Port: 9902

- **OmniUI MCP** (`source/mcp/omni_ui_mcp`)
  - omni.ui widget documentation and styling
  - UI code examples and window patterns
  - Class and method reference
  - Port: 9901

- **Isaac Sim MCP** (`source/mcp/isaacsim_mcp`)
  - Isaac Sim extension documentation and code examples
  - Settings discovery across Isaac Sim configuration
  - Robotics-focused developer instructions
  - Port: 9904

#### Deployment Options

- **NVIDIA API**: Cloud-based embeddings and reranking (no GPU required)
- **Local NIMs**: On-premise deployment with NVIDIA NIM containers
- **Docker Compose**: Ready-to-use configurations for both options

#### Documentation

- `QUICKSTART.md`: Pure Python setup guide for Windows, macOS, and Linux
- `LOCAL_DEPLOYMENT.md`: Comprehensive deployment guide with Docker

---

## Previous Release

The initial public release contained **Chat USD**, a multi-agent AI assistant for USD development within Omniverse Kit:
- USD code generation and execution
- USD asset search
- Scene information retrieval

[Unreleased]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/releases/tag/v1.3.0
