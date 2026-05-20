# Isaac Sim MCP Documentation

Welcome to the Isaac Sim MCP documentation. This directory contains comprehensive documentation for the Isaac Sim Model Context Protocol (MCP) server implementation.

## Overview

The Isaac Sim MCP provides AI-powered development tools for NVIDIA Omniverse Isaac Sim, enabling intelligent assistance for robotics and simulation development through 5 specialized tools:

- **get_isaac_sim_instructions**: System documentation and best practices
- **search_isaac_sim_extensions**: Semantic search across Isaac Sim extensions
- **get_isaac_sim_extension_details**: Detailed extension information
- **search_isaac_sim_code_examples**: Find relevant code examples
- **search_isaac_sim_settings**: Search configuration settings

## Documentation Structure

### 1. [Architecture](architecture.md) - Comprehensive Technical Architecture

**Purpose**: Complete technical specification of the Isaac Sim MCP system

**Contents**:
- Tool definitions and input schemas
- Architecture principles and patterns
- Service layer architecture
- Data collection pipeline overview
- Performance considerations
- Integration with NAT framework
- API summary and tool descriptions

**Audience**: Developers, architects, contributors

**When to Read**:
- Understanding the complete system design
- Adding new tools or features
- Integrating with other systems
- Performance optimization

### 2. [Prompt Architecture](prompt_arch.md) - Conversational Overview

**Purpose**: High-level conversational description of the Isaac Sim MCP system

**Contents**:
- Simplified explanation of the MCP approach
- Tool descriptions in plain language
- Use cases and workflows
- Key differences from Kit MCP
- Integration overview

**Audience**: Product managers, new developers, AI model developers

**When to Read**:
- First introduction to Isaac Sim MCP
- Understanding the "why" behind the design
- Explaining to non-technical stakeholders
- Quick reference for capabilities

### 3. [Settings Architecture](settings_architecture.md) - Settings Pipeline Deep Dive

**Purpose**: Detailed guide to the settings collection and search pipeline

**Contents**:
- Settings collection process
- Data structure design
- Embedding generation
- FAISS database construction
- MCP implementation guide
- Usage examples and troubleshooting

**Audience**: Data engineers, contributors, advanced users

**When to Read**:
- Working with the settings pipeline
- Understanding semantic search implementation
- Debugging settings extraction
- Extending the pipeline

## Quick Start Guide

### For AI Model Developers

If you're integrating Isaac Sim MCP with an AI model or agent:

1. **Start here**: [Prompt Architecture](prompt_arch.md)
   - Understand available tools
   - Learn use cases and workflows

2. **Then read**: [Architecture](architecture.md) - Tool Definitions section
   - Detailed tool parameters
   - Expected return formats

### For Contributors

If you're contributing to Isaac Sim MCP:

1. **Start here**: [Architecture](architecture.md)
   - Complete system design
   - Implementation patterns

2. **Deep dive**: [Settings Architecture](settings_architecture.md)
   - Pipeline implementation details
   - Data processing workflows

3. **Reference**: [Prompt Architecture](prompt_arch.md)
   - User-facing perspective
   - Use case validation

### For End Users (via Cursor/IDEs)

If you're using Isaac Sim MCP through Cursor or another IDE:

1. **Quick reference**: [Prompt Architecture](prompt_arch.md)
   - Tool descriptions
   - Example queries

2. **Troubleshooting**: [Settings Architecture](settings_architecture.md) - Troubleshooting section
   - Common issues and solutions

## Key Concepts

### Hierarchical Information Retrieval

Isaac Sim MCP uses a scaffolded approach to information:

```
High Level → System Instructions (get_isaac_sim_instructions)
    ↓
Mid Level → Extension Discovery (search_isaac_sim_extensions)
    ↓
Detailed → Extension Details (get_isaac_sim_extension_details)
    ↓
Practical → Code Examples (search_isaac_sim_code_examples)
    ↓
Config → Settings (search_isaac_sim_settings)
```

### Semantic Search

All search tools use semantic search powered by:
- **NVIDIA Embeddings**: `nv-embedqa-e5-v5` model
- **FAISS**: Vector similarity search
- **LangChain**: Integration layer

This allows natural language queries like:
- "robot articulation setup"
- "physics simulation configuration"
- "sensor data collection patterns"

### Batch Processing

Many tools support batch operations:
```python
# Single
get_isaac_sim_extension_details("omni.isaac.core")

# Batch (70% faster)
get_isaac_sim_extension_details(["omni.isaac.core", "omni.isaac.sensor", "omni.isaac.manipulators"])
```

### Versioned Data

All data is versioned by Isaac Sim version (controlled by the `MCP_ISAACSIM_VERSION` env var, default `6.0`):
```
data/
├── 6.0/            # Isaac Sim 6.0
└── ...             # Additional versions as needed
```

## Integration Examples

### Cursor IDE

Add to `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "isaacsim-mcp": {
      "type": "mcp",
      "url": "http://localhost:9904/mcp"
    }
  }
}
```

### Python Script

```python
import requests

# Search for extensions
response = requests.post(
    "http://localhost:9904/tools/call",
    json={
        "name": "search_isaac_sim_extensions",
        "arguments": {
            "query": "robot manipulation",
            "top_k": 5
        }
    }
)

results = response.json()
```

## Data Sources

Isaac Sim MCP relies on data generated by the automated pipeline:

### Location
```
source/aiq/isaacsim_fns/src/isaacsim_fns/data/{version}/
```

### Contents
- **extensions/**: Extension metadata, Code Atlas, FAISS database
- **code_examples/**: Extracted code examples, FAISS database
- **settings/**: Settings database, FAISS database
- **instructions/**: System documentation markdown files

The data corpus ships pre-built in the wheel; users do not need to regenerate it.

## Technology Stack

- **NAT Framework (nvidia-nat >= 1.4.0)**: Function registration and MCP server
- **LangChain**: Vector store and embedding integrations
- **FAISS**: High-performance semantic search
- **NVIDIA AI Endpoints**: Embedding generation (`nv-embedqa-e5-v5`)
- **Pydantic**: Input validation
- **Redis** (optional): Telemetry and usage tracking
- **Python 3.11+**: Core implementation

## Performance Notes

### Typical Query Times
- **search_isaac_sim_extensions**: 50-200ms
- **get_isaac_sim_extension_details**: 10-50ms per extension
- **search_isaac_sim_code_examples**: 100-300ms
- **search_isaac_sim_settings**: 50-150ms

### Optimization Tips
1. Use batch operations for multiple items
2. Specify filters (prefix, type) to narrow results
3. Adjust `top_k` based on needs
4. Services automatically cache frequently accessed data

## Port Configuration

- **Isaac Sim MCP**: Port 9904 (default)
- **Kit MCP**: Port 9902
- **Can run simultaneously**: Yes

Set custom port:
```bash
export MCP_PORT=9904
```

## Development

### Running Locally

```bash
# Setup (once)
./setup-dev.sh

# Run server
./run.sh
```

### Testing

```bash
cd source/aiq/isaacsim_fns
poetry run pytest tests/
```

### Adding New Tools

1. Add function to `isaacsim_fns` package
2. Update `workflows/config.yaml`
3. Update `src/isaacsim_mcp/__main__.py` tool list
4. Update documentation
5. Reinstall: `poetry install`

## Support and Contribution

### Reporting Issues

- Check [Troubleshooting](settings_architecture.md#troubleshooting) first
- Search existing issues in the repository
- Provide minimal reproduction steps
- Include log output and configuration

### Contributing

1. Read [Architecture](architecture.md) for design patterns
2. Follow existing code style
3. Add tests for new features
4. Update relevant documentation
5. Submit merge request with clear description

## Related Documentation

- **Main README**: [../README.md](../README.md) - Quick start and installation
- **Functions Docs**: [../../../aiq/isaacsim_fns/docs/ISAACSIM_FNS_DOCUMENTATION.md](../../../aiq/isaacsim_fns/docs/ISAACSIM_FNS_DOCUMENTATION.md) - Functions module documentation

## Glossary

- **MCP**: Model Context Protocol - Standard protocol for AI tool integration
- **NAT**: NeMo Agent Toolkit — NVIDIA's framework for building AI agent tools and MCP servers
- **FAISS**: Facebook AI Similarity Search - Vector database library
- **Code Atlas**: Structured representation of Python code APIs
- **Embeddings**: Vector representations of text for semantic search
- **Extension**: Omniverse Kit plugin/module
- **Settings**: Configuration parameters in Carbonite settings system

## Version History

- **v2.0.0**: Migration to NAT (NeMo Agent Toolkit) 1.4+, Isaac Sim 6.0 data, 49 instruction sets
- **v0.5.0**: Initial release with 5 core tools
- **v0.4.0**: Beta testing and refinement
- **v0.3.0**: Alpha release with basic functionality

## License

Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

See LICENSE file in the repository root for details.
