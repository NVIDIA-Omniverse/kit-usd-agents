# Isaac Sim NAT Functions

A standalone NAT (NeMo Agent Toolkit) functions module for NVIDIA Omniverse Isaac Sim development, providing comprehensive Isaac Sim extension information, APIs, code examples, and application templates.

## Overview

This module contains all the NAT functions for Isaac Sim development, separated from the MCP server for better modularity and reusability. These functions can be used:
- Directly in NAT/AIQ workflows
- Through the Isaac Sim MCP server
- In other Python projects

## Features

### Core Functions

- **get_isaac_sim_instructions**: Retrieve Isaac Sim framework documentation and best practices
- **search_isaac_sim_extensions**: Semantic search across Isaac Sim extensions
- **get_isaac_sim_extension_details**: Comprehensive information about specific extensions
- **search_isaac_sim_code_examples**: Find relevant code examples using semantic search
- **search_isaac_sim_settings**: Search for Isaac Sim configuration settings

## Installation

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd source/aiq/isaacsim_fns

# Run setup script
./setup-dev.sh  # Unix/Linux/macOS

# Activate virtual environment
poetry shell
```

### Using as a Dependency

Add to your project's dependencies:

```toml
[tool.poetry.dependencies]
isaacsim-fns = {path = "../path/to/isaacsim_fns", develop = true}
```

Or install directly:

```bash
pip install -e /path/to/source/aiq/isaacsim_fns
```

## Usage

### In NAT/AIQ Workflows

Functions are registered with NAT (with AIQ backwards-compatible entry points) and can be used in workflow configurations:

```yaml
functions:
  get_isaac_sim_instructions:
    _type: isaacsim_fns/get_isaac_sim_instructions
    verbose: false

  search_isaac_sim_extensions:
    _type: isaacsim_fns/search_isaac_sim_extensions
    verbose: false
```

### Direct Python Import

```python
from isaacsim_fns.functions.get_instructions import get_instructions
from isaacsim_fns.functions.search_extensions import search_extensions

# Get Isaac Sim system instructions
result = await get_instructions(["isaacsim_system"])

# Search for robotics extensions
results = await search_extensions("robotics", top_k=10)
```

## Architecture

```
isaacsim_fns/
├── src/isaacsim_fns/
│   ├── functions/         # Function implementations
│   ├── register_*.py      # NAT function registrations
│   ├── services/          # Service layer
│   ├── utils/            # Utility modules
│   └── data/             # Data files and indices (versioned)
│       └── {version}/     # e.g., 6.0/ (controlled by MCP_ISAACSIM_VERSION)
│           ├── instructions/  # Documentation files
│           ├── extensions/    # Extension database
│           ├── code_examples/ # Code example indices
│           └── settings/      # Settings database
```

## Dependencies

- Python >=3.11, <3.14
- nvidia-nat >= 1.4.0 (NeMo Agent Toolkit)
- nvidia-nat-langchain >= 1.4.0
- LangChain for RAG functionality
- FAISS for vector search
- NVIDIA AI Endpoints for embeddings and reranking

## Development

### Adding New Functions

1. Create function implementation in `src/isaacsim_fns/functions/`
2. Create registration wrapper in `src/isaacsim_fns/register_<function>.py`
3. Add entry point in `pyproject.toml`
4. Update documentation

### Testing

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=isaacsim_fns --cov-report=term-missing
```

## License

Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
