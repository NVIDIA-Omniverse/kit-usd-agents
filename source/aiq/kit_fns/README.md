# Kit AIQ Functions

A standalone AIQ functions module for NVIDIA Omniverse Kit development, providing comprehensive Kit extension information, APIs, code examples, and application templates.

## Overview

This module contains all the AIQ functions for Kit development, separated from the MCP server for better modularity and reusability. These functions can be used:
- Directly in AIQ workflows
- Through the Kit MCP server
- In other Python projects

## Features

### Core Functions

- **get_kit_instructions**: Retrieve Kit framework documentation and best practices
- **search_kit_extensions**: Semantic search across 400+ Kit extensions
- **get_kit_extension_details**: Comprehensive information about specific extensions
- **get_kit_extension_dependencies**: Analyze extension dependency graphs
- **get_kit_extension_apis**: List all APIs provided by extensions
- **get_kit_api_details**: Detailed documentation for specific APIs
- **search_kit_code_examples**: Find relevant code examples using semantic search
- **search_kit_test_examples**: Find test implementations and patterns
- **search_kit_settings**: Search for Kit configuration settings
- **search_kit_app_templates**: Search for application templates
- **get_kit_app_template_details**: Retrieve specific application examples

## Installation

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd source/aiq/kit_fns

# Run setup script
./setup-dev.sh  # Unix/Linux/macOS

# Activate virtual environment
poetry shell
```

### Using as a Dependency

Add to your project's dependencies:

```toml
[tool.poetry.dependencies]
kit-fns = {path = "../path/to/kit_fns", develop = true}
```

Or install directly:

```bash
pip install -e /path/to/source/aiq/kit_fns
```

## Usage

### In AIQ Workflows

Functions are registered with AIQ and can be used in workflow configurations:

```yaml
functions:
  get_kit_instructions:
    _type: kit_fns/get_kit_instructions
    verbose: false
  
  search_kit_extensions:
    _type: kit_fns/search_kit_extensions
    verbose: false
```

### Direct Python Import

```python
from kit_fns.functions.get_kit_instructions import get_kit_instructions
from kit_fns.functions.search_kit_extensions import search_kit_extensions

# Get Kit system instructions
result = await get_kit_instructions("kit_system")

# Search for UI extensions
results = await search_kit_extensions("user interface", top_k=10)
```

## Architecture

```
kit_fns/
├── src/kit_fns/
│   ├── functions/         # Function implementations
│   ├── register_*.py      # AIQ function registrations
│   ├── services/          # Service layer
│   ├── utils/            # Utility modules
│   └── data/             # Data files and indices
│       ├── instructions/  # Documentation files
│       ├── extensions/    # Extension database
│       ├── code_examples/ # Code example indices
│       └── settings/      # Settings database
```

## Dependencies

- Python 3.11+
- AIQ Toolkit 1.1.0
- LangChain for RAG functionality
- FAISS for vector search
- NVIDIA AI Endpoints for embeddings and reranking

## Development

### Adding New Functions

1. Create function implementation in `src/kit_fns/functions/`
2. Create registration wrapper in `src/kit_fns/register_<function>.py`
3. Add entry point in `pyproject.toml`
4. Update documentation

### Testing

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=kit_fns --cov-report=term-missing
```

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.