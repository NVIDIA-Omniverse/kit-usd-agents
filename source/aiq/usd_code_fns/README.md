# USD Code AIQ Functions

A standalone AIQ functions module for USD/OpenUSD development, providing comprehensive USD code examples, documentation, and knowledge retrieval capabilities.

## Overview

This module contains all the AIQ functions for USD Code development, designed for use with NVIDIA's AI Query (AIQ) toolkit. These functions can be used:
- Directly in AIQ workflows
- Through MCP (Model Context Protocol) servers
- In other Python projects

## Features

### Core Functions

- **list_usd_modules**: Get a list of all available USD modules
- **list_usd_classes**: Get a list of classes in a USD module
- **get_usd_module_detail**: Get detailed information about a specific USD module
- **get_usd_class_detail**: Get detailed information about a USD class including methods and properties
- **get_usd_method_detail**: Get detailed information about a specific method
- **search_usd_code_examples**: Retrieve relevant USD code examples using semantic search
- **search_usd_knowledge**: Search USD documentation and knowledge base
- **give_usd_feedback**: Provide feedback on USD code examples and documentation

## Installation

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd source/aiq/usd_code_fns

# Run setup script
./setup-dev.sh  # Unix/Linux/macOS

# Activate virtual environment
poetry shell
```

### Using as a Dependency

Add to your project's dependencies:

```toml
[tool.poetry.dependencies]
usd-code-aiq = {path = "../path/to/usd_code_fns", develop = true}
```

Or install directly:

```bash
pip install -e /path/to/source/aiq/usd_code_fns
```

## Usage

### In AIQ Workflows

Functions are registered with AIQ and can be used in workflow configurations:

```yaml
functions:
  list_usd_modules:
    _type: usd_code/list_usd_modules
    verbose: false

  search_usd_code_examples:
    _type: usd_code/search_usd_code_examples
    verbose: false
```

### Direct Python Import

```python
from omni_aiq_usd_code.functions.list_usd_modules import list_usd_modules
from omni_aiq_usd_code.functions.search_usd_code_examples import search_usd_code_examples

# Get list of USD modules
result = await list_usd_modules()

# Search for code examples
results = await search_usd_code_examples("How to create a prim", top_k=5)
```

## Architecture

```
usd_code_fns/
├── src/omni_aiq_usd_code/
│   ├── functions/         # Function implementations
│   ├── register_*.py      # AIQ function registrations
│   ├── services/          # Service layer
│   ├── utils/            # Utility modules
│   ├── models/           # Data models
│   └── data/             # Data files and indices
│       ├── code_examples/ # USD code example database
│       ├── documentation/ # USD documentation
│       └── knowledge/     # USD knowledge base
```

## Dependencies

- Python 3.11+
- AIQ Toolkit 1.1.0
- LangChain for RAG functionality
- FAISS for vector search
- NVIDIA AI Endpoints for embeddings and reranking

## Development

### Adding New Functions

1. Create function implementation in `src/omni_aiq_usd_code/functions/`
2. Create registration wrapper in `src/omni_aiq_usd_code/register_<function>.py`
3. Add entry point in `pyproject.toml`
4. Update documentation

### Testing

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=omni_aiq_usd_code --cov-report=term-missing
```

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
