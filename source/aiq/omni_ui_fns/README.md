# OmniUI AIQ Functions

A standalone AIQ functions module for NVIDIA Omniverse UI development, providing comprehensive OmniUI class information, documentation, code examples, and styling guidance.

## Overview

This module contains all the AIQ functions for OmniUI development, separated from the MCP server for better modularity and reusability. These functions can be used:
- Directly in AIQ workflows
- Through the OmniUI MCP server
- In other Python projects

## Features

### Core Functions

- **search_ui_code_examples**: Semantic search for OmniUI code examples with reranking
- **search_ui_window_examples**: Retrieve window and UI layout examples
- **list_ui_classes**: List all available OmniUI classes
- **list_ui_modules**: Browse OmniUI module hierarchy
- **get_ui_class_detail**: Detailed information about specific classes
- **get_ui_module_detail**: Module documentation and contents
- **get_ui_method_detail**: Method signatures and documentation
- **get_ui_instructions**: General OmniUI system instructions
- **get_ui_class_instructions**: Class-specific usage instructions
- **get_ui_style_docs**: Styling and theming documentation

## Installation

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd source/aiq/omni_ui_fns

# Run setup script
./setup-dev.sh  # Unix/Linux/macOS

# Activate virtual environment
poetry shell
```

### Using as a Dependency

Add to your project's dependencies:

```toml
[tool.poetry.dependencies]
omni-ui-fns = {path = "../path/to/omni_ui_fns", develop = true}
```

Or install directly:

```bash
pip install -e /path/to/source/aiq/omni_ui_fns
```

## Usage

### In AIQ Workflows

Functions are registered with AIQ and can be used in workflow configurations:

```yaml
functions:
  search_ui_code_examples:
    _type: omni_ui_fns/search_ui_code_examples
    verbose: false
    enable_rerank: true
  
  list_ui_classes:
    _type: omni_ui_fns/list_ui_classes
    verbose: false
```

### Direct Python Import

```python
from omni_ui_fns.functions.search_ui_code_examples import search_ui_code_examples
from omni_ui_fns.functions.list_ui_classes import list_ui_classes

# Search for button examples
results = await search_ui_code_examples("create button with icon", top_k=5)

# Get all UI classes
classes = await list_ui_classes()
```

## Architecture

```
omni_ui_fns/
├── src/omni_ui_fns/
│   ├── functions/         # Function implementations
│   ├── register_*.py      # AIQ function registrations
│   ├── services/          # Service layer (RAG, retrieval)
│   ├── utils/            # Utility modules
│   └── data/             # Data files and indices
│       ├── instructions/  # Documentation files
│       ├── styles/       # Styling documentation
│       ├── faiss_index_omni_ui/ # Vector indices
│       └── ui_window_examples_faiss/
```

## Dependencies

- Python 3.11+
- AIQ Toolkit 1.1.0
- LangChain for RAG functionality
- FAISS for vector search
- NVIDIA AI Endpoints for embeddings and reranking

## Development

### Adding New Functions

1. Create function implementation in `src/omni_ui_fns/functions/`
2. Create registration wrapper in `src/omni_ui_fns/register_<function>.py`
3. Add entry point in `pyproject.toml`
4. Update documentation

### Testing

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=omni_ui_fns --cov-report=term-missing
```

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.