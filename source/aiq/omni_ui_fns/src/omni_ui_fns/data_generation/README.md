# Data Generation Scripts

This directory contains scripts used to **build and prepare data** for the OmniUI MCP server. These scripts are **not used at runtime** by the MCP server.

## Purpose

These scripts are used during development/deployment to:
- Generate embeddings from OmniUI documentation
- Build FAISS vector databases for semantic search
- Analyze and process UI function examples
- Test the generated data assets

## Scripts

### Build Pipeline Scripts

- **`build_complete_ui_examples_pipeline.py`** - Orchestrates the complete data generation pipeline
- **`build_embedding_vectors.py`** - Generates embedding vectors from function descriptions
- **`build_faiss_database.py`** - Creates FAISS indices from embedding vectors

### Analysis & Generation

- **`analyze_ui_functions.py`** - Analyzes UI functions and generates descriptions using Claude

### Testing

- **`test_ui_window_examples.py`** - Tests the generated FAISS database and retrieval functionality

## Usage

These scripts are typically run during:
1. Initial setup of the MCP server data
2. Updates to OmniUI documentation
3. Regeneration of search indices

The generated data assets are stored in the `data/` directory and loaded at runtime by the services in `services/`.

## Runtime Services

For runtime services used by the MCP server during operation, see `../services/`.
