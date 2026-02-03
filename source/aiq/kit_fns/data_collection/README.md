# Kit Data Collection Module

**Complete automated pipeline** for processing Omniverse Kit extensions and generating all data files needed by the `kit_fns` MCP service.

## 🚀 Quick Start

### 1. Installation

```bash
cd source/aiq/kit_fns/data_collection
poetry install

# Optional: Set NVIDIA API key for embeddings
export NVIDIA_API_KEY="your-nvidia-api-key"
```

### 2. Run Complete Pipeline

There are two ways to run the pipeline:

#### Option A: Use Local Kit Build (Recommended for Development)
```bash
# Auto-detect Kit cache and run everything
python run_pipeline.py

# Or specify Kit cache path
python run_pipeline.py --kit-cache /path/to/extscache

# Quick test with limited extensions (skips embeddings/FAISS)
python run_pipeline.py --max-extensions 25 --quick
```

#### Option B: Clone and Build Repository Automatically
```bash
# Configure repo in pipeline_config.toml, then run full pipeline
python data_collection_pipeline.py

# The pipeline will:
# 1. Clone the Kit repository
# 2. Build it to generate extscache
# 3. Process all extensions
# 4. Generate embeddings and FAISS databases
```

### 3. Pipeline Recovery & Resume

The pipeline automatically saves checkpoints after each successful stage. If a stage fails, you can resume from where it left off:

```bash
# Resume from last checkpoint (automatic recovery)
python data_collection_pipeline.py --resume

# Force restart from beginning (ignore checkpoint)
python data_collection_pipeline.py --force

# List all available stages
python data_collection_pipeline.py --list-stages

# Run specific stage range
python data_collection_pipeline.py --start extension_data --end settings
```

**How it works:**
- Checkpoint file (`.pipeline_checkpoint.json`) is saved after each successful stage
- Contains: completed stages, config snapshot (kit cache path, kit version)
- On `--resume`, skips completed stages and continues from the first incomplete stage
- On success, checkpoint is automatically cleared
- On failure, checkpoint is preserved for next resume attempt

**Auto-resume mode:**
Set `resume_on_failure = true` in `pipeline_config.toml` to enable automatic checkpoint detection without needing `--resume` flag (useful for automated/scheduled runs)

**Example workflow:**
```bash
# Start pipeline
python data_collection_pipeline.py

# If it fails at stage 3/8, the output will show:
# "Pipeline failed at stage: code_examples"
# "You can resume from this point using: --resume"

# Simply resume
python data_collection_pipeline.py --resume
# Skips stages 1-2, continues from stage 3
```

This generates **all data files** needed by kit_fns:
- Extension metadata & Code Atlas (versioned by Kit version)
- Code examples extracted from Code Atlas
- Settings discovery & documentation
- Embeddings for semantic search
- FAISS vector databases

**See [PIPELINE_README.md](PIPELINE_README.md) for complete documentation.**

## Output Structure

The pipeline generates versioned data organized by Kit version:

```
target_dir/
├── extensions/
│   └── {kit_version}/
│       ├── extensions_database.json
│       ├── extensions_summary.json
│       ├── codeatlas/
│       ├── api_docs/
│       └── extensions_faiss/
├── code_examples/
│   └── {kit_version}/
│       ├── extension_analysis_summary.json
│       ├── extracted_methods/
│       └── code_examples_faiss/
└── settings/
    └── {kit_version}/
        ├── setting_summary.json
        ├── setting_summary_simple.json
        ├── setting_statistics.json
        └── settings_faiss/
```

## Individual Pipeline Components

You can also run individual pipeline components directly:

### Extension Database Builder
```bash
poetry run python extension_data/build_extension_database.py
```

### Code Examples Extractor (Code Atlas-based)
```bash
poetry run python code_example_pipeline/scan_extensions_codeatlas.py
```

### Settings Scanner
```bash
poetry run python settings_pipeline/scan_extension_settings.py
```

### Embedding Generators
```bash
poetry run python extension_data/generate_extension_embeddings.py
poetry run python code_example_pipeline/generate_code_examples_embeddings.py
poetry run python settings_pipeline/generate_settings_embeddings.py
```

### FAISS Database Builders
```bash
poetry run python extension_data/build_extensions_faiss_database.py
poetry run python code_example_pipeline/build_code_examples_faiss_database.py
poetry run python settings_pipeline/build_settings_faiss_database.py
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black extension_data/
poetry run ruff check extension_data/
```

## Architecture

The pipeline consists of multiple stages:

### Pre-Stages
1. **PullRepoExtsStage** (optional) - Clone and build Kit repository to generate extscache
2. **PreparationStage** - Validate inputs and prepare output directories

### Main Pipeline Stages
1. **ExtensionDataStage** - Process extension.toml files and generate Code Atlas
2. **CodeExamplesStage** - Extract code examples from Code Atlas
3. **SettingsStage** - Discover settings from TOML and source code
4. **EmbeddingsStage** - Generate semantic embeddings for search
5. **FAISSStage** - Build vector databases for similarity search
6. **FinalAssemblyStage** - Copy files to target structure with versioning

### Key Components
- **ExtensionProcessor** - Processes individual extensions
- **SimpleCodeAtlasCollector** - AST-based Python code analysis using lc_agent
- **ExtensionsDatabaseBuilder** - Builds the master database index
- **Code Atlas Models** - Data models matching LC Agent's Code Atlas format

## Configuration

Edit `pipeline_config.toml` to customize:
- Repository URL and branch
- Output directories and file locations
- Processing thresholds (min lines, complexity)
- Max extensions to process (for testing)
- Embedding model and batch sizes
- Scan mode (regular, test, etc.)

## Dependencies

- Python 3.11+
- Poetry for package management
- lc_agent module (local path dependency)
- LangChain ecosystem for embeddings
- FAISS for vector search
- tiktoken for token counting
- See pyproject.toml for full dependency list