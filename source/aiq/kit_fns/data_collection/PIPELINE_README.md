# Automated Kit Extensions Data Collection Pipeline

This directory contains the **complete automated pipeline** for processing Omniverse Kit extensions and generating all data files needed by the `kit_fns` MCP service.

## Overview

The pipeline chains together multiple data collection processes to extract:
- **Extension metadata & APIs** (from `extension.toml` and Python code)
- **Code examples** (extracted from Code Atlas using lc_agent)
- **Settings** (from TOML files and source code usage)
- **Embeddings** (for semantic search using NVIDIA models)
- **FAISS databases** (for vector similarity search)
- **Versioned output** (organized by Kit version for multi-version support)

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies
cd source/aiq/kit_fns/data_collection
poetry install

# Set NVIDIA API key for embeddings (optional but recommended)
export NVIDIA_API_KEY="your-nvidia-api-key"
```

### 2. Choose Your Approach

There are two ways to run the pipeline:

#### Option A: Use Existing Kit Build (Faster for Development)
```bash
# Auto-detect Kit cache from local builds
python run_pipeline.py

# Or specify Kit cache path directly
python run_pipeline.py --kit-cache /path/to/_build/linux-x86_64/release/extscache

# Quick test with limited extensions (skips embeddings/FAISS)
python run_pipeline.py --max-extensions 25 --quick
```

#### Option B: Clone and Build Repository (Automated Full Build)
```bash
# Edit pipeline_config.toml to set repo URL and branch:
[input]
repo_url = "https://gitlab-master.nvidia.com/omniverse/kit-github/kit-app-template.git"
branch = "production/107.3"

# Run full pipeline (clones, builds, then processes)
python data_collection_pipeline.py
```

### 3. Verify Output

The pipeline generates versioned data in `../src/kit_fns/data/`:
```
src/kit_fns/data/
├── extensions/
│   └── {kit_version}/              # e.g., 107.3.1
│       ├── extensions_database.json
│       ├── extensions_summary.json
│       ├── codeatlas/              # Code Atlas files per extension
│       ├── api_docs/               # Public API documentation
│       └── extensions_faiss/       # FAISS search database
├── code_examples/
│   └── {kit_version}/
│       ├── extension_analysis_summary.json
│       ├── extracted_methods/      # Extracted code examples
│       └── code_examples_faiss/    # FAISS search database
└── settings/
    └── {kit_version}/
        ├── setting_summary.json
        ├── setting_summary_simple.json
        ├── setting_statistics.json
        └── settings_faiss/         # FAISS search database
```

## Pipeline Stages

The pipeline consists of **2 pre-stages** and **6 main stages**:

### Pre-Stage 1: Pull Repo & Build (Optional)
- Clones Kit repository from configured Git URL
- Runs build process to generate extscache
- Detects Kit version from build
- Auto-configures pipeline to use generated extscache
- **Output**: Built extscache directory and Kit version
- **Note**: Skipped if using `run_pipeline.py` with existing build

### Pre-Stage 2: Preparation
- Validates Kit cache directory exists and is accessible
- Creates output directories for intermediate files
- Counts extensions to process (applies filters and limits)
- **Output**: Validated directory structure

### Stage 1: Extension Data Collection
- Processes `extension.toml` files for metadata
- Generates Code Atlas for Python APIs using `lc_agent`
- Creates public API documentation
- Includes extension dependencies and version info
- **Output**: `extensions_database.json`, `extensions_summary.json`, Code Atlas files, API docs

### Stage 2: Code Examples Extraction
- Analyzes Code Atlas to find "interesting" methods
- Filters by size, complexity, and pattern criteria
- Extracts complete source code with context
- Counts tokens for LLM usage estimation
- **Output**: `extension_analysis_summary.json`, extracted method files

### Stage 3: Settings Discovery
- Extracts settings from `extension.toml` files
- Scans Python/C++ source for settings usage patterns
- Converts to canonical slash format (`/exts/name/setting`)
- Tracks usage counts and locations
- **Output**: Settings summary, simple summary, and statistics files

### Stage 4: Embeddings Generation
- Generates semantic embeddings for extensions, code examples, and settings
- Uses NVIDIA embedding models (configurable)
- Processes in batches for efficiency
- Truncates text to max token limits
- **Output**: Embedding JSON files for each data type

### Stage 5: FAISS Database Creation
- Builds FAISS vector databases from embeddings
- Creates searchable indexes for similarity search
- Stores metadata for result retrieval
- **Output**: FAISS index files, metadata, and pickle files

### Stage 6: Final Assembly
- Copies all generated files to target directory structure
- Organizes by Kit version for multi-version support
- Generates completion report with statistics
- Optionally cleans up intermediate files
- **Output**: Versioned data structure in target directory

## Advanced Usage

### Run Specific Stages

```bash
# Run only specific stages (note: pre-stages always run first)
python data_collection_pipeline.py --start extension_data --end settings

# Run from settings extraction to completion
python data_collection_pipeline.py --start settings

# List all available stages
python data_collection_pipeline.py --list-stages
```

### Runner Script Options

The `run_pipeline.py` script provides additional convenience options:

```bash
# Include function source code in Code Atlas (significantly increases size)
python run_pipeline.py --include-source

# Keep intermediate files after completion
python run_pipeline.py --keep-intermediate

# Dry run to see what would be done
python run_pipeline.py --dry-run

# Combined options
python run_pipeline.py --max-extensions 50 --include-source --keep-intermediate
```

### Configuration Options

Key configuration parameters in `pipeline_config.toml`:

```toml
[input]
# Repository configuration (for automated cloning)
repo_url = "https://gitlab-master.nvidia.com/omniverse/kit-github/kit-app-template.git"
branch = "production/107.3"

# Extensions to exclude from processing
excluded_extensions = [
    "omni.kit.test",
    "omni.kit.pip_archive"
]

[output]
# Directory for intermediate pipeline files
work_dir = "../../../../_pipeline_output"

# Final output directory
target_dir = "../src/kit_fns/data"

# Keep intermediate files for debugging
keep_intermediates = true

[processing]
# Include function source code (increases output size significantly)
include_source_code = false

# Code examples filtering thresholds
code_examples_min_lines = 50          # Minimum lines for "interesting" methods
code_examples_min_complexity = 3      # Minimum complexity score
code_examples_scan_mode = "regular"   # Scan mode: "regular", "test", etc.

# Limit processing for testing (-1 = all extensions)
max_extensions = -1

# Number of parallel workers (future enhancement)
parallel_workers = 4

[embeddings]
# NVIDIA embedding model configuration
model = "nvidia/nv-embedqa-e5-v5"
batch_size = 50
nvidia_api_key = "${NVIDIA_API_KEY}"
endpoint_url = ""                     # Optional custom endpoint
encoding_model = "cl100k_base"        # Token encoding model
max_tokens = 500                       # Max tokens per embedding

[logging]
level = "INFO"                         # DEBUG, INFO, WARNING, ERROR
log_file = "pipeline.log"              # Set to null for console only
```

## Individual Pipeline Components

You can also run individual pipeline components directly for debugging or custom workflows:

### Extension Database Builder
Processes extension.toml files and generates Code Atlas:
```bash
poetry run python extension_data/build_extension_database.py
```

### Code Examples Extractor (Code Atlas-based)
Analyzes Code Atlas to extract interesting code examples:
```bash
poetry run python code_example_pipeline/scan_extensions_codeatlas.py
```

### Settings Scanner
Discovers settings from TOML files and source code:
```bash
poetry run python settings_pipeline/scan_extension_settings.py
```

### Embedding Generators
Generate semantic embeddings for search:
```bash
# Extensions embeddings
poetry run python extension_data/generate_embeddings_descriptions.py
poetry run python extension_data/generate_extension_embeddings.py

# Code examples embeddings
poetry run python code_example_pipeline/generate_code_examples_embeddings.py

# Settings embeddings
poetry run python settings_pipeline/generate_settings_embeddings.py
```

### FAISS Database Builders
Build vector search indexes:
```bash
# Extensions FAISS database
poetry run python extension_data/build_extensions_faiss_database.py

# Code examples FAISS database
poetry run python code_example_pipeline/build_code_examples_faiss_database.py

# Settings FAISS database
poetry run python settings_pipeline/build_settings_faiss_database.py
```

## Dependencies

The pipeline requires:
- **Python 3.11+**
- **Poetry** for dependency management
- **lc_agent** module (local path dependency)
- **LangChain** ecosystem for embeddings/vector stores
- **NVIDIA AI Endpoints** for embeddings (optional)
- **FAISS** for vector similarity search
- **tiktoken** for token counting

## Output Data Formats

All data files are organized by Kit version in the output structure.

### Extensions Database (`extensions/{version}/extensions_database.json`)
```json
{
  "database_version": "1.0.0",
  "generated_at": "2024-01-01T12:00:00",
  "kit_version": "107.3.1",
  "total_extensions": 450,
  "extensions": {
    "omni.ui": {
      "version": "2.22.5",
      "title": "Omni UI Framework",
      "description": "Core UI framework for Omniverse",
      "has_python_api": true,
      "codeatlas_path": "codeatlas/omni.ui.json",
      "codeatlas_token_count": 45000,
      "api_docs_path": "api_docs/omni.ui.md",
      "dependencies": ["omni.ui.core"]
    }
  }
}
```

### Code Examples (`code_examples/{version}/extracted_methods/*.example.json`)
```json
{
  "extension_name": "omni.ui",
  "kit_version": "107.3.1",
  "interesting_methods_count": 15,
  "scan_mode": "regular",
  "methods": [{
    "name": "build_ui",
    "file_path": "omni/ui/window.py",
    "line_start": 45,
    "line_end": 112,
    "line_count": 67,
    "token_count": 890,
    "complexity_score": 8,
    "source_code": "def build_ui(self):\n    # Complete method source..."
  }]
}
```

### Settings Summary (`settings/{version}/setting_summary.json`)
```json
{
  "metadata": {
    "kit_version": "107.3.1",
    "total_settings": 1200,
    "total_extensions_scanned": 450,
    "generated_at": "2024-01-01T12:00:00"
  },
  "settings": {
    "/exts/omni.ui/enabled": {
      "default_value": true,
      "type": "bool",
      "documentation": "Enable the UI framework",
      "extensions": ["omni.ui"],
      "usage_locations": ["omni.ui.core", "omni.ui.window"],
      "usage_count": 5
    }
  }
}
```

### FAISS Database Structure
Each FAISS database directory contains:
- `index.faiss` - FAISS vector index for similarity search
- `index.pkl` - Pickled metadata for result retrieval
- `metadata.json` - Index configuration and statistics

## Troubleshooting

### Common Issues

1. **"Kit cache directory does not exist"**
   - If using `run_pipeline.py`: Ensure you've built Kit with `build.sh -r` or equivalent
   - If using `data_collection_pipeline.py`: The pipeline will clone and build automatically
   - Check the path in `pipeline_config.toml`
   - Verify correct platform architecture (linux-x86_64, linux-aarch64, etc.)

2. **"Repository clone failed"**
   - Verify Git credentials for the repository URL
   - Check network connectivity
   - Ensure sufficient disk space for clone and build (~10GB+)
   - Verify the branch name is correct

3. **"Build failed" during Pull Repo stage**
   - Check build logs in the work directory
   - Ensure all system dependencies are installed (build-essential, etc.)
   - Verify Python 3.11+ is available
   - Try building manually first to identify issues

4. **"NVIDIA API key not available"**
   - Set `NVIDIA_API_KEY` environment variable
   - Or run with `--quick` to skip embeddings generation
   - Alternatively, set a custom endpoint URL in config

5. **"Import error: lc_agent not found"**
   - The pipeline should automatically find prebundle in build directory
   - If building outside Kit context: `cd ../../../modules/lc_agent && pip install -e .`
   - Check that prebundle_path is correctly detected in logs

6. **"Memory issues with large codebases"**
   - Use `--max-extensions` to limit processing: `python run_pipeline.py --max-extensions 50`
   - Disable source code inclusion: `include_source_code = false` in config
   - Use `--quick` mode to skip embeddings/FAISS generation
   - Process in batches by running specific stage ranges

7. **"Code Atlas token count too high"**
   - This is informational - indicates large API surface
   - Set `include_source_code = false` to reduce size
   - Consider filtering specific extensions via exclusion list

### Debug Mode

Enable detailed logging and keep intermediate files:
```toml
[logging]
level = "DEBUG"
log_file = "pipeline_debug.log"

[output]
keep_intermediates = true
```

Or use command line:
```bash
python run_pipeline.py --keep-intermediate
```

### Platform-Specific Issues

#### Linux
- Ensure build-essential and git are installed
- Check disk space and permissions in work directory

#### Windows
- Not currently supported for automated repo building
- Use `run_pipeline.py` with pre-built Kit cache instead

## Extending the Pipeline

### Adding New Data Types

1. Create new stage class inheriting from `PipelineStage`:
```python
class MyCustomStage(PipelineStage):
    def _execute(self) -> bool:
        # Your extraction logic here
        work_dir = Path(self.config.get("output.work_dir"))
        # Process data and save to work_dir / "my_data"
        return True
```

2. Add stage to pipeline in `DataCollectionPipeline.__init__()`:
```python
self.stages = [
    ExtensionDataStage("extension_data", self.config, self.logger),
    # ... existing stages ...
    MyCustomStage("my_custom_stage", self.config, self.logger),
]
```

3. Update file mappings in `FinalAssemblyStage._get_file_mappings()`:
```python
# My custom data
custom_src = work_dir / "my_data"
custom_dst = target_dir / "my_data" / kit_version
mappings.append((custom_src / "my_output.json", custom_dst / "my_output.json"))
```

4. Optionally add embeddings and FAISS stages for your data type

### Custom Processing

The pipeline is modular - you can:
- **Skip stages** using `--start`/`--end` flags for partial runs
- **Modify individual components** in their respective directories
- **Add custom filters** via the exclusion list in config
- **Run multiple versions** by changing the branch in config
- **Integrate with CI/CD** for automated data updates

## Performance Considerations

### Runtime Metrics (for ~400-500 extensions)

- **Pull Repo & Build**: 10-30 minutes (only if cloning and building)
- **Extension Data**: 5-15 minutes (depends on Code Atlas generation)
- **Code Examples**: 3-8 minutes (Code Atlas parsing)
- **Settings**: 2-5 minutes (file scanning)
- **Embeddings**: 5-15 minutes (depends on API rate limits)
- **FAISS**: 1-3 minutes (vector index creation)
- **Total Runtime**: 20-60 minutes for full pipeline

### Resource Requirements

- **Disk space**:
  - Build directory (if cloning): ~8-12GB
  - Work directory: ~500MB-1GB intermediate files
  - Output: ~200MB-2GB (depends on `include_source_code`)

- **Memory usage**:
  - Peak: ~2-8GB (scales with extension count and Code Atlas size)
  - Typical: ~1-4GB during processing

- **Network**:
  - Required for repository cloning (if using)
  - Required for embedding generation (NVIDIA API)
  - ~100-500MB transfer for embeddings

### Optimization Strategies

For large Kit installations:
- **Use `run_pipeline.py`** with existing build to skip clone/build
- **Limit processing** with `--max-extensions` for testing/iteration
- **Use `--quick` mode** to skip embeddings/FAISS for rapid iteration
- **Process in stages** using `--start`/`--end` for partial updates
- **Disable source code** with `include_source_code = false` to save 50-70% space
- **Run on build servers** for regular automated updates
- **Use SSD storage** for faster file I/O during processing

## Kit Version Compatibility

The pipeline supports:
- **Kit 105.x and newer** (tested up to 107.x)
- **Extension.toml** format v2+
- **Python 3.11** code analysis via lc_agent
- **C++ source scanning** for settings (basic pattern matching)
- **Multiple Kit versions** via versioned output structure

### Multi-Version Support

The pipeline automatically versions output by Kit version, allowing you to maintain data for multiple Kit versions:

```bash
# Process Kit 106.x
python data_collection_pipeline.py  # with branch = "production/106.x"

# Process Kit 107.x
python data_collection_pipeline.py  # with branch = "production/107.3"

# Both versions are preserved in the output structure
```

### Upgrading Kit Versions

When upgrading to a new Kit version:

1. **Update configuration**:
   ```toml
   [input]
   branch = "production/107.3"  # New version branch
   ```

2. **Run pipeline** - it will create new versioned directories alongside existing ones

3. **Verify output** - check that new version directory was created:
   ```
   target_dir/
   ├── extensions/
   │   ├── 106.x/
   │   └── 107.3/  ← New version
   ```

4. **Update service** to use the new version (see Integration section below)

## Integration with MCP Service

The generated data is automatically formatted for the `kit_fns` MCP service:

### Data Format Compatibility

1. **File locations** - Versioned structure matches service expectations
2. **JSON schemas** - Compatible with service data models
3. **FAISS databases** - Ready for semantic search queries
4. **Code Atlas format** - Matches lc_agent's expected format
5. **Token counts** - Pre-calculated for LLM context management

### Using Generated Data

After running the pipeline, the kit_fns service can load the new data:

```bash
# Data is copied to ../src/kit_fns/data/ by default
cd ../src/kit_fns/

# The MCP service automatically detects and loads:
# - extensions/{version}/
# - code_examples/{version}/
# - settings/{version}/
```

### Version Selection

The service can be configured to use specific Kit versions:
```python
# In service configuration
kit_version = "107.3"  # Use specific version
# or
kit_version = "latest"  # Use most recent
```

### Continuous Integration

For automated updates:

```bash
# CI/CD pipeline example
cd source/aiq/kit_fns/data_collection

# Run pipeline for latest production branch
python data_collection_pipeline.py

# Deploy generated data
# (data is already in target_dir, ready for MCP service)

# Restart MCP service to pick up changes
# (service-specific restart procedure)
```