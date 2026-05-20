# Isaac Sim Settings Pipeline Architecture

## Overview

This document provides a comprehensive guide to the NVIDIA Omniverse Isaac Sim settings pipeline, detailing how settings are collected, processed, embedded, and made searchable through a FAISS vector database. This pipeline enables intelligent semantic search and retrieval of Isaac Sim configuration settings across all extensions.

## Table of Contents

1. [Background](#background)
2. [Settings Collection Pipeline](#settings-collection-pipeline)
3. [Data Structure Design](#data-structure-design)
4. [Embedding Generation Pipeline](#embedding-generation-pipeline)
5. [FAISS Database Construction](#faiss-database-construction)
6. [MCP Implementation Guide](#mcp-implementation-guide)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

---

## Background

### What are Isaac Sim Settings?

NVIDIA Omniverse Isaac Sim uses the Kit settings system based on Carbonite settings. These settings control everything from physics simulation parameters to rendering options, robot behaviors, and sensor configurations. Settings follow a hierarchical path-based structure using forward slashes (e.g., `/physics/updateFPS`, `/isaac/sim/default/nucleus/path`).

### The Challenge

With Isaac Sim's extensive set of extensions and specialized robotics/simulation settings, developers face challenges in:
- Finding the right physics simulation parameters
- Understanding robot and sensor configuration options
- Discovering Isaac-specific settings vs general Kit settings
- Tracking which extensions use specific settings
- Configuring simulation behavior correctly

### The Solution

Our pipeline creates a comprehensive, searchable database of all Isaac Sim settings by:
1. Scanning Isaac Sim extension source code and configuration files
2. Extracting settings with their metadata and documentation
3. Generating semantic embeddings for intelligent search
4. Building a FAISS vector database for efficient similarity search

---

## Settings Collection Pipeline

### 1. Scanner Architecture

The settings scanner is designed to extract settings from multiple sources within Isaac Sim extensions:

#### Source Types
1. **Extension TOML files** (`extension.toml`, `config/*.toml`)
   - Settings defined in `[settings]` sections
   - Default values and types
   - Documentation comments

2. **Python source files** (`*.py`)
   - Runtime settings access via `carb.settings`
   - Dynamic setting creation
   - Settings used in simulation logic
   - Robot and sensor configuration

3. **C++ source files** (`*.cpp`, `*.h`, `*.hpp`)
   - Native Carbonite settings API usage
   - Low-level physics and simulation settings

#### Key Components

```python
class ExtensionSettingsScanner:
    def __init__(self, extensions_dir: str):
        self.extensions_dir = Path(extensions_dir)
        self.settings_data = defaultdict(lambda: {
            "default_value": None,
            "type": None,
            "description": None,
            "documentation": None,
            "found_in": [],
            "extensions": set()
        })
```

### 2. Setting Path Normalization

Isaac Sim settings can appear in different formats:
- **TOML format**: `exts."omni.isaac.core".physics.updateFPS`
- **Code format**: `/exts/omni.isaac.core/physics/updateFPS`

The scanner normalizes all paths to the canonical slash format for consistency.

#### Conversion Rules
1. Extension names with dots are preserved: `omni.isaac.core`
2. Setting paths use forward slashes: `/physics/updateFPS`
3. Common prefixes:
   - `/exts/` - Extension-specific settings
   - `/app/` - Application settings
   - `/persistent/` - Settings saved between sessions
   - `/rtx/` - Rendering settings
   - `/physics/` - Physics simulation settings
   - `/isaac/` - Isaac Sim specific settings

### 3. Data Extraction Process

#### From TOML Files

```python
def scan_extension_toml(self, toml_path: Path, extension_name: str):
    # Parse TOML structure
    data = toml.loads(content)

    # Extract settings with documentation
    if 'settings' in data:
        for key, value in data['settings'].items():
            # Extract comment documentation
            doc = self.extract_toml_comment(lines, line_num)
            # Convert to canonical path
            canonical_key = self.convert_dot_to_slash(key)
            # Store with metadata
```

**Example Isaac Sim TOML Setting:**
```toml
[settings]
# Default physics update frequency for simulation
exts."omni.isaac.core".physics.updateFPS = 60

# Default nucleus path for Isaac Sim assets
exts."omni.isaac.sim".default.nucleus.path = "omniverse://localhost/NVIDIA/Assets/Isaac"
```

#### From Python Code

The scanner uses regex patterns to find settings usage:
```python
patterns = [
    # settings.get("/path/to/setting")
    (r'settings\.get(?:_as_\w+)?\s*\(\s*["\']([^"\']+)["\']', 'get'),
    # settings.set("/path/to/setting", value)
    (r'settings\.set\s*\(\s*["\']([^"\']+)["\'](?:\s*,\s*([^,\)]+))?', 'set'),
]
```

### 4. Quality Control

The scanner implements several quality control measures:

#### Partial Path Filtering
Incomplete paths like `/exts`, `/app`, `/physics` are detected and excluded as they represent path prefixes used in code, not actual settings.

#### Type Inference
Setting types are inferred from:
1. TOML value types (bool, int, float, string, array, object)
2. Python literal evaluation
3. Method names (e.g., `get_as_int()` implies integer type)

#### Documentation Extraction
Documentation is gathered from:
1. Comments above TOML settings
2. Docstrings in Python code
3. Adjacent comments describing physics or robot parameters
4. Isaac Sim specific configuration guides

---

## Data Structure Design

### Settings Summary Format (`setting_summary.json`)

```json
{
  "metadata": {
    "total_settings": 850,
    "total_extensions_scanned": 120,
    "scan_directory": "/path/to/isaac-sim/extensions",
    "kit_version": "109.0",
    "version": "1.0.0"
  },
  "settings": {
    "/physics/updateFPS": {
      "default_value": 60,
      "type": "int",
      "description": null,
      "documentation": "Physics simulation update frequency in frames per second",
      "extensions": [
        "omni.isaac.core",
        "omni.isaac.dynamic_control"
      ],
      "found_in": [
        "omni.isaac.core-2.0.0/config/extension.toml@45",
        "omni.isaac.dynamic_control/python/scripts/control.py@120"
      ],
      "usage_count": 12
    },
    "/isaac/sim/default/nucleus/path": {
      "default_value": "omniverse://localhost/NVIDIA/Assets/Isaac",
      "type": "string",
      "description": null,
      "documentation": "Default nucleus server path for Isaac Sim assets",
      "extensions": [
        "omni.isaac.sim"
      ],
      "found_in": [
        "omni.isaac.sim-1.0.0/config/extension.toml@23"
      ],
      "usage_count": 8
    }
  }
}
```

### Key Design Decisions

1. **Flat Structure**: Settings are stored in a flat dictionary with canonical paths as keys for efficient lookup.

2. **Extension Association**: Each setting tracks all extensions that use it, enabling cross-reference analysis.

3. **Location Tracking**: The `found_in` field uses a compact format (`file@line`) for source traceability.

4. **Usage Metrics**: `usage_count` helps identify commonly used settings, particularly important for physics and robot configuration.

---

## Embedding Generation Pipeline

### 1. Document Creation Strategy

Each setting is converted into a comprehensive text document for embedding:

```python
def create_setting_document(setting_key: str, setting_data: Dict) -> str:
    doc_parts = []
    doc_parts.append(f"Setting: {setting_key}")
    doc_parts.append(f"Type: {setting_data['type']}")
    doc_parts.append(f"Default: {setting_data['default_value']}")
    doc_parts.append(f"Documentation: {setting_data['documentation']}")
    doc_parts.append(f"Used by extensions: {', '.join(setting_data['extensions'])}")
    doc_parts.append(f"Usage count: {setting_data['usage_count']}")

    # Add category hints for better semantic matching
    if '/physics/' in setting_key:
        doc_parts.append("Category: Physics simulation")
    elif '/isaac/' in setting_key:
        doc_parts.append("Category: Isaac Sim specific")
    elif '/robot/' in setting_key:
        doc_parts.append("Category: Robot configuration")

    return "\n".join(doc_parts)
```

### 2. Embedding Generation Process

#### Model Selection
- **Model**: NVIDIA `nv-embedqa-e5-v5`
- **Dimension**: 1024 (typical for this model)
- **Truncation**: END (truncate from the end if text is too long)

#### Batch Processing
Settings are processed in batches of 50 for efficiency:
```python
for i in range(0, len(settings_list), batch_size):
    batch = settings_list[i:i+batch_size]
    embeddings = embedder.embed_documents(batch_texts)
```

### 3. Embeddings Storage Format (`settings_embeddings.json`)

```json
{
  "metadata": {
    "model": "nvidia/nv-embedqa-e5-v5",
    "total_settings": 850,
    "embedding_dimension": 1024,
    "generated_at": "2024-01-15T10:30:00",
    "kit_version": "109.0",
    "successful_embeddings": 850,
    "failed_embeddings": 0
  },
  "settings": {
    "/physics/updateFPS": {
      "embedding": [0.0234, -0.0156, ...],  // 1024-dimensional vector
      "type": "int",
      "default_value": 60,
      "extensions_count": 2,
      "usage_count": 12,
      "has_documentation": true
    }
  }
}
```

---

## FAISS Database Construction

### 1. Database Building Process

The FAISS database combines embeddings with searchable metadata:

```python
def build_faiss_database():
    # Load embeddings and metadata
    embeddings_data = load_json("settings_embeddings.json")
    settings_data = load_json("setting_summary.json")

    # Create LangChain documents
    for setting_key, embedding_data in embeddings_data['settings'].items():
        doc = Document(
            page_content=create_setting_page_content(setting_key, setting_info),
            metadata={
                "setting_key": setting_key,
                "type": setting_info.get('type'),
                "prefix": extract_prefix(setting_key),
                "extension_name": extract_extension_name(setting_key),
                "category": categorize_setting(setting_key),
                ...
            }
        )

    # Build FAISS index
    vectorstore = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(documents, embeddings)],
        embedding=embedder,
        metadatas=[doc.metadata for doc in documents]
    )
```

### 2. Metadata Enhancement

Each document in the FAISS database includes rich metadata for filtering:

```python
metadata = {
    "setting_key": "/physics/updateFPS",
    "type": "int",
    "default_value": "60",
    "has_documentation": true,
    "usage_count": 12,
    "prefix": "physics",
    "category": "physics_simulation",
    "documentation": "Physics simulation update frequency...",
    "sample_extensions": "omni.isaac.core, omni.isaac.dynamic_control",
    "sample_locations": "extension.toml@45, control.py@120"
}
```

### 3. Index Structure

The FAISS database uses:
- **Index Type**: Flat L2 (exact search for highest quality)
- **Distance Metric**: L2 (Euclidean distance)
- **Storage**: Local file system with pickle serialization

---

## MCP Implementation Guide

### 1. Integration Architecture

```python
class SettingsSearchService:
    def __init__(self, kit_version: str = "109.0"):
        self.kit_version = kit_version
        self.faiss_db_path = Path(f"data/{kit_version}/settings/settings_faiss")
        self.embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
        self.vectorstore = None
        self._load_faiss_database()

    def _load_faiss_database(self):
        """Load the pre-built FAISS database."""
        if self.faiss_db_path.exists():
            self.vectorstore = FAISS.load_local(
                str(self.faiss_db_path),
                self.embedder,
                allow_dangerous_deserialization=True
            )
```

### 2. Search Implementation

```python
def search_isaac_sim_settings(
    self,
    query: str,
    top_k: int = 20,
    prefix_filter: Optional[str] = None,
    type_filter: Optional[str] = None
):
    """Search for settings using semantic similarity."""
    if not self.vectorstore:
        return []

    # Build filters
    filters = {}
    if prefix_filter:
        filters["prefix"] = prefix_filter
    if type_filter:
        filters["type"] = type_filter

    # Perform similarity search
    results = self.vectorstore.similarity_search_with_score(
        query,
        k=top_k,
        filter=filters
    )

    # Format results
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "setting": doc.metadata['setting_key'],
            "type": doc.metadata['type'],
            "default": doc.metadata['default_value'],
            "documentation": doc.metadata.get('documentation', ''),
            "extensions": doc.metadata.get('sample_extensions', '').split(', '),
            "relevance_score": float(1.0 / (1.0 + score))
        })

    return formatted_results
```

### 3. MCP Tool Definition

```python
@register_function(config_type=SearchSettingsConfig)
async def search_isaac_sim_settings(
    query: str,
    top_k: int = 20,
    prefix_filter: Optional[str] = None,
    type_filter: Optional[str] = None
) -> Dict:
    """
    Search Isaac Sim settings using semantic search.

    Args:
        query: Natural language search query
        top_k: Number of results to return
        prefix_filter: Filter by prefix (exts, app, persistent, rtx, physics, isaac)
        type_filter: Filter by type (bool, int, float, string, array, object)

    Returns:
        Formatted text with matching settings and metadata
    """
    results = settings_service.search_isaac_sim_settings(
        query, top_k, prefix_filter, type_filter
    )

    # Format for LLM consumption
    return format_settings_results(results)
```

### 4. Usage in MCP Context

```python
# Example queries for Isaac Sim

# Search physics settings
results = await search_isaac_sim_settings(
    query="physics simulation frequency and timestep",
    prefix_filter="physics",
    top_k=10
)

# Find robot configuration settings
results = await search_isaac_sim_settings(
    query="robot articulation configuration",
    prefix_filter="exts",
    type_filter="bool"
)

# Locate Isaac Sim specific settings
results = await search_isaac_sim_settings(
    query="nucleus path and asset loading",
    prefix_filter="isaac"
)

# Find rendering settings for sensors
results = await search_isaac_sim_settings(
    query="camera sensor rendering quality",
    prefix_filter="rtx"
)
```

---

## Usage Examples

The settings database ships pre-built in the wheel; no pipeline regeneration is needed for normal use.

### Querying the Database

```python
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load the database
embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
vectorstore = FAISS.load_local(
    "data/109.0/settings/settings_faiss",
    embedder,
    allow_dangerous_deserialization=True
)

# Search for settings
results = vectorstore.similarity_search(
    "how to configure physics simulation rate",
    k=5
)

for doc in results:
    print(f"Setting: {doc.metadata['setting_key']}")
    print(f"Type: {doc.metadata['type']}")
    print(f"Default: {doc.metadata['default_value']}")
    print(f"Documentation: {doc.metadata.get('documentation', 'N/A')}")
    print("---")
```

### 3. Common Search Patterns

#### Find Physics Settings
```python
results = vectorstore.similarity_search(
    "physics timestep and update frequency",
    filter={"prefix": "physics"}
)
```

#### Find Isaac Sim Specific Settings
```python
results = vectorstore.similarity_search(
    "isaac sim default configuration",
    filter={"prefix": "isaac"}
)
```

#### Find Boolean Flags
```python
results = vectorstore.similarity_search(
    "enable disable simulation features",
    filter={"type": "bool"}
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Missing Embeddings API Key
**Error**: `No NVIDIA_API_KEY found in environment variables`

**Solution**:
```bash
export NVIDIA_API_KEY='your-api-key'
# Get key from: https://build.nvidia.com/
```

#### 2. Incomplete Settings Extraction
**Issue**: Some Isaac Sim settings are missing from the scan

**Possible Causes**:
- Settings defined dynamically at runtime
- Settings in non-standard locations
- Settings in compiled extensions

**Solution**:
- Check scan logs for warnings
- Verify extension directory structure
- Add custom patterns for Isaac-specific usage

#### 3. FAISS Database Load Errors
**Error**: `ValueError: allow_dangerous_deserialization must be True`

**Solution**:
```python
vectorstore = FAISS.load_local(
    path,
    embedder,
    allow_dangerous_deserialization=True  # Required for pickle
)
```

#### 4. Embedding Dimension Mismatch
**Error**: `Embedding dimension mismatch`

**Cause**: Using different embedding models for generation and search

**Solution**: Ensure consistent model usage:
```python
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"  # Use same model everywhere
```

### Performance Optimization

#### 1. Batch Processing
Adjust batch size based on available memory:
```python
batch_size = 50  # Increase for faster processing if memory allows
```

#### 2. Incremental Updates
For continuous integration, implement incremental scanning:
```python
def scan_modified_extensions(since_timestamp):
    """Scan only extensions modified since given timestamp."""
    for ext_dir in extension_dirs:
        if get_modification_time(ext_dir) > since_timestamp:
            scan_extension(ext_dir)
```

---

## Appendix

### A. File Structure

```
source/aiq/isaacsim_fns/
└── src/isaacsim_fns/data/
    └── {kit_version}/                      # e.g., 109.0
        └── settings/
            ├── setting_summary.json        # Complete settings database
            ├── setting_summary_simple.json # Simplified lookup
            ├── setting_statistics.json     # Statistics
            └── settings_faiss/            # FAISS database
                ├── index.faiss
                ├── index.pkl
                └── metadata.json
```

### B. Setting Path Examples

Common Isaac Sim setting paths and their purposes:

| Path Pattern | Purpose | Example |
|--------------|---------|---------|
| `/physics/*` | Physics simulation settings | `/physics/updateFPS`, `/physics/timeStepsPerSecond` |
| `/isaac/*` | Isaac Sim specific settings | `/isaac/sim/default/nucleus/path` |
| `/exts/omni.isaac.*/*` | Isaac extension settings | `/exts/omni.isaac.core/startup/physics` |
| `/persistent/isaac/*` | Saved Isaac preferences | `/persistent/isaac/asset_path` |
| `/rtx/*` | Rendering for sensors | `/rtx/raytracing/enabled` |
| `/app/robot/*` | Robot configuration | `/app/robot/defaultController` |

### C. Data Flow Diagram

```
Isaac Sim Extensions Directory
        │
        ▼
[Scanner (scan_extension_settings.py)]
        │
        ├─► setting_summary.json
        │   (Settings with metadata)
        │
        ▼
[Embeddings Generator (generate_settings_embeddings.py)]
        │
        ├─► settings_embeddings.json
        │   (Vector representations)
        │
        ▼
[FAISS Builder (build_settings_faiss_database.py)]
        │
        ├─► settings_faiss/
        │   (Searchable vector database)
        │
        ▼
[MCP Service Implementation]
        │
        ▼
[User Queries via Semantic Search]
```

---

## Conclusion

The Isaac Sim Settings Pipeline provides a comprehensive solution for discovering, understanding, and searching NVIDIA Omniverse Isaac Sim configuration settings. By combining static analysis, semantic embeddings, and vector search, it transforms scattered settings across extensions into an intelligent, searchable knowledge base.

This architecture enables:
- **Semantic search**: Find settings using natural language queries about physics, robots, sensors
- **Cross-reference analysis**: Understand which extensions use which settings
- **Documentation discovery**: Surface settings with their purposes and defaults
- **Type-safe configuration**: Understand setting types and valid values
- **Efficient retrieval**: Fast similarity search via FAISS indexing
- **Robotics focus**: Special attention to physics and simulation configuration

The pipeline is designed to be extensible, maintainable, and integrable with Model Context Protocol (MCP) services, providing a foundation for intelligent Isaac Sim configuration assistance.
