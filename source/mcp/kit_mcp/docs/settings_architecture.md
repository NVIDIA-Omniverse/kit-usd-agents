# Kit Settings Pipeline Architecture

## Overview

This document provides a comprehensive guide to the NVIDIA Omniverse Kit settings pipeline, detailing how settings are collected, processed, embedded, and made searchable through a FAISS vector database. This pipeline enables intelligent semantic search and retrieval of Kit configuration settings across hundreds of extensions.

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

### What are Kit Settings?

NVIDIA Omniverse Kit uses a sophisticated configuration system based on Carbonite settings. These settings control everything from rendering options to extension behavior, UI preferences, and system configurations. Settings follow a hierarchical path-based structure using forward slashes (e.g., `/exts/omni.kit.viewport.window/enabled`).

### The Challenge

With over 400 extensions in a typical Kit application and over 1,000 unique settings, developers face challenges in:
- Finding the right settings for their needs
- Understanding setting relationships and dependencies
- Discovering undocumented or poorly documented settings
- Tracking which extensions use specific settings

### The Solution

Our pipeline creates a comprehensive, searchable database of all Kit settings by:
1. Scanning extension source code and configuration files
2. Extracting settings with their metadata
3. Generating semantic embeddings for intelligent search
4. Building a FAISS vector database for efficient similarity search

---

## Settings Collection Pipeline

### 1. Scanner Architecture (`scan_extension_settings.py`)

The settings scanner is designed to extract settings from multiple sources within Kit extensions:

#### Source Types
1. **Extension TOML files** (`extension.toml`, `config/*.toml`)
   - Settings defined in `[settings]` sections
   - Default values and types
   - Documentation comments

2. **Python source files** (`*.py`)
   - Runtime settings access via `carb.settings`
   - Dynamic setting creation
   - Settings used in code logic

3. **C++ source files** (`*.cpp`, `*.h`, `*.hpp`)
   - Native Carbonite settings API usage
   - Low-level settings access

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

Kit settings can appear in different formats:
- **TOML format**: `exts."omni.kit.viewport.menubar.timeline".buttonOrder`
- **Code format**: `/exts/omni.kit.viewport.menubar.timeline/buttonOrder`

The scanner normalizes all paths to the canonical slash format for consistency.

#### Conversion Rules
1. Extension names with dots are preserved: `omni.kit.viewport.menubar.timeline`
2. Setting paths use forward slashes: `/buttonOrder`
3. Common prefixes:
   - `/exts/` - Extension-specific settings
   - `/app/` - Application settings
   - `/persistent/` - Settings saved between sessions
   - `/rtx/` - Rendering settings

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

**Example TOML Setting:**
```toml
[settings]
# Define the button order in the viewport menubar
exts."omni.kit.viewport.menubar.timeline".buttonOrder = 95
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
Incomplete paths like `/exts`, `/app`, `/persistent` are detected and excluded as they represent path prefixes used in code, not actual settings.

#### Type Inference
Setting types are inferred from:
1. TOML value types (bool, int, float, string, array, object)
2. Python literal evaluation
3. Method names (e.g., `get_as_bool()` implies boolean type)

#### Documentation Extraction
Documentation is gathered from:
1. Comments above TOML settings
2. Docstrings in Python code
3. Adjacent comments in source files

---

## Data Structure Design

### Settings Summary Format (`setting_summary.json`)

```json
{
  "metadata": {
    "total_settings": 1004,
    "total_extensions_scanned": 407,
    "scan_directory": "/path/to/extensions",
    "version": "1.0.0"
  },
  "settings": {
    "/exts/omni.kit.viewport.menubar.timeline/buttonOrder": {
      "default_value": 95,
      "type": "int",
      "description": null,
      "documentation": "Define the button order in the viewport menubar",
      "extensions": [
        "omni.kit.viewport.menubar.timeline"
      ],
      "found_in": [
        "omni.kit.viewport.menubar.timeline-1.0.2/config/extension.toml@42"
      ],
      "usage_count": 1
    }
  }
}
```

### Key Design Decisions

1. **Flat Structure**: Settings are stored in a flat dictionary with canonical paths as keys for efficient lookup.

2. **Extension Association**: Each setting tracks all extensions that use it, enabling cross-reference analysis.

3. **Location Tracking**: The `found_in` field uses a compact format (`file@line`) for source traceability.

4. **Usage Metrics**: `usage_count` helps identify commonly used settings.

---

## Embedding Generation Pipeline

### 1. Document Creation Strategy (`generate_settings_embeddings.py`)

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
    return "\n".join(doc_parts)
```

### 2. Embedding Generation Process

#### Model Selection
- **Model**: NVIDIA `nv-embedqa-e5-v5`
- **Dimension**: 1024 (typical for this model)
- **Truncation**: END (truncate from the end if text is too long)

#### Batch Processing
Settings are processed in batches of 10 for efficiency:
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
    "total_settings": 1004,
    "embedding_dimension": 1024,
    "generated_at": "2024-01-15T10:30:00",
    "successful_embeddings": 1004,
    "failed_embeddings": 0
  },
  "settings": {
    "/exts/omni.kit.viewport.menubar.timeline/buttonOrder": {
      "embedding": [0.0234, -0.0156, ...],  // 1024-dimensional vector
      "type": "int",
      "default_value": 95,
      "extensions_count": 1,
      "usage_count": 1,
      "has_documentation": true
    }
  }
}
```

---

## FAISS Database Construction

### 1. Database Building Process (`build_settings_faiss_database.py`)

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
    "setting_key": "/exts/omni.kit.viewport.menubar.timeline/buttonOrder",
    "type": "int",
    "default_value": "95",
    "has_documentation": true,
    "usage_count": 1,
    "prefix": "exts",
    "extension_name": "omni.kit.viewport.menubar.timeline",
    "documentation": "Define the button order...",
    "sample_extensions": "omni.kit.viewport.menubar.timeline",
    "sample_locations": "extension.toml@42"
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
    def __init__(self):
        self.faiss_db_path = Path("setting_data/settings_faiss")
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
def search_kit_settings(self, query: str, top_k: int = 10, filters: Dict = None):
    """Search for settings using semantic similarity."""
    if not self.vectorstore:
        return []
    
    # Perform similarity search
    results = self.vectorstore.similarity_search_with_score(
        query, 
        k=top_k,
        filter=filters  # e.g., {"prefix": "exts", "type": "bool"}
    )
    
    # Format results
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "setting": doc.metadata['setting_key'],
            "type": doc.metadata['type'],
            "default": doc.metadata['default_value'],
            "documentation": doc.metadata.get('documentation', ''),
            "relevance_score": float(1.0 / (1.0 + score))
        })
    
    return formatted_results
```

### 3. MCP Tool Definition

```python
@tool(
    name="search_kit_settings",
    description="Search for NVIDIA Omniverse Kit configuration settings"
)
async def search_kit_settings(
    query: str,
    top_k: int = 10,
    prefix_filter: Optional[str] = None,
    type_filter: Optional[str] = None
) -> List[Dict]:
    """
    Search Kit settings using semantic search.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
        prefix_filter: Filter by prefix (exts, app, persistent, rtx)
        type_filter: Filter by type (bool, int, float, string, array)
    
    Returns:
        List of matching settings with metadata
    """
    filters = {}
    if prefix_filter:
        filters["prefix"] = prefix_filter
    if type_filter:
        filters["type"] = type_filter
    
    return settings_service.search_kit_settings(query, top_k, filters)
```

### 4. Usage in MCP Context

```python
# Example queries
results = await search_kit_settings(
    query="viewport rendering settings",
    top_k=5,
    prefix_filter="exts"
)

# Get specific extension settings
results = await search_kit_settings(
    query="omni.kit.viewport.window configuration",
    type_filter="bool"
)

# Find persistent settings
results = await search_kit_settings(
    query="settings saved between sessions",
    prefix_filter="persistent"
)
```

---

## Usage Examples

### 1. Running the Pipeline

```bash
# Step 1: Scan extensions for settings
cd source/mcp/kit_mcp/data_collection/settings_pipeline
python3 scan_extension_settings.py

# Step 2: Generate embeddings
export NVIDIA_API_KEY='your-api-key'
python3 generate_settings_embeddings.py

# Step 3: Build FAISS database
python3 build_settings_faiss_database.py
```

### 2. Querying the Database

```python
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load the database
embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
vectorstore = FAISS.load_local("setting_data/settings_faiss", embedder)

# Search for settings
results = vectorstore.similarity_search(
    "how to enable viewport grid",
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

#### Find Settings by Extension
```python
results = vectorstore.similarity_search(
    "omni.kit.viewport.window settings",
    filter={"extension_name": "omni.kit.viewport.window"}
)
```

#### Find Settings by Type
```python
results = vectorstore.similarity_search(
    "boolean flags for rendering",
    filter={"type": "bool", "prefix": "rtx"}
)
```

#### Find Undocumented Settings
```python
results = vectorstore.similarity_search(
    query,
    filter={"has_documentation": False}
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
**Issue**: Some settings are missing from the scan

**Possible Causes**:
- Settings defined dynamically at runtime
- Settings in non-standard locations
- Settings in binary/compiled extensions

**Solution**: 
- Check scan logs for warnings
- Verify extension directory structure
- Add custom patterns for non-standard usage

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
batch_size = 10  # Increase for faster processing if memory allows
```

#### 2. Parallel Scanning
For large extension sets, consider parallel processing:
```python
from multiprocessing import Pool

def scan_extension_parallel(ext_dir):
    scanner = ExtensionSettingsScanner(base_dir)
    return scanner.scan_extension(ext_dir)

with Pool(processes=4) as pool:
    results = pool.map(scan_extension_parallel, extension_dirs)
```

#### 3. Incremental Updates
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
source/mcp/kit_mcp/
├── data_collection/
│   └── settings_pipeline/
│       ├── scan_extension_settings.py      # Settings scanner
│       ├── generate_settings_embeddings.py # Embedding generator
│       ├── build_settings_faiss_database.py # FAISS builder
│       └── setting_data/                   # Output directory
│           ├── setting_summary.json        # Scanned settings
│           ├── setting_summary_simple.json # Simplified lookup
│           ├── setting_statistics.json     # Statistics
│           ├── settings_embeddings.json    # Embeddings
│           └── settings_faiss/            # FAISS database
│               ├── index.faiss
│               ├── index.pkl
│               └── metadata.json
└── docs/
    └── settings_architecture.md           # This document
```

### B. Setting Path Examples

Common setting paths and their purposes:

| Path Pattern | Purpose | Example |
|--------------|---------|---------|
| `/app/window/*` | Application window settings | `/app/window/title` |
| `/exts/{extension}/*` | Extension-specific settings | `/exts/omni.kit.viewport.window/enabled` |
| `/persistent/app/*` | Settings saved between sessions | `/persistent/app/viewport/camMoveVelocity` |
| `/rtx/*` | RTX rendering settings | `/rtx/rendermode` |
| `/renderer/*` | General renderer settings | `/renderer/active` |
| `/physics/*` | Physics simulation settings | `/physics/updateFPS` |

### C. Data Flow Diagram

```
Extensions Directory
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

The Kit Settings Pipeline provides a comprehensive solution for discovering, understanding, and searching NVIDIA Omniverse Kit configuration settings. By combining static analysis, semantic embeddings, and vector search, it transforms thousands of scattered settings into an intelligent, searchable knowledge base.

This architecture enables:
- **Semantic search**: Find settings using natural language queries
- **Cross-reference analysis**: Understand which extensions use which settings
- **Documentation discovery**: Surface undocumented or poorly documented settings
- **Type-safe configuration**: Understand setting types and valid values
- **Efficient retrieval**: Fast similarity search via FAISS indexing

The pipeline is designed to be extensible, maintainable, and integrable with Model Context Protocol (MCP) services, providing a foundation for intelligent Kit configuration assistance.