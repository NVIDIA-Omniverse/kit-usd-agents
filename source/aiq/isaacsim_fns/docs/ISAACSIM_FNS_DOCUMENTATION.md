# Isaac Sim Functions Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Functionalities](#key-functionalities)
4. [Services Layer](#services-layer)
5. [Functions Reference](#functions-reference)
6. [Data Sources](#data-sources)
7. [Usage Patterns](#usage-patterns)

---

## Overview

The **Isaac Sim Functions** module is a comprehensive AI-powered documentation and code discovery system for NVIDIA Omniverse Isaac Sim. It provides AI models with direct access to Isaac Sim's ecosystem of extensions, APIs, configuration settings, and code examples.

### What It Does

The Isaac Sim Functions module serves as an intelligent bridge between AI models and the Omniverse Isaac Sim ecosystem, enabling:

- **Semantic Search**: Find relevant extensions, settings, and code examples using natural language queries
- **Documentation Retrieval**: Access detailed extension information and system instructions
- **Code Discovery**: Locate implementation examples across the Isaac Sim codebase
- **Configuration Discovery**: Find and understand Isaac Sim configuration settings

### Core Purpose

This system eliminates the need for AI models to navigate complex documentation hierarchies or search through massive codebases manually. Instead, it provides curated, context-aware access to exactly the information needed for Isaac Sim development tasks.

---

## Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                   AI Model (via AIQ/MCP Protocol)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Function Registration Layer (AIQ)               │
│  • Function registration with Pydantic schemas               │
│  • Input validation and flexible format handling            │
│  • Usage logging and telemetry integration                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Functions Layer                           │
│  5 async functions implementing tool logic                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Services Layer                            │
│  • ExtensionService: Extension search & metadata            │
│  • CodeSearchService: Code example search                   │
│  • SettingsService: Configuration settings search           │
│  • TelemetryService: Usage tracking (Redis)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  • FAISS Vector Databases (semantic search)                 │
│  • JSON Metadata Files (extensions_database.json)           │
│  • Code examples (per-extension analysis)                   │
│  • Settings database (setting_summary.json)                 │
│  • Instruction Sets (markdown guides)                       │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **AIQ Framework**: Tool registration and workflow management
- **LangChain**: Vector store and embedding integrations
- **FAISS**: High-performance semantic search with NVIDIA embeddings
- **Pydantic**: Input validation and schema definition
- **Redis**: Distributed telemetry and usage tracking
- **Python 3.11+**: Core implementation language

---

## Key Functionalities

### 1. **Intelligent Search Capabilities**

The system provides three specialized search functions, each optimized for different types of queries:

- **Extension Search**: Find extensions by functionality, category, or use case
- **Code Example Search**: Locate implementation patterns and working code examples
- **Settings Search**: Discover configuration options

### 2. **Detailed Information Retrieval**

Complementing search, the system offers precise retrieval functions:

- **Extension Details**: Complete metadata, features, dependencies, and API information
- **Instructions**: Development guidance and best practices

### 3. **Flexible Input Handling**

All tools support multiple input formats for maximum AI model compatibility:

- Single strings: `"omni.isaac.core"`
- Native arrays: `["omni.isaac.core", "omni.isaac.sensor"]`
- JSON strings: `'["omni.isaac.core", "omni.isaac.sensor"]'`
- Comma-separated: `"omni.isaac.core, omni.isaac.sensor"`
- Empty/null: Returns listing or help information

### 4. **Usage Telemetry**

Comprehensive telemetry captures:
- Function call patterns and frequency
- Performance metrics (execution time)
- Success/failure rates
- Parameter usage statistics
- Error tracking for system improvement

---

## Services Layer

### ExtensionService

**Purpose**: Manages Isaac Sim extension metadata and provides semantic search across the extension database.

**Key Methods**:
- `search_isaac_sim_extensions(query, top_k)`: Semantic search for extensions
- `get_isaac_sim_extension_details(extension_ids)`: Detailed extension information

**Data Sources**:
- `extensions_database.json`: Metadata for all extensions
- `extensions_faiss/`: FAISS vector database for semantic search

**Features**:
- FAISS-powered semantic search with fallback to keyword search
- Fuzzy matching for typo tolerance
- Category filtering and relevance scoring
- Extension metadata caching

### CodeSearchService

**Purpose**: Enables semantic search for code examples.

**Key Methods**:
- `search_isaac_sim_code_examples(query, top_k)`: Find production code examples

**Data Sources**:
- `code_examples_faiss/`: FAISS database of code examples
- `extracted_methods_regular/`: JSON files with method metadata

**Features**:
- Relevance scoring based on semantic similarity
- Tag-based categorization
- Source code extraction with line numbers

### SettingsService

**Purpose**: Manages Isaac Sim configuration settings with semantic search.

**Key Methods**:
- `search_isaac_sim_settings(query, top_k, prefix_filter, type_filter)`: Search settings
- `get_setting_details(setting_keys)`: Detailed setting information
- `get_settings_by_extension(extension_id)`: Extension-specific settings

**Data Sources**:
- `settings_faiss/`: FAISS database of settings
- `setting_summary.json`: Complete settings database

**Features**:
- Hierarchical setting path structure (`/app/`, `/exts/`, `/rtx/`)
- Type filtering (bool, int, float, string, array, object)
- Prefix filtering by category
- Usage tracking across extensions

### TelemetryService

**Purpose**: Centralized usage tracking using Redis Streams.

**Key Methods**:
- `capture_call(function_name, request_data, duration_ms, success, error)`: Log function calls
- `track_call(function_name, request_data)`: Context manager for automatic timing

**Features**:
- Async Redis integration
- Automatic timing and error tracking
- Session grouping capabilities
- Performance metrics collection

---

## Functions Reference

### 1. search_isaac_sim_extensions

**Description**: Search for Isaac Sim extensions using semantic search.

**Input Parameters**:
```python
{
  "query": str,        # Search query (e.g., "robotics simulation")
  "top_k": int = 10   # Number of results to return
}
```

**Returns**: Formatted text with:
- Extension names and IDs
- Relevance scores
- Brief descriptions
- Key features (top 3)
- Dependencies
- Version information

**Use Cases**:
- Finding extensions for specific functionality
- Discovering tools for robotics development
- Locating simulation or sensor extensions

**Example**:
```python
search_isaac_sim_extensions("robot control and manipulation", top_k=5)
```

---

### 2. get_isaac_sim_extension_details

**Description**: Get comprehensive information about specific Isaac Sim extensions.

**Input Parameters**:
```python
{
  "extension_ids": Optional[Union[str, List[str]]]
  # Single: "omni.isaac.core"
  # Multiple: ["omni.isaac.core", "omni.isaac.sensor"]
  # Empty/null: Lists all available extensions
}
```

**Returns**: Detailed JSON with:
- Complete extension metadata
- Features and capabilities
- Dependencies (required and optional)
- API counts and symbols (sample)
- Documentation availability
- Token counts for documentation

**Batch Processing Benefits**:
- 70% faster for multiple extensions
- Single API call vs multiple round-trips
- Efficient context window usage

**Example**:
```python
get_isaac_sim_extension_details(["omni.isaac.core", "omni.isaac.sensor"])
```

---

### 3. search_isaac_sim_code_examples

**Description**: Find relevant Isaac Sim code examples using semantic search.

**Input Parameters**:
```python
{
  "query": str,       # Description of desired functionality
  "top_k": int = 10  # Number of examples to return
}
```

**Returns**: Code examples with:
- Complete implementation code
- File paths and line numbers
- Extension IDs and context
- Descriptions and use cases
- Relevance scores
- Associated tags

**Search Capabilities**:
- Extension implementations and patterns
- Robot control examples
- Sensor usage examples
- USD operation examples
- Physics simulation patterns

**Example**:
```python
search_isaac_sim_code_examples("create robot with articulation", top_k=5)
```

---

### 4. search_isaac_sim_settings

**Description**: Search Isaac Sim configuration settings.

**Input Parameters**:
```python
{
  "query": str,                           # Setting search query
  "top_k": int = 20,                     # Number of results
  "prefix_filter": Optional[str] = None, # "exts", "app", "persistent", "rtx"
  "type_filter": Optional[str] = None    # "bool", "int", "float", "string", "array", "object"
}
```

**Returns**: Settings information with:
- Full setting paths
- Data types and default values
- Documentation (when available)
- Extensions using each setting
- Usage counts across codebase
- Source file locations

**Setting Prefixes**:
- `/exts/`: Extension-specific settings
- `/app/`: Application-level settings
- `/persistent/`: Settings saved between sessions
- `/rtx/`: RTX rendering settings
- `/renderer/`: General renderer settings
- `/physics/`: Physics simulation settings

**Example**:
```python
search_isaac_sim_settings("physics simulation", prefix_filter="physics", top_k=10)
```

---

### 5. get_isaac_sim_instructions

**Description**: Retrieve Isaac Sim system instructions and development documentation.

**Input Parameters**:
```python
{
  "instruction_sets": Optional[Union[str, List[str]]]
  # Single: "isaac_system"
  # Multiple: ["isaac_system", "extensions"]
  # Empty/null: Lists all available instruction sets
}
```

**Available Instruction Sets**:

1. **isaac_system**: Core Isaac Sim framework fundamentals
   - Extension system architecture
   - USD integration patterns
   - Application architecture
   - Simulation workflows

2. **extensions**: Extension development guidelines
   - Configuration and dependencies
   - Lifecycle management
   - Service registration patterns

3. **robotics**: Robotics development patterns
   - Robot creation and control
   - Articulation setup
   - Sensor integration

**Returns**: Formatted documentation with:
- Description and use cases
- Complete instruction content
- Markdown formatting
- Code examples and patterns

**Example**:
```python
get_isaac_sim_instructions(["isaac_system", "robotics"])
```

---

## Data Sources

### FAISS Vector Databases

**Location**: `src/isaacsim_fns/data/`

1. **extensions_faiss/**: Extension metadata embeddings
   - Isaac Sim extensions
   - Descriptions, features, categories
   - NVIDIA NV-EmbedQA-E5-v5 embeddings

2. **code_examples_faiss/**: Code example embeddings
   - Production code patterns
   - Method implementations
   - Tagged by functionality

3. **settings_faiss/**: Settings embeddings
   - Configuration settings
   - Documentation and usage patterns

### JSON Databases

1. **extensions_database.json**: Complete extension metadata
   - Title, version, category
   - Dependencies and keywords
   - API counts and documentation tokens
   - Storage paths

2. **setting_summary.json**: Settings database
   - Full setting paths
   - Types, defaults, documentation
   - Usage counts and locations

### Instruction Sets

**Location**: `src/isaacsim_fns/data/instructions/`

Markdown files containing:
- Development guidelines
- Best practices
- Code patterns
- Usage examples

---

## Usage Patterns

### Pattern 1: Extension Discovery Workflow

```python
# 1. Search for relevant extensions
search_isaac_sim_extensions("robot simulation", top_k=5)

# 2. Get detailed information
get_isaac_sim_extension_details("omni.isaac.core")
```

### Pattern 2: Learning by Example

```python
# 1. Search for code examples
search_isaac_sim_code_examples("create articulation root")

# 2. Get implementation guidance
get_isaac_sim_instructions("robotics")
```

### Pattern 3: Configuration Discovery

```python
# 1. Search for settings
search_isaac_sim_settings("physics timestep", prefix_filter="physics")

# 2. Find extension-specific settings
search_isaac_sim_settings("isaac simulation", prefix_filter="exts")

# 3. Filter by type
search_isaac_sim_settings("enable features", type_filter="bool")
```

### Pattern 4: Comprehensive Learning Path

```python
# New to Isaac Sim development? Follow this sequence:

# 1. Start with fundamentals
get_isaac_sim_instructions("isaac_system")

# 2. Learn extension development
get_isaac_sim_instructions("extensions")

# 3. Explore available extensions
search_isaac_sim_extensions("your domain of interest")

# 4. Study code examples
search_isaac_sim_code_examples("basic robot setup")

# 5. Understand robotics patterns
get_isaac_sim_instructions("robotics")
```

---

## Performance Considerations

### Semantic Search Performance

- **FAISS Queries**: ~50-200ms depending on database size
- **Embedding Generation**: Handled by NVIDIA API
- **Fallback to Keyword**: Automatic if FAISS unavailable
- **Caching**: Extension data cached after first load

### Optimization Strategies

1. **Batch Requests**: Use array inputs for multiple items
   - Example: `get_isaac_sim_extension_details(["ext1", "ext2", "ext3"])`
   - 70% faster than individual calls

2. **Appropriate top_k**: Balance between completeness and speed
   - Default values are optimized for most use cases
   - Increase only when more results needed

3. **Filter Early**: Use prefix and type filters in search
   - Reduces result set before processing
   - Improves relevance of results

4. **Cache Results**: Store frequently accessed data
   - Services implement automatic caching
   - Extension metadata cached on first load

### Telemetry Impact

- Async telemetry capture: ~1-5ms overhead
- Redis connection pooling for efficiency
- Graceful degradation if Redis unavailable
- No impact on tool functionality

---

## Error Handling

### Common Error Scenarios

1. **Invalid Input Format**:
   - Returns clear error message with expected format
   - Suggestion provided for correct usage

2. **Not Found**:
   - Fuzzy matching suggestions when available
   - Example: "Did you mean 'omni.isaac.sensor'?"
   - Lists similar items for discovery

3. **Data Unavailable**:
   - Clear indication of missing data
   - Fallback to alternative sources when possible

4. **Empty Results**:
   - Helpful message about search refinement
   - Suggestions for alternative searches
   - Links to related tools

### Error Response Format

All tools return consistent error structure:
```json
{
  "success": false,
  "error": "Detailed error message",
  "suggestion": "Helpful guidance for user",
  "result": ""
}
```

---

## Conclusion

The Isaac Sim Functions module provides a comprehensive, AI-optimized interface to the NVIDIA Omniverse Isaac Sim ecosystem. By combining semantic search, structured data access, and intelligent caching, it enables AI models to efficiently discover and utilize Isaac Sim's extensive capabilities.

The system's flexible input handling, batch processing support, and comprehensive documentation make it an essential tool for AI-assisted Isaac Sim development, from initial learning to advanced robotics application creation.
