# Kit Functions MCP Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Functionalities](#key-functionalities)
4. [Services Layer](#services-layer)
5. [MCP Tools Reference](#mcp-tools-reference)
6. [Data Sources](#data-sources)
7. [Usage Patterns](#usage-patterns)

---

## Overview

The **Kit Functions MCP** (Model Context Protocol) is a comprehensive AI-powered documentation and code discovery system for NVIDIA Omniverse Kit. It provides AI models with direct access to Kit's extensive ecosystem of 400+ extensions, thousands of APIs, configuration settings, code examples, and application templates.

### What It Does

The Kit Functions MCP serves as an intelligent bridge between AI models and the Omniverse Kit ecosystem, enabling:

- **Semantic Search**: Find relevant extensions, APIs, settings, and code examples using natural language queries
- **Documentation Retrieval**: Access detailed API documentation, extension information, and system instructions
- **Code Discovery**: Locate implementation examples and test patterns across the Kit codebase
- **Template Access**: Retrieve complete application templates with configuration files
- **Dependency Analysis**: Understand extension dependency trees and relationships

### Core Purpose

This system eliminates the need for AI models to navigate complex documentation hierarchies or search through massive codebases manually. Instead, it provides curated, context-aware access to exactly the information needed for Kit development tasks.

---

## Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                   AI Model (via MCP Protocol)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MCP Tool Registration Layer (AIQ)               │
│  • Function registration with Pydantic schemas               │
│  • Input validation and flexible format handling            │
│  • Usage logging and telemetry integration                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Functions Layer                           │
│  11 async functions implementing tool logic                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Services Layer                            │
│  • ExtensionService: Extension search & metadata            │
│  • APIService: API documentation & symbol lookup            │
│  • CodeSearchService: Code/test example search              │
│  • SettingsService: Configuration settings search           │
│  • KitExtensionsAtlasService: Code Atlas data mgmt          │
│  • TelemetryService: Usage tracking (Redis)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  • FAISS Vector Databases (semantic search)                 │
│  • JSON Metadata Files (extensions_database.json)           │
│  • Code Atlas Files (per-extension analysis)                │
│  • API Documentation (extracted docs)                       │
│  • Application Templates (README, .kit files)               │
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

The system provides four specialized search functions, each optimized for different types of queries:

- **Extension Search**: Find extensions by functionality, category, or use case across 400+ Kit extensions
- **Code Example Search**: Locate implementation patterns and working code examples
- **Test Example Search**: Find test implementations and testing patterns
- **Settings Search**: Discover configuration options from 1,000+ settings
- **Application Search**: Find appropriate application templates for your use case

### 2. **Detailed Information Retrieval**

Complementing search, the system offers precise retrieval functions:

- **Extension Details**: Complete metadata, features, dependencies, and API counts
- **API Details**: Full documentation with signatures, parameters, return types, and docstrings
- **API Listings**: Enumeration of all APIs provided by specific extensions
- **Dependency Analysis**: Recursive dependency trees with configurable depth
- **App Templates**: Complete README and .kit configuration files

### 3. **Development Guidance**

Built-in instruction sets provide comprehensive guidance:

- **kit_system**: Core Kit framework architecture and fundamentals
- **extensions**: Extension development patterns and best practices
- **testing**: Test writing strategies and framework usage
- **usd**: USD integration and scene description workflows
- **ui**: UI development with widgets and layouts

### 4. **Flexible Input Handling**

All tools support multiple input formats for maximum AI model compatibility:

- Single strings: `"omni.ui"`
- Native arrays: `["omni.ui", "omni.ui.scene"]`
- JSON strings: `'["omni.ui", "omni.ui.scene"]'`
- Comma-separated: `"omni.ui, omni.ui.scene"`
- Empty/null: Returns listing or help information

### 5. **Usage Telemetry**

Comprehensive telemetry captures:
- Function call patterns and frequency
- Performance metrics (execution time)
- Success/failure rates
- Parameter usage statistics
- Error tracking for system improvement

---

## Services Layer

### ExtensionService

**Purpose**: Manages Kit extension metadata and provides semantic search across the extension database.

**Key Methods**:
- `search_kit_extensions(query, top_k)`: Semantic search for extensions
- `get_kit_extension_details(extension_ids)`: Detailed extension information
- `get_kit_extension_dependencies(extension_id, depth, include_optional)`: Dependency analysis

**Data Sources**:
- `extensions_database.json`: Metadata for all extensions
- `extensions_faiss/`: FAISS vector database for semantic search
- Code Atlas files: Per-extension analysis data

**Features**:
- FAISS-powered semantic search with fallback to keyword search
- Fuzzy matching for typo tolerance
- Category filtering and relevance scoring
- Extension metadata caching

### APIService

**Purpose**: Provides access to Kit API documentation and symbol lookup.

**Key Methods**:
- `get_kit_api_details(api_references)`: Detailed API documentation
- `get_kit_extension_apis(extension_ids)`: List all APIs for extensions
- `get_api_list()`: Complete list of available API references

**Data Sources**:
- `codeatlas/*.codeatlas.json`: Full code analysis with symbols
- `api_docs/*.api_docs.json`: Extracted API documentation

**Features**:
- Unified access to both Code Atlas and API docs
- Symbol resolution with format `extension_id@symbol`
- Class, method, and function documentation
- Parameter types, return values, and docstrings

### CodeSearchService

**Purpose**: Enables semantic search for code examples and test patterns.

**Key Methods**:
- `search_kit_code_examples(query, top_k)`: Find production code examples
- `search_kit_test_examples(query, top_k)`: Find test implementations

**Data Sources**:
- `code_examples_faiss/`: FAISS database of code examples
- `extracted_methods_regular/`: JSON files with method metadata

**Features**:
- Separate databases for production and test code
- Relevance scoring based on semantic similarity
- Tag-based categorization (async, ui, usd, etc.)
- Source code extraction with line numbers

### SettingsService

**Purpose**: Manages Kit configuration settings with semantic search.

**Key Methods**:
- `search_kit_settings(query, top_k, prefix_filter, type_filter)`: Search settings
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

### KitExtensionsAtlasService

**Purpose**: Low-level service for accessing Code Atlas data.

**Key Methods**:
- `load_codeatlas(extension_id)`: Load Code Atlas for extension
- `load_api_docs(extension_id)`: Load API documentation
- `get_api_symbols(extension_id)`: Extract all API symbols
- `list_ui_modules(extension_id)`, `list_ui_classes(extension_id)`: Code structure queries

**Features**:
- Lazy loading with caching
- Version-aware file loading
- Structured access to classes, methods, modules
- Symbol extraction for API discovery

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

## MCP Tools Reference

### 1. search_kit_extensions

**Description**: Search for Kit extensions using semantic search across 400+ available extensions.

**Input Parameters**:
```python
{
  "query": str,        # Search query (e.g., "window management tools")
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
- Discovering tools for UI development
- Locating rendering or physics extensions
- Identifying viewport and camera extensions

**Example**:
```python
search_kit_extensions("ui widgets and controls", top_k=5)
```

---

### 2. get_kit_extension_details

**Description**: Get comprehensive information about specific Kit extensions.

**Input Parameters**:
```python
{
  "extension_ids": Optional[Union[str, List[str]]]
  # Single: "omni.ui"
  # Multiple: ["omni.ui", "omni.kit.window.console"]
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
get_kit_extension_details(["omni.ui", "omni.ui.scene"])
```

---

### 3. get_kit_extension_dependencies

**Description**: Analyze and visualize extension dependency graphs.

**Input Parameters**:
```python
{
  "extension_id": str,              # Extension to analyze
  "depth": int = 2,                # Dependency tree depth
  "include_optional": bool = False # Include optional deps
}
```

**Returns**: Dependency tree information:
- Extension name and version
- Required dependencies at each level
- Optional dependencies (if requested)
- Hierarchical structure
- Version requirements
- Circular dependency detection

**Use Cases**:
- Understanding extension requirements
- Identifying potential conflicts
- Planning extension installation
- Analyzing dependency impact

**Example**:
```python
get_kit_extension_dependencies("omni.ui", depth=3, include_optional=true)
```

---

### 4. get_kit_extension_apis

**Description**: List all APIs provided by specified Kit extensions.

**Input Parameters**:
```python
{
  "extension_ids": Optional[Union[str, List[str]]]
  # Accepts same flexible formats as get_kit_extension_details
}
```

**Returns**: Structured API listing with:
- Classes and their methods
- Module-level functions
- API reference format (`extension_id@symbol`)
- API count per extension
- Brief docstrings

**Use Cases**:
- Discovering available APIs in an extension
- Finding the right class or function
- Understanding extension capabilities
- Preparing for detailed API lookup

**Example**:
```python
get_kit_extension_apis(["omni.ui", "omni.ui.scene"])
```

---

### 5. get_kit_api_details

**Description**: Get complete API documentation for specific symbols.

**Input Parameters**:
```python
{
  "api_references": Optional[Union[str, List[str]]]
  # Format: "extension_id@symbol"
  # Single: "omni.ui@Window"
  # Multiple: ["omni.ui@Window", "omni.ui@Button"]
  # Empty/null: Lists available API references
}
```

**Returns**: Complete API documentation:
- Full docstrings and descriptions
- Method signatures with parameters
- Parameter types and defaults
- Return types and exceptions
- Property information
- Usage examples (if available)
- Class inheritance information

**API Reference Format**:
- Class: `omni.ui@Window`
- Method: `omni.ui@Window.set_visibility`
- Function: `omni.kit.commands@execute`

**Example**:
```python
get_kit_api_details(["omni.ui@Window", "omni.ui@Button"])
```

---

### 6. search_kit_code_examples

**Description**: Find relevant Kit code examples using semantic search.

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
- Associated tags (async, ui, usd, etc.)

**Search Capabilities**:
- Extension implementations and patterns
- UI component creation examples
- USD operation examples
- Widget usage and styling
- Layout and container implementations
- Event handling and callbacks

**Example**:
```python
search_kit_code_examples("create window with buttons", top_k=5)
```

---

### 7. search_kit_test_examples

**Description**: Find Kit test implementations and patterns.

**Input Parameters**:
```python
{
  "query": str,       # Test scenario to find examples for
  "top_k": int = 10  # Number of test examples
}
```

**Returns**: Test examples with:
- Complete test method code
- Setup and teardown patterns
- Test assertions and validation
- File paths and test locations
- Testing framework usage
- Test categories and tags

**Use Cases**:
- Learning test writing patterns
- Finding UI widget test examples
- Understanding USD stage testing
- Async test patterns
- Extension lifecycle testing

**Example**:
```python
search_kit_test_examples("ui widget testing", top_k=5)
```

---

### 8. search_kit_settings

**Description**: Search Kit configuration settings across 1,000+ settings from 400+ extensions.

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
search_kit_settings("viewport rendering", prefix_filter="rtx", top_k=10)
```

---

### 9. search_kit_app_templates

**Description**: Search for Kit application templates using semantic search.

**Input Parameters**:
```python
{
  "query": str,                     # Application use case description
  "top_k": int = 5,                # Number of results
  "category_filter": str = ""      # "editor", "authoring", "visualization", "streaming", "configuration"
}
```

**Returns**: Template matches with:
- Ranked list with relevance scores
- Brief descriptions and key features
- Template IDs for detailed retrieval
- Use case summaries
- Streaming support indicators
- Dependency counts

**Available Templates**:
- **kit_base_editor**: Minimal 3D editing application starter
- **usd_composer**: Professional USD content creation suite
- **usd_explorer**: Large-scale environment visualization
- **usd_viewer**: Streaming-optimized RTX viewer
- **streaming_configs**: Streaming configuration layers

**Example**:
```python
search_kit_app_templates("large scale visualization", category_filter="visualization")
```

---

### 10. get_kit_app_template_details

**Description**: Retrieve complete Kit application template examples with full details.

**Input Parameters**:
```python
{
  "template_ids": Optional[Union[str, List[str]]]
  # Single: "kit_base_editor"
  # Multiple: ["kit_base_editor", "usd_viewer"]
  # Empty/null: Lists all available templates
}
```

**Returns**: Complete template information:
- Full README documentation
- Complete .kit configuration files
- Detailed metadata and features
- Use cases and implementation guidance
- Dependency information
- Streaming configuration details

**Workflow**:
1. Search: `search_kit_app_templates("your use case")`
2. Review results and scores
3. Get details: `get_kit_app_template_details("template_id")`
4. Access complete README and configuration

**Example**:
```python
get_kit_app_template_details(["kit_base_editor", "usd_viewer"])
```

---

### 11. get_kit_instructions

**Description**: Retrieve Kit system instructions and development documentation.

**Input Parameters**:
```python
{
  "instruction_sets": Optional[Union[str, List[str]]]
  # Single: "kit_system"
  # Multiple: ["kit_system", "extensions", "testing"]
  # Empty/null: Lists all available instruction sets
}
```

**Available Instruction Sets**:

1. **kit_system**: Core Kit framework fundamentals
   - Extension system architecture
   - USD integration patterns
   - Application architecture
   - Carbonite services integration

2. **extensions**: Extension development guidelines
   - Configuration and dependencies
   - Lifecycle management
   - Service registration patterns
   - UI extension development

3. **testing**: Test writing best practices
   - Unit tests for Kit code
   - UI testing with omni.kit.test
   - USD stage testing patterns
   - Performance testing strategies

4. **usd**: USD integration patterns
   - Stage management and context handling
   - Prim creation and manipulation
   - Layer composition and editing
   - Transform and animation workflows

5. **ui**: UI development guidance
   - Window and layout creation
   - Widget usage and styling
   - Model-view data binding
   - 3D UI with omni.ui.scene

**Returns**: Formatted documentation with:
- Description and use cases
- Complete instruction content
- Markdown formatting
- Code examples and patterns

**Example**:
```python
get_kit_instructions(["kit_system", "extensions"])
```

---

## Data Sources

### FAISS Vector Databases

**Location**: `src/kit_fns/data/`

1. **extensions_faiss/**: Extension metadata embeddings
   - 400+ Kit extensions
   - Descriptions, features, categories
   - NVIDIA NV-EmbedQA-E5-v5 embeddings

2. **code_examples_faiss/**: Code example embeddings
   - Production code patterns
   - Method implementations
   - Tagged by functionality

3. **settings_faiss/**: Settings embeddings
   - 1,000+ configuration settings
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

3. **app_templates.json**: Application template metadata
   - Template descriptions
   - Use cases and features
   - Configuration details

### Per-Extension Files

1. **Code Atlas** (`codeatlas/*.codeatlas.json`):
   - Complete code structure analysis
   - Classes, methods, functions
   - Docstrings and signatures
   - Line numbers and file paths

2. **API Docs** (`api_docs/*.api_docs.json`):
   - Cleaned API documentation
   - Class and method signatures
   - Parameter and return types

### Application Templates

**Location**: `src/kit_fns/data/app_templates/`

Each template directory contains:
- `README.md`: Complete usage documentation
- `*.kit`: Kit configuration file
- Template-specific configuration files

### Instruction Sets

**Location**: `src/kit_fns/data/instructions/`

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
search_kit_extensions("viewport and camera management", top_k=5)

# 2. Get detailed information
get_kit_extension_details("omni.kit.viewport.window")

# 3. Check dependencies
get_kit_extension_dependencies("omni.kit.viewport.window", depth=2)

# 4. Explore available APIs
get_kit_extension_apis("omni.kit.viewport.window")
```

### Pattern 2: API Documentation Lookup

```python
# 1. Find extension with desired functionality
search_kit_extensions("ui widgets")

# 2. List available APIs
get_kit_extension_apis("omni.ui")

# 3. Get detailed API documentation
get_kit_api_details(["omni.ui@Window", "omni.ui@Button", "omni.ui@VStack"])
```

### Pattern 3: Learning by Example

```python
# 1. Search for code examples
search_kit_code_examples("create window with menu bar")

# 2. Find related test examples
search_kit_test_examples("window creation test")

# 3. Get implementation guidance
get_kit_instructions("ui")
```

### Pattern 4: Application Development Start

```python
# 1. Search for appropriate template
search_kit_app_templates("streaming visualization for large environments")

# 2. Get complete template details
get_kit_app_template_details("usd_explorer")

# 3. Understand required extensions
get_kit_extension_details(["omni.kit.streaming.core", "omni.kit.viewport.bundle"])

# 4. Load relevant instructions
get_kit_instructions(["kit_system", "extensions"])
```

### Pattern 5: Configuration Discovery

```python
# 1. Search for settings
search_kit_settings("ray tracing quality", prefix_filter="rtx")

# 2. Find extension-specific settings
search_kit_settings("viewport camera", prefix_filter="exts")

# 3. Filter by type
search_kit_settings("debug flags", type_filter="bool")
```

### Pattern 6: Comprehensive Learning Path

```python
# New to Kit development? Follow this sequence:

# 1. Start with fundamentals
get_kit_instructions("kit_system")

# 2. Learn extension development
get_kit_instructions("extensions")

# 3. Explore available extensions
search_kit_extensions("your domain of interest")

# 4. Study code examples
search_kit_code_examples("basic extension structure")

# 5. Understand testing
get_kit_instructions("testing")
search_kit_test_examples("extension lifecycle")

# 6. Choose application template
search_kit_app_templates("your application type")
get_kit_app_template_details("chosen_template_id")
```

---

## Performance Considerations

### Semantic Search Performance

- **FAISS Queries**: ~50-200ms depending on database size
- **Embedding Generation**: Handled by NVIDIA API
- **Fallback to Keyword**: Automatic if FAISS unavailable
- **Caching**: Extension and API data cached after first load

### Optimization Strategies

1. **Batch Requests**: Use array inputs for multiple items
   - Example: `get_kit_extension_details(["ext1", "ext2", "ext3"])`
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
   - Example: "Invalid API reference format: 'omni.ui'"
   - Suggestion: "Use format 'extension_id@symbol'"

2. **Not Found**:
   - Fuzzy matching suggestions when available
   - Example: "Did you mean 'omni.ui.scene'?"
   - Lists similar items for discovery

3. **Data Unavailable**:
   - Clear indication of missing data
   - Fallback to alternative sources when possible
   - Example: Using Code Atlas when API docs unavailable

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

## Future Enhancements

### Planned Features

1. **Streaming Responses**: For large documentation retrievals
2. **Diff and Comparison**: Compare extensions or API versions
3. **Interactive Examples**: Executable code samples
4. **Visual Dependency Graphs**: GraphQL-style visualization
5. **Custom Instruction Sets**: User-defined documentation
6. **Real-time Updates**: Live extension database synchronization

### Data Expansion

1. Additional code example sources
2. More granular test categorization
3. Enhanced API documentation with examples
4. Community-contributed templates
5. Performance benchmark data
6. Historical version information

---

## Conclusion

The Kit Functions MCP provides a comprehensive, AI-optimized interface to the entire NVIDIA Omniverse Kit ecosystem. By combining semantic search, structured data access, and intelligent caching, it enables AI models to efficiently discover and utilize Kit's extensive capabilities.

The system's flexible input handling, batch processing support, and comprehensive documentation make it an essential tool for AI-assisted Kit development, from initial learning to advanced application creation.
