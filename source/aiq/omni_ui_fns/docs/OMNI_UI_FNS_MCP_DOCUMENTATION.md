# Omni UI Functions MCP Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Functionalities](#key-functionalities)
4. [Services Layer](#services-layer)
5. [MCP Tools Reference](#mcp-tools-reference)
6. [Data Sources](#data-sources)
7. [Usage Patterns](#usage-patterns)
8. [Performance Considerations](#performance-considerations)
9. [Error Handling](#error-handling)

---

## Overview

The **Omni UI Functions MCP** (Model Context Protocol) is a comprehensive AI-powered toolkit for NVIDIA Omniverse UI development. It provides AI models with intelligent access to OmniUI's extensive API documentation, code examples, styling guides, and implementation patterns through semantic search and structured knowledge retrieval.

### What It Does

The Omni UI Functions MCP serves as an intelligent bridge between AI models and the OmniUI framework, enabling:

- **Semantic Code Search**: Find relevant UI implementation patterns and code examples using natural language queries
- **API Discovery**: Browse and explore 150+ OmniUI classes, 50+ modules, and 1,000+ methods with comprehensive documentation
- **Class Documentation**: Access detailed per-class usage guides organized by category with examples and styling information
- **Style Guidance**: Retrieve complete CSS-like styling documentation covering themes, colors, and widget-specific styles
- **Window Patterns**: Discover specialized window and dialog creation examples
- **System Instructions**: Load framework fundamentals, patterns, and best practices for OmniUI development

### Core Purpose

This system eliminates the need for AI models to navigate complex UI documentation or search through fragmented code examples manually. Instead, it provides curated, context-aware access to exactly the information needed for OmniUI development tasks, leveraging advanced semantic search with NVIDIA embeddings and reranking to deliver highly relevant results.

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
│  10 async functions implementing tool logic                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Services Layer                            │
│  • OmniUI Atlas Service: API database management            │
│  • Retrieval Service: FAISS semantic search                 │
│  • Reranking Service: NVIDIA relevance scoring              │
│  • UI Window Examples Service: Window pattern retrieval     │
│  • Telemetry Service: Usage tracking (Redis)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  • FAISS Vector Databases (semantic search)                 │
│  • UI Atlas JSON (complete API reference)                   │
│  • Instruction Files (system and class-level)               │
│  • Style Documentation (comprehensive styling guide)        │
│  • Class Instructions (per-class usage guides)              │
│  • RAG Collection (code example corpus)                     │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **AIQ Framework**: Tool registration and workflow management
- **LangChain**: Vector store and embedding integrations
- **FAISS**: High-performance semantic search with NVIDIA embeddings
- **Pydantic**: Input validation and schema definition
- **Redis**: Distributed telemetry and usage tracking
- **NVIDIA Embeddings**: nv-embedqa-e5-v5 for semantic understanding
- **NVIDIA Reranking**: llama-3.2-nv-rerankqa-1b-v2 for relevance scoring
- **Python 3.11+**: Core implementation language

---

## Key Functionalities

### 1. **Semantic Code Search**

The system provides two specialized semantic search functions optimized for different UI development scenarios:

- **Code Example Search**: Find OmniUI implementation patterns, widget usage, layout strategies, and event handling patterns from curated production code
- **Window Example Search**: Discover window and dialog creation patterns with complete implementations, specialized for window management use cases

Both leverage a sophisticated RAG (Retrieval-Augmented Generation) pipeline:
```
Query → Embeddings (NVIDIA) → FAISS Search (Top-90) → Reranking (NVIDIA) → Top-10 Results
```

### 2. **API Discovery and Documentation**

Comprehensive API exploration capabilities across the OmniUI framework:

- **Class Listing**: Enumerate all 150+ OmniUI classes across core widgets, containers, layouts, inputs, windows, and 3D scene components
- **Module Listing**: Browse 50+ OmniUI modules to understand framework organization
- **Class Details**: Retrieve comprehensive information including methods, parent classes, docstrings, and API structure
- **Module Details**: Explore module contents, contained classes, and functions
- **Method Details**: Access detailed method signatures with parameters, return types, and documentation

All detail functions support:
- **Fuzzy Matching**: Find entities even with typos or partial names
- **Batch Operations**: Retrieve multiple entities in a single call (70-80% faster)
- **Flexible Inputs**: Accept strings, arrays, JSON, comma-separated values, or null for listing

### 3. **Structured Learning Resources**

Built-in documentation and guidance systems:

- **System Instructions**: Four comprehensive guides covering framework fundamentals, patterns, and best practices
  - `agent_system`: Core system prompt with omni.ui basics
  - `classes`: Class API reference and model patterns
  - `omni_ui_scene_system`: 3D UI system documentation
  - `omni_ui_system`: Core widgets, containers, layouts

- **Class Instructions**: Per-class usage guides organized into 10 categories (61+ classes)
  - Models, Shapes, Widgets, Containers, Layouts, Inputs, Windows, Scene, Units, System
  - Each includes usage overview, properties, styling, examples, best practices

- **Style Documentation**: Complete styling reference with 12 sections
  - Overview, Styling syntax, Units, Fonts, Color palettes (shades)
  - Widget-specific styling (buttons, sliders, containers, windows)
  - 37,820+ tokens of comprehensive styling guidance

### 4. **Flexible Input Handling**

All tools support multiple input formats for maximum AI model compatibility:

- Single strings: `"Button"`
- Native arrays: `["Button", "Label", "TreeView"]`
- JSON strings: `'["Button", "Label"]'`
- Comma-separated: `"Button, Label, TreeView"`
- Empty/null: Returns listings or combined documentation

This flexibility ensures seamless integration across:
- Direct Python calls
- Command-line interfaces
- AI model tool use
- Web APIs and RESTful services

### 5. **Usage Telemetry**

Comprehensive telemetry captures:
- Function call patterns and frequency
- Performance metrics (execution time)
- Success/failure rates
- Parameter usage statistics
- Error tracking for system improvement
- Session-based analytics

---

## Services Layer

### OmniUI Atlas Service

**Purpose**: Manages the comprehensive OmniUI API database providing structured access to classes, modules, and methods.

**Key Methods**:
- `get_class_names()`: List all available class names
- `get_module_names()`: List all available module names
- `get_class(class_name, fuzzy)`: Retrieve class details with fuzzy matching
- `get_module(module_name, fuzzy)`: Retrieve module details with fuzzy matching
- `get_method(method_name, fuzzy)`: Retrieve method details with fuzzy matching
- `fuzzy_match(query, candidates, threshold)`: Find best matches for typos

**Data Sources**:
- `ui_atlas.json`: Complete OmniUI API reference extracted from runtime inspection
- Contains 150+ classes, 50+ modules, 1,000+ methods with full documentation

**Features**:
- Fuzzy matching with configurable threshold (default: 60/100)
- Hierarchical data structure (Module → Classes → Methods)
- Complete type information (parameters, return types)
- Comprehensive docstrings
- Parent class relationships
- Method categorization

### Retrieval Service

**Purpose**: Provides semantic search capabilities using FAISS vector databases and NVIDIA embeddings.

**Key Components**:
- **Embeddings**: Converts queries to vectors using NVIDIA's nv-embedqa-e5-v5 model
- **FAISS Search**: Performs similarity search across indexed code examples
- **Context Formatting**: Structures results for RAG consumption

**Configuration Options**:
```python
{
    "model": "nvidia/nv-embedqa-e5-v5",
    "endpoint": None,  # Uses NVIDIA API by default
    "api_key": "${NVIDIA_API_KEY}"
}
```

**Features**:
- Configurable top-k retrieval (default: 90 before reranking)
- Maximum context size: 30,000 characters
- Support for custom FAISS indices
- Metadata extraction from code examples

### Reranking Service

**Purpose**: Improves search result relevance through NVIDIA's advanced reranking model.

**How It Works**:
1. Receives top-k candidates from FAISS search (typically 90)
2. Uses NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model to score relevance
3. Returns top-n most relevant results (typically 10)

**Configuration Options**:
```python
{
    "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    "endpoint": None,  # Uses NVIDIA API by default
    "api_key": "${NVIDIA_API_KEY}"
}
```

**Benefits**:
- **Higher Precision**: Focuses on most relevant results
- **Context Understanding**: Considers query-passage relationships beyond embeddings
- **Cost Efficiency**: Reduces context window usage by filtering results
- **Graceful Fallback**: Returns original order if reranking fails

### UI Window Examples Service

**Purpose**: Specialized retrieval service for window and dialog creation patterns.

**Key Methods**:
- `retrieve(query, top_k, format_type)`: Search window examples with multiple output formats
- Supports three format types: `structured`, `formatted`, `raw`

**Data Sources**:
- `ui_window_examples_faiss/`: FAISS index of window implementations
- Schema includes descriptions, complete code, file paths, class/function names

**Features**:
- Structured output with rank, description, code, metadata
- Formatted markdown output for human readability
- Raw data access for programmatic processing
- Configurable top-k results

### Telemetry Service

**Purpose**: Centralized usage tracking using Redis Streams for analytics and monitoring.

**Key Methods**:
- `capture_call(function_name, request_data, duration_ms, success, error)`: Log function calls
- `track_call(function_name, request_data)`: Context manager for automatic timing

**Redis Schema**:
```
Key: omni_ui_mcp:telemetry:YYYY-MM-DD:HH-MM-SS-microseconds:call_id

Value: {
  "service": "omni_ui_mcp",
  "function_name": "search_ui_code_examples",
  "call_id": "uuid",
  "timestamp": "ISO-8601",
  "duration_ms": 123.45,
  "success": true,
  "request_data": {...},
  "session_id": "optional"
}
```

**Features**:
- Async non-blocking capture
- Automatic timestamp-based sorting
- Configurable expiration (default: 7 days)
- Session grouping capabilities
- Performance metrics collection
- Error tracking and categorization

---

## MCP Tools Reference

### 1. search_ui_code_examples

**Description**: Retrieve relevant OmniUI code examples using semantic vector search and optional reranking.

**Input Parameters**:
```python
{
  "request": str,                      # Natural language query
  "rerank_k": int = 10,               # Results after reranking
  "enable_rerank": bool = True,       # Use reranking
  "embedding_config": dict = None,    # Embedding service config
  "reranking_config": dict = None     # Reranking service config
}
```

**Returns**: Formatted text with:
- Multiple code examples (typically 10)
- File paths and line numbers
- Method/class context
- Complete source code
- Relevance ranking

**Use Cases**:
- Finding widget implementation patterns
- Learning event handling mechanisms
- Discovering layout strategies
- Understanding styling approaches
- Exploring callback patterns

**Example**:
```python
search_ui_code_examples("How to create a search field?")
search_ui_code_examples("Button styling with themes", rerank_k=5)
search_ui_code_examples("VStack and HStack layout examples")
```

**Query Tips**:
- Use specific OmniUI terminology (SearchField, ZStack, VStack)
- Include UI operations (build_ui, style, event handling)
- Reference widget types (Button, Label, Rectangle)
- Ask about patterns (callback, subscription, model binding)

---

### 2. search_ui_window_examples

**Description**: Retrieve UI window and dialog examples from a specialized database focused on window creation patterns.

**Input Parameters**:
```python
{
  "request": str,                      # Query for window example
  "top_k": int = 5,                   # Number of examples
  "format_type": str = "formatted",   # structured, formatted, raw
  "embedding_config": dict = None,    # Embedding service config
  "faiss_index_path": str = None      # Custom FAISS path
}
```

**Returns**: Window examples with:
- Complete window implementations
- Descriptions and use cases
- File paths and function names
- Class names and line numbers
- Ranked by relevance

**Format Types**:
- `structured`: JSON with metadata (rank, description, code, file_path, etc.)
- `formatted`: Markdown-formatted for readability
- `raw`: Raw retrieval data

**Use Cases**:
- Learning window creation patterns
- Understanding dialog box implementations
- Discovering modal vs. non-modal patterns
- Exploring window configuration options
- Finding container layout examples

**Example**:
```python
search_ui_window_examples("Create a modal dialog with buttons")
search_ui_window_examples("Window with sliders", top_k=3, format_type="structured")
search_ui_window_examples("Error message dialog box")
```

---

### 3. list_ui_classes

**Description**: Returns a complete list of all OmniUI class names from the Atlas database.

**Input Parameters**: None (zero-argument function)

**Returns**: JSON with:
- Complete list of class full names
- Total class count
- Alphabetically sorted
- Description of data source

**Example Response**:
```json
{
  "class_full_names": [
    "omni.ui.Button",
    "omni.ui.Label",
    "omni.ui.TreeView",
    "omni.ui.Window",
    "omni.ui.scene.Line",
    "..."
  ],
  "total_count": 150,
  "description": "OmniUI classes from Atlas data"
}
```

**Use Cases**:
- Discovering available OmniUI widgets
- Exploring the class hierarchy
- Finding specific component types
- Understanding framework coverage

**Example**:
```python
list_ui_classes()
```

---

### 4. list_ui_modules

**Description**: Returns a complete list of all OmniUI module names from the Atlas database.

**Input Parameters**: None (zero-argument function)

**Returns**: JSON with:
- Complete list of module names
- Total module count
- Alphabetically sorted
- Description of data source

**Example Response**:
```json
{
  "module_names": [
    "omni.ui",
    "omni.ui.scene",
    "omni.ui.workspace",
    "..."
  ],
  "total_count": 50,
  "description": "OmniUI modules from Atlas data"
}
```

**Use Cases**:
- Understanding module organization
- Navigating the framework structure
- Finding specialized modules
- Module import discovery

**Example**:
```python
list_ui_modules()
```

---

### 5. get_ui_class_detail

**Description**: Retrieves detailed information about one or more OmniUI classes including methods, parent classes, and docstrings.

**Input Parameters**:
```python
{
  "class_names": Optional[Union[str, List[str]]]
  # Single: "TreeView"
  # Multiple: ["Button", "Label", "TreeView"]
  # JSON: "[\"Button\", \"Label\"]"
  # Comma: "Button, Label, TreeView"
  # Empty/null: Lists all available classes
}
```

**Returns**: Detailed JSON with:
- Class name and full name
- Module name
- Complete docstring
- Parent classes
- All methods (names only)
- Detailed method information (signatures, parameters, return types)
- Fuzzy match score if search was fuzzy
- Total method count

**Single Class Response Structure**:
```json
{
  "name": "TreeView",
  "full_name": "omni.ui.TreeView",
  "module_name": "omni.ui",
  "docstring": "A tree view widget...",
  "parent_classes": ["Widget"],
  "methods": ["__init__", "set_root_visible", ...],
  "method_details": [
    {
      "name": "__init__",
      "parameters": [...],
      "return_type": "None",
      "docstring": "...",
      "is_static": false
    }
  ],
  "total_methods": 25,
  "match_score": 1.0
}
```

**Batch Processing Benefits**:
- 80% faster for multiple classes
- Single API call vs multiple round-trips
- Efficient context window usage

**Example**:
```python
get_ui_class_detail(["TreeView"])
get_ui_class_detail(["Button", "Label", "TreeView"])
get_ui_class_detail(None)  # List all classes
```

---

### 6. get_ui_module_detail

**Description**: Retrieves detailed information about one or more OmniUI modules including contained classes and functions.

**Input Parameters**:
```python
{
  "module_names": Optional[Union[str, List[str]]]
  # Accepts same flexible formats as get_ui_class_detail
}
```

**Returns**: Detailed JSON with:
- Module name and full name
- File path
- Extension name
- Class names (list)
- Function names (list)
- Detailed class information
- Detailed function information
- Total counts
- Fuzzy match score if applicable

**Module Response Structure**:
```json
{
  "name": "omni.ui",
  "full_name": "omni.ui",
  "file_path": "/path/to/omni/ui/__init__.py",
  "extension_name": "omni.ui",
  "class_names": ["Button", "Label", ...],
  "function_names": ["create_window", ...],
  "classes": [...],
  "functions": [...],
  "match_score": 1.0,
  "total_classes": 50,
  "total_functions": 10
}
```

**Batch Processing Benefits**:
- 70% faster for multiple modules
- Comprehensive module overview
- Relationship mapping

**Example**:
```python
get_ui_module_detail(["omni.ui"])
get_ui_module_detail(["omni.ui", "omni.ui.scene"])
```

---

### 7. get_ui_method_detail

**Description**: Retrieves detailed information about one or more OmniUI methods including signatures, parameters, and return types.

**Input Parameters**:
```python
{
  "method_names": Optional[Union[str, List[str]]]
  # Accepts same flexible formats as other detail functions
}
```

**Returns**: Detailed JSON with:
- Method name and full name
- Class name
- Parameters with types
- Return type
- Complete docstring
- Static/classmethod/property flags
- Fuzzy match score if applicable

**Method Response Structure**:
```json
{
  "name": "__init__",
  "full_name": "omni.ui.Button.__init__",
  "class_name": "omni.ui.Button",
  "parameters": [
    {"name": "self", "type": ""},
    {"name": "text", "type": "str"},
    {"name": "**kwargs", "type": ""}
  ],
  "return_type": "None",
  "docstring": "Button constructor...",
  "is_static": false,
  "is_classmethod": false,
  "is_property": false,
  "match_score": 1.0
}
```

**Batch Processing Benefits**:
- 60-80% faster for multiple methods
- Complete API reference in one call
- Pattern discovery across methods

**Example**:
```python
get_ui_method_detail(["__init__"])
get_ui_method_detail(["__init__", "clicked_fn", "set_value"])
```

---

### 8. get_ui_instructions

**Description**: Retrieves OmniUI system-level instructions and documentation for different aspects of the framework.

**Input Parameters**:
```python
{
  "name": Optional[str]
  # Instruction set name or null to list all
}
```

**Available Instruction Sets**:

| Name | Description | Topics Covered |
|------|-------------|----------------|
| `agent_system` | Core system prompt | Framework fundamentals, widget filters, options menus, general guidelines |
| `classes` | Class API reference | AbstractValueModel, data models, custom implementations, callbacks |
| `omni_ui_scene_system` | 3D UI documentation | 3D shapes, SceneView, camera controls, transforms, manipulators |
| `omni_ui_system` | Core widgets/layouts | Basic widgets, HStack/VStack/ZStack, styling, drag & drop, MDV pattern |

**Returns**: Formatted documentation with:
- Complete instruction content (markdown)
- Description and use cases
- Metadata (name, filename, size, line count)

**Use Cases**:
- Loading framework fundamentals for AI agents
- Understanding data model patterns
- Learning 3D UI development
- Exploring widget and layout systems
- Getting best practices and guidelines

**Example**:
```python
get_ui_instructions("agent_system")
get_ui_instructions("omni_ui_scene_system")
get_ui_instructions(None)  # List all available
```

---

### 9. get_ui_class_instructions

**Description**: Retrieves class-specific usage instructions organized by category, with comprehensive examples and styling information.

**Input Parameters**:
```python
{
  "class_names": Optional[Union[str, List[str]]]
  # Class names, category commands, or null
}
```

**Available Categories** (61+ classes):

| Category | Count | Examples |
|----------|-------|----------|
| `models` | 3 | AbstractValueModel, AbstractItemModel, AbstractItemDelegate |
| `shapes` | 12 | Rectangle, Circle, Triangle, Line, + Free variants |
| `widgets` | 11 | Button, Label, TreeView, CheckBox, ComboBox |
| `containers` | 7 | Frame, ScrollingFrame, HStack, VStack, ZStack |
| `layouts` | 3 | VGrid, HGrid, Placer |
| `inputs` | 7 | FloatSlider, IntSlider, FloatDrag, IntDrag |
| `windows` | 6 | Window, MainWindow, Menu, MenuBar, Tooltip |
| `scene` | 9 | All omni.ui.scene 3D UI components |
| `units` | 3 | Pixel, Percent, Fraction |
| `system` | 1 | Style |

**Special Commands** (string input only):
- `"categories"`: List all categories
- `"category:widgets"`: List classes in widgets category
- `"category:scene"`: List 3D scene classes

**Scene Class Format**:
```python
# Any of these work:
get_ui_class_instructions(["scene.Line"])
get_ui_class_instructions(["omni.ui.scene.Line"])
get_ui_class_instructions(["Line"])  # If unique
```

**Returns**: Formatted documentation with:
- Usage overview and purpose
- Basic and advanced examples
- Properties and their types
- Styling options (CSS-like syntax)
- Best practices
- Related classes
- Category information

**Batch Processing Benefits**:
- 75% faster for multiple classes
- Combined documentation in one response
- Category-based organization

**Example**:
```python
get_ui_class_instructions(["Button"])
get_ui_class_instructions(["Button", "Label", "TreeView"])
get_ui_class_instructions(["scene.Line", "scene.Rectangle"])
get_ui_class_instructions(["categories"])  # List categories
get_ui_class_instructions(["category:widgets"])  # List widget classes
get_ui_class_instructions(None)  # Show all categories
```

---

### 10. get_ui_style_docs

**Description**: Retrieves comprehensive OmniUI styling documentation covering CSS-like styling syntax, themes, colors, and widget-specific styles.

**Input Parameters**:
```python
{
  "sections": Optional[Union[str, List[str]]]
  # Section names, or null for complete documentation
}
```

**Available Sections**:

| Section | Description |
|---------|-------------|
| `overview` | High-level introduction to styling system |
| `styling` | Core styling syntax and rules |
| `units` | Measurement system (px, %, em, rem) |
| `fonts` | Typography system and text styling |
| `shades` | Color palettes and theme management |
| `window` | Window-level styling and frame customization |
| `containers` | Layout components (Frame, Stack, Grid) |
| `widgets` | Individual UI components (Label, Input, etc.) |
| `buttons` | Button variations and states |
| `sliders` | Slider and range components |
| `shapes` | Basic geometric elements |
| `line` | Line and curve elements |

**Returns**: Styling documentation with:
- CSS-like syntax overview
- Property reference
- Selectors and states
- Color palettes and themes
- Common styling patterns
- Best practices

**Complete Documentation**:
- Size: 37,820+ tokens
- All 12 sections combined
- Comprehensive styling reference
- Best for AI context loading

**Example Styling Syntax**:
```python
# Basic styling
Button.Label:label {
    color: 0xFF00FF00
    font_size: 14
}

# State-based styling
Button:hovered {
    background_color: 0xFF303030
}

# Complex selectors
Window > Frame > Button {
    border_radius: 4
    padding: 5
}
```

**Example**:
```python
get_ui_style_docs(["buttons"])
get_ui_style_docs(["buttons", "widgets", "containers"])
get_ui_style_docs(None)  # Complete documentation
```

**Use Cases**:
- Learning OmniUI styling syntax
- Understanding theme system
- Finding widget-specific styles
- Discovering color palettes
- Implementing custom themes
- Styling states (hover, pressed, disabled)

---

## Data Sources

### FAISS Vector Databases

**Location**: `src/omni_ui_fns/data/`

1. **faiss_index_omni_ui/**: Code examples embeddings
   - Curated OmniUI implementations from extensions
   - Production code patterns and widget usage
   - NVIDIA nv-embedqa-e5-v5 embeddings
   - Schema: file paths, method names, source code

2. **ui_window_examples_faiss/**: Window examples embeddings
   - Window and dialog implementations
   - Complete code with descriptions
   - NVIDIA nv-embedqa-e5-v5 embeddings
   - Schema: descriptions, code, file paths, class/function names

### UI Atlas Database

**Location**: `src/omni_ui_fns/data/ui_atlas.json`

A comprehensive JSON database containing:
- **150+ classes** with complete API documentation
- **50+ modules** with contents and relationships
- **1,000+ methods** with signatures and docstrings
- **Type information**: Parameters, return types
- **Documentation**: Complete docstrings for all entities
- **Relationships**: Module → Classes → Methods hierarchy

**Generation**: Extracted from OmniUI runtime inspection

**Structure**:
```json
{
  "modules": {
    "omni.ui": {
      "name": "omni.ui",
      "class_names": [...],
      "function_names": [...]
    }
  },
  "classes": {
    "omni.ui.Button": {
      "name": "Button",
      "methods": [...],
      "docstring": "..."
    }
  },
  "methods": {
    "omni.ui.Button.__init__": {
      "name": "__init__",
      "parameters": [...],
      "return_type": "...",
      "docstring": "..."
    }
  }
}
```

### RAG Collection

**Location**: `src/omni_ui_fns/data/omni_ui_rag_collection.json`

Code examples corpus for semantic search:
- Curated code snippets
- Method implementations
- Widget usage patterns
- Complete with metadata

### Instruction Files

**Location**: `src/omni_ui_fns/data/instructions/`

**System-Level Instructions**:
- `agent_system.md`: Core system prompt
- `classes.md`: Class API reference
- `omni_ui_scene_system.md`: 3D UI system documentation
- `omni_ui_system.md`: Core widgets and layouts

**Class Instructions** (`instructions/classes/`):
Organized by category:
- `models/`: Data models (3 files)
- `shapes/`: Basic shapes (12 files)
- `widgets/`: UI widgets (11 files)
- `containers/`: Layout containers (7 files)
- `layouts/`: Layout managers (3 files)
- `inputs/`: Input controls (7 files)
- `windows/`: Windows and dialogs (6 files)
- `scene/`: 3D UI components (9 files)
- `units/`: Measurement types (3 files)
- `system/`: System components (1 file)

Each file contains:
- Usage overview
- Properties and types
- Styling options
- Code examples
- Best practices

### Style Documentation

**Location**: `src/omni_ui_fns/data/styles/`

**Complete Guide**: `all_styling_combined.md` (37,820+ tokens)

**Individual Sections**:
- `overview.md`, `styling.md`, `units.md`, `fonts.md`
- `shades.md`, `window.md`, `containers.md`, `widgets.md`
- `buttons.md`, `sliders.md`, `shapes.md`, `line.md`

Content includes:
- CSS-like syntax guide
- Color palettes and themes
- Widget-specific styling
- Layout and spacing
- State selectors (hover, pressed, disabled)

---

## Usage Patterns

### Pattern 1: AI-Powered Code Generation

```python
# 1. Load system context
system = await get_ui_instructions("omni_ui_system")

# 2. Get class-specific instructions
button_guide = await get_ui_class_instructions(["Button"])

# 3. Find implementation examples
examples = await search_ui_code_examples("button with custom style")

# 4. Get styling reference
styles = await get_ui_style_docs(["buttons"])

# → AI now has complete context to generate OmniUI code
```

### Pattern 2: Framework Exploration

```python
# 1. List all classes
all_classes = await list_ui_classes()

# 2. Get details on interesting classes
details = await get_ui_class_detail(["Button", "Label", "Window"])

# 3. Understand module organization
module_info = await get_ui_module_detail(["omni.ui"])

# 4. Learn usage patterns
class_guides = await get_ui_class_instructions(["Button", "Label"])
```

### Pattern 3: Window Development Workflow

```python
# 1. Find similar window implementations
window_examples = await search_ui_window_examples("settings dialog")

# 2. Get widget documentation
widget_detail = await get_ui_class_detail(["Button", "Slider"])

# 3. Check styling options
widget_styles = await get_ui_style_docs(["buttons", "sliders"])

# 4. Find callback patterns
callback_examples = await search_ui_code_examples("button click callback")
```

### Pattern 4: Learning by Category

```python
# 1. Explore available categories
categories = await get_ui_class_instructions(None)

# 2. List classes in category
widgets = await get_ui_class_instructions(["category:widgets"])

# 3. Get detailed guides for category
widget_guides = await get_ui_class_instructions([
    "Button", "Label", "TreeView", "CheckBox"
])

# 4. Find examples
examples = await search_ui_code_examples("interactive widget examples")
```

### Pattern 5: Styling and Theming

```python
# 1. Get complete styling documentation
all_styles = await get_ui_style_docs(None)

# 2. Focus on specific widget styling
button_styles = await get_ui_style_docs(["buttons", "widgets"])

# 3. Find styled examples
styled_examples = await search_ui_code_examples("custom styled button")

# 4. Get class styling properties
button_props = await get_ui_class_instructions(["Button"])
```

### Pattern 6: 3D UI Development

```python
# 1. Load 3D UI system documentation
scene_system = await get_ui_instructions("omni_ui_scene_system")

# 2. List scene classes
scene_classes = await get_ui_class_instructions(["category:scene"])

# 3. Get specific scene class details
scene_details = await get_ui_class_instructions([
    "scene.Line", "scene.Rectangle", "scene.Transform"
])

# 4. Find 3D UI examples
scene_examples = await search_ui_code_examples("3D scene shapes")
```

### Pattern 7: Complete Learning Path

```python
# New to OmniUI? Follow this sequence:

# 1. Start with fundamentals
fundamentals = await get_ui_instructions("agent_system")

# 2. Learn core UI system
core_ui = await get_ui_instructions("omni_ui_system")

# 3. Explore available widgets
classes = await list_ui_classes()

# 4. Get widget documentation
widget_details = await get_ui_class_detail([
    "Button", "Label", "VStack", "HStack"
])

# 5. Learn styling
styling = await get_ui_style_docs(["overview", "styling", "shades"])

# 6. Study examples
examples = await search_ui_code_examples("basic UI layout with buttons")

# 7. Get class-specific guides
guides = await get_ui_class_instructions([
    "Button", "Label", "VStack", "HStack"
])
```

---

## Performance Considerations

### Semantic Search Performance

- **FAISS Queries**: ~200-500ms depending on database size
- **Embedding Generation**: Handled by NVIDIA API (~1-2s)
- **Reranking**: NVIDIA API (~500-1000ms for 90 results → 10)
- **Total Latency**: ~2-3s for complete semantic search with reranking

### Atlas Query Performance

- **Class Lookup**: ~50ms (local JSON query)
- **Module Lookup**: ~50ms (local JSON query)
- **Method Lookup**: ~50ms (local JSON query)
- **Fuzzy Matching**: ~100ms (includes similarity computation)
- **Batch Operations**: ~200ms for multiple entities (70-80% faster than individual)

### Optimization Strategies

1. **Use Batch Operations**: Retrieve multiple entities in one call
   - Example: `get_ui_class_detail(["Button", "Label", "TreeView"])`
   - 80% faster than individual calls
   - Reduces API overhead
   - More efficient context window usage

2. **Disable Reranking for Speed**: Skip reranking when speed is critical
   - Example: `search_ui_code_examples("query", enable_rerank=False)`
   - Reduces latency by ~500-1000ms
   - Acceptable for broad queries
   - Use reranking for precision-critical searches

3. **Cache Results**: Store frequently accessed data
   - Class and module listings rarely change
   - System instructions are static
   - Style documentation is static
   - Example: Cache `list_ui_classes()`, `list_ui_modules()` results

4. **Use Specific Queries**: Better queries → better results
   - ❌ Vague: "how to make UI"
   - ✅ Specific: "create Button with custom style and click handler"

5. **Choose Appropriate top_k/rerank_k**: Balance between completeness and speed
   - Default values optimized for most use cases
   - Reduce for faster responses with fewer results
   - Increase only when more results needed

### Telemetry Impact

- Async telemetry capture: ~1-5ms overhead
- Redis connection pooling for efficiency
- Graceful degradation if Redis unavailable
- No impact on tool functionality

### Memory Considerations

- FAISS indices loaded on demand
- Atlas JSON loaded once and cached
- Instruction files loaded on access
- Peak memory: ~500MB for all data loaded

---

## Error Handling

### Common Error Scenarios

1. **Invalid Input Format**:
   - Returns clear error message with expected format
   - Example: "Invalid class name format"
   - Suggestion: "Use simple class name or full name like 'omni.ui.Button'"

2. **Not Found with Fuzzy Matching**:
   - Fuzzy matching suggestions when available
   - Example: "Class 'Buton' not found. Did you mean 'Button'?"
   - Lists similar items for discovery
   - Configurable fuzzy match threshold

3. **Data Unavailable**:
   - Clear indication of missing data
   - Example: "FAISS index not found at path: ..."
   - Guidance on data setup or installation
   - Graceful degradation when possible

4. **Empty Results**:
   - Helpful message about search refinement
   - Example: "No relevant examples found for query '...'"
   - Suggestions for alternative searches
   - Links to related tools or functions

5. **API Key Missing**:
   - Clear error about missing NVIDIA_API_KEY
   - Instructions for setting environment variable
   - Example: "No API key available. Set NVIDIA_API_KEY environment variable"

6. **Reranking Failures**:
   - Graceful fallback to original FAISS order
   - Log warning but continue execution
   - Results still useful without reranking

### Error Response Format

All tools return consistent error structure:
```json
{
  "success": false,
  "result": null,
  "error": "Detailed error message with helpful context"
}
```

### Validation Errors

Input validation errors for:
- Empty queries (semantic search)
- Invalid JSON in string format
- Type mismatches
- Out-of-range parameters

All include:
- Clear error description
- Expected format
- Example of valid input

---

## Conclusion

The Omni UI Functions MCP provides a comprehensive, AI-optimized interface to the entire NVIDIA Omniverse UI ecosystem. By combining semantic search with NVIDIA embeddings and reranking, structured API documentation access, and curated learning resources, it enables AI models to efficiently discover, learn, and implement OmniUI components.

The system's flexible input handling, batch processing support, fuzzy matching, and comprehensive documentation make it an essential tool for AI-assisted OmniUI development. From initial learning and framework exploration to advanced window creation and custom theming, the Omni UI Functions MCP accelerates UI development with intelligent, context-aware assistance.

Key differentiators:
- **Advanced RAG Pipeline**: NVIDIA embeddings + FAISS + reranking for highest relevance
- **Complete API Coverage**: 150+ classes, 50+ modules, 1,000+ methods fully documented
- **Organized Learning**: 61+ class guides in 10 categories, 4 system instruction sets
- **Comprehensive Styling**: 37,820+ tokens covering all aspects of OmniUI theming
- **Production-Ready**: Telemetry, error handling, validation, graceful degradation
- **AI-First Design**: Flexible inputs, clear outputs, optimized for model consumption

This system transforms how AI models interact with UI development frameworks, providing the intelligence layer needed for effective AI-assisted development.
