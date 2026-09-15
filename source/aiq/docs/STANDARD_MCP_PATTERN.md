# Standard MCP Pattern for Omniverse

## Overview

This document defines the common architectural pattern used across Omniverse MCP (Model Context Protocol) systems. Both **Kit Functions MCP** and **Omni UI Functions MCP** implement this standardized pattern, providing consistency in design, implementation, and usage.

### Pattern Purpose

The Standard MCP Pattern enables AI models to access complex technical knowledge through:
- **Semantic Search**: Natural language queries against vector-indexed knowledge
- **Structured Atlas**: Comprehensive API/entity reference databases
- **Curated Instructions**: Learning materials and development guidance
- **Flexible Code Discovery**: Real implementation examples from production code

---

## Four-Layer Architecture

All Omniverse MCP systems follow a consistent four-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                   AI Model (via MCP Protocol)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         [1] MCP Tool Registration Layer (AIQ)                │
│  • Pydantic schemas for input validation                    │
│  • Flexible input format handling                           │
│  • Usage logging integration                                │
│  • Tool metadata and descriptions                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              [2] Functions Layer                             │
│  • Async Python functions                                   │
│  • Business logic implementation                            │
│  • Standardized return format                               │
│  • Telemetry capture                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              [3] Services Layer                              │
│  • Atlas Service (structured knowledge)                     │
│  • Retrieval Service (semantic search)                      │
│  • Reranking Service (relevance scoring)                    │
│  • Telemetry Service (usage tracking)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              [4] Data Layer                                  │
│  • FAISS vector databases                                   │
│  • JSON Atlas databases                                     │
│  • Instruction markdown files                               │
│  • Code example collections                                 │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Layer 1 - Registration**: AIQ integration, input validation, tool exposure
**Layer 2 - Functions**: Core business logic, orchestration, response formatting
**Layer 3 - Services**: Reusable components shared across functions
**Layer 4 - Data**: Static knowledge bases and vector indices

---

## Standard Services Layer

### 1. Atlas Service Pattern

**Purpose**: Structured API/entity reference database with fuzzy matching

**Common Capabilities**:
- List all entities (classes, modules, extensions)
- Get detailed entity information
- Fuzzy matching for typo tolerance
- Hierarchical relationships (parent-child)
- Batch retrieval operations

**Data Structure**:
```json
{
  "entities": {
    "entity_id": {
      "name": "...",
      "full_name": "...",
      "docstring": "...",
      "relationships": [...],
      "metadata": {...}
    }
  }
}
```

**Implementations**:
- **Kit Functions**: `KitExtensionsAtlasService` - Per-extension Code Atlas files
- **Omni UI Functions**: `OmniUI Atlas Service` - Single consolidated UI Atlas JSON

### 2. Retrieval Service Pattern

**Purpose**: Semantic search using FAISS vector databases

**Common Capabilities**:
- Embed queries using NVIDIA embeddings (nemotron-3-embed-1b)
- FAISS similarity search
- Configurable top-k retrieval
- Result formatting for RAG

**Standard Workflow**:
```
Query → Embeddings (NVIDIA) → FAISS Search (Top-K) → Context Formatting
```

**Implementations**:
- **Kit Functions**: Extension search, code search, settings search
- **Omni UI Functions**: Code examples search, window examples search

### 3. Reranking Service Pattern (Optional)

**Purpose**: Improve search relevance through neural reranking

**Common Capabilities**:
- Score query-document pairs
- NVIDIA reranking model (llama-nemotron-rerank-vl-1b-v2)
- Top-N selection from candidates
- Graceful fallback on failure

**Standard Workflow**:
```
FAISS Results (90) → Reranking (NVIDIA) → Top Results (10)
```

**Implementations**:
- **Kit Functions**: Not explicitly used
- **Omni UI Functions**: Optional reranking for code/window examples

### 4. Telemetry Service Pattern

**Purpose**: Distributed usage tracking and analytics

**Common Capabilities**:
- Async non-blocking call capture
- Redis Streams for storage
- Performance metrics (duration, success rate)
- Session-based grouping
- Graceful degradation if unavailable

**Redis Schema**:
```
Key: {service}:telemetry:YYYY-MM-DD:HH-MM-SS-microseconds:call_id

Value: {
  "service": "...",
  "function_name": "...",
  "call_id": "uuid",
  "timestamp": "ISO-8601",
  "duration_ms": float,
  "success": bool,
  "request_data": {...}
}
```

---

## Standard Data Layer

### 1. FAISS Vector Databases

**Purpose**: Enable semantic search through pre-computed embeddings

**Standard Structure**:
- FAISS index files (typically IVF or Flat)
- Metadata JSON with document IDs
- Embedded using NVIDIA nemotron-3-embed-1b

**Common Indices**:
- Entity search (extensions, classes, modules)
- Code examples
- Settings/configuration
- Specialized patterns (windows, tests)

**Naming Convention**: `{entity_type}_faiss/`

### 2. JSON Atlas Databases

**Purpose**: Structured reference for entities and their relationships

**Standard Structure**:
```json
{
  "metadata": {
    "version": "...",
    "generated": "...",
    "total_count": int
  },
  "entities": {
    "entity_id": {
      "name": "...",
      "type": "...",
      "description": "...",
      "relationships": [...],
      "properties": {...}
    }
  }
}
```

**Common Atlases**:
- **Kit Functions**: `extensions_database.json`, `settings_summary.json`, Code Atlas per extension
- **Omni UI Functions**: `ui_atlas.json`, `omni_ui_rag_collection.json`

### 3. Instruction Files

**Purpose**: Curated learning materials and development guidance

**Standard Structure**:
- Markdown format
- Organized by topic or category
- Include examples and best practices
- System-level and entity-level instructions

**Organization Patterns**:
```
instructions/
├── {system_topic}.md          # System-level (fundamentals)
└── {category}/                # Entity-level (per-class/extension)
    └── {entity_name}.md
```

**Common Topics**:
- System fundamentals
- Framework architecture
- Development patterns
- Testing strategies
- Entity-specific usage guides

### 4. Code Example Collections

**Purpose**: Real implementation examples from production code

**Standard Metadata**:
```json
{
  "file_path": "...",
  "code": "...",
  "description": "...",
  "entity_name": "...",
  "tags": [...],
  "line_number": int
}
```

---

## Standard Function Categories

### 1. Search Functions

**Pattern**: `search_{entity_type}(query, top_k, filters)`

**Purpose**: Semantic search across specific entity types

**Standard Parameters**:
- `query` (str): Natural language search query
- `top_k` (int): Number of results (default: 5-10)
- Filters (optional): Type, category, prefix filters

**Standard Return**:
```json
{
  "success": true,
  "result": "Formatted text with ranked results",
  "error": null
}
```

**Examples**:
- `search_kit_extensions` (Kit Functions)
- `search_kit_code_examples` (Kit Functions)
- `search_ui_code_examples` (Omni UI Functions - semantic search)

### 2. List Functions

**Pattern**: `get_{entity_type}s()`

**Purpose**: Enumerate all available entities

**Standard Parameters**: None (zero-argument)

**Standard Return**:
```json
{
  "success": true,
  "result": "{
    \"entity_names\": [...],
    \"total_count\": int,
    \"description\": \"...\"
  }",
  "error": null
}
```

**Examples**:
- `list_ui_classes` (Omni UI Functions)
- `list_ui_modules` (Omni UI Functions)

### 3. Detail Functions

**Pattern**: `get_{entity_type}_detail(entity_names)`

**Purpose**: Retrieve comprehensive information about entities

**Standard Parameters**:
- `entity_names` (Optional[Union[str, List[str]]]): Flexible input format
- Accepts: single string, array, JSON string, comma-separated, or null

**Standard Return**:
```json
{
  "success": true,
  "result": "{
    \"name\": \"...\",
    \"full_name\": \"...\",
    \"description\": \"...\",
    \"properties\": {...},
    \"relationships\": [...],
    \"metadata\": {...}
  }",
  "error": null
}
```

**Batch Support**: Multiple entities in single call (70-80% faster)

**Examples**:
- `get_kit_extension_details` (Kit Functions)
- `get_kit_api_details` (Kit Functions)
- `get_ui_class_detail` (Omni UI Functions)
- `get_ui_module_detail` (Omni UI Functions)
- `get_ui_method_detail` (Omni UI Functions)

### 4. Instruction Functions

**Pattern**: `get_instructions(instruction_name)` or `get_{entity_type}_instructions(entity_names)`

**Purpose**: Access learning materials and usage guides

**Standard Parameters**:
- `instruction_name` or `entity_names` (Optional[str]): Instruction set or entity name
- Null/empty returns listing

**Standard Return**:
```json
{
  "success": true,
  "result": "# Markdown Content\n\n...",
  "metadata": {
    "name": "...",
    "description": "...",
    "content_length": int,
    "line_count": int
  },
  "error": null
}
```

**Examples**:
- `get_{scope}_instructions` (Kit Functions, Omni UI Functions)
- `get_ui_class_instructions` (Omni UI Functions)
- `get_ui_style_docs` (Omni UI Functions - specialized)

---

## Standard RAG Workflow

The Standard MCP Pattern implements a consistent RAG (Retrieval-Augmented Generation) workflow for semantic search:

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Query Input                                         │
│  User provides natural language query                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Embedding Generation                                │
│  NVIDIA nemotron-3-embed-1b converts query to vector           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: FAISS Similarity Search                             │
│  Find top-K candidates (typically 90)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Reranking (Optional)                                │
│  NVIDIA llama-nemotron-rerank-vl-1b-v2 scores relevance        │
│  Select top-N (typically 10)                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Context Formatting                                  │
│  Format results with metadata for AI consumption            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Telemetry Capture                                   │
│  Log performance metrics and usage data                      │
└─────────────────────────────────────────────────────────────┘
```

### Configuration Parameters

**Embeddings**:
- Model: `nvidia/nemotron-3-embed-1b`
- Endpoint: NVIDIA API (default) or custom
- API Key: `${NVIDIA_API_KEY}` environment variable

**FAISS Search**:
- top_k: 90 (before reranking) or 10 (without reranking)
- Max context size: 30,000 characters

**Reranking** (Optional):
- Model: `nvidia/llama-nemotron-rerank-vl-1b-v2`
- Endpoint: NVIDIA API (default) or custom
- rerank_k: 10 (final results)
- Enable/disable flag: `enable_rerank` parameter

---

## Standard Input Handling

### Flexible Format Support

All detail and instruction functions accept multiple input formats:

```python
# Single string
get_entity_detail("EntityName")

# Array
get_entity_detail(["Entity1", "Entity2", "Entity3"])

# JSON string
get_entity_detail("[\"Entity1\", \"Entity2\"]")

# Comma-separated
get_entity_detail("Entity1, Entity2, Entity3")

# Null/empty (list all)
get_entity_detail(None)
get_entity_detail([])
```

### Validation Pattern

1. Check if input is None/empty → Return listing
2. Parse string to list if needed
3. Handle JSON string parsing
4. Handle comma-separated parsing
5. Normalize to list format
6. Validate entity names
7. Proceed with batch processing

---

## Standard Return Format

All functions return a consistent structure:

```python
{
  "success": bool,      # True if operation succeeded
  "result": Any,        # Result data (string, JSON, or structured)
  "error": str | None   # Error message if success=False
}
```

### Success Response
```python
{
  "success": True,
  "result": "...",
  "error": None
}
```

### Error Response
```python
{
  "success": False,
  "result": None,
  "error": "Detailed error message with context and suggestions"
}
```

---

## Common Technology Stack

### Required Components

1. **AIQ Framework**: Tool registration and workflow management
2. **LangChain**: Vector store and embedding integrations
3. **FAISS**: High-performance vector similarity search
4. **Pydantic**: Input validation and schema definition
5. **Redis**: Distributed telemetry and caching
6. **Python 3.11+**: Core implementation language

### NVIDIA Services

1. **Embeddings**: `nvidia/nemotron-3-embed-1b`
   - Purpose: Query and document embedding
   - Access: NVIDIA API with API key

2. **Reranking** (Optional): `nvidia/llama-nemotron-rerank-vl-1b-v2`
   - Purpose: Relevance scoring
   - Access: NVIDIA API with API key

---

## Key Design Principles

### 1. AI-First Design
- Flexible input formats for different AI model capabilities
- Clear, structured outputs optimized for model consumption
- Comprehensive tool descriptions with use cases
- Natural language query support

### 2. Batch Operations
- Support multiple entities in single call
- 70-80% faster than individual calls
- Efficient context window usage
- Reduced API overhead

### 3. Fuzzy Matching
- Typo tolerance with configurable threshold
- Suggest similar entities on "not found"
- Levenshtein distance or similar algorithms
- Default threshold: 60/100

### 4. Graceful Degradation
- Fallback behavior when services unavailable
- Continue operation if telemetry fails
- Return FAISS results if reranking fails
- Clear error messages with suggestions

### 5. Observability
- Telemetry for all function calls
- Performance metrics (latency, success rate)
- Usage patterns for optimization
- Session-based analytics

### 6. Performance Optimization
- Lazy loading of data
- Caching frequently accessed data
- Async/await for non-blocking I/O
- Connection pooling (Redis)
- Batch processing support

---

## Implementation Checklist

When implementing a new MCP system following this standard pattern:

### Architecture
- [ ] Four-layer architecture (Registration → Functions → Services → Data)
- [ ] AIQ integration with Pydantic schemas
- [ ] Async function implementations
- [ ] Standard return format (`{success, result, error}`)

### Services
- [ ] Atlas Service for structured knowledge
- [ ] Retrieval Service for semantic search
- [ ] Reranking Service (optional)
- [ ] Telemetry Service for usage tracking

### Data
- [ ] FAISS vector databases for semantic search
- [ ] JSON Atlas database for entity reference
- [ ] Instruction files for learning materials
- [ ] Code example collections (if applicable)

### Functions
- [ ] Search functions for semantic queries
- [ ] List functions for entity enumeration
- [ ] Detail functions with flexible input handling
- [ ] Instruction functions for documentation access

### Features
- [ ] Flexible input format handling
- [ ] Batch operations support
- [ ] Fuzzy matching with suggestions
- [ ] Graceful degradation
- [ ] Comprehensive error messages

### Configuration
- [ ] NVIDIA API key environment variable
- [ ] Configurable embedding service
- [ ] Configurable reranking service
- [ ] Telemetry toggle option

---

## Standard Workflow Examples

### Discovery Workflow
```python
# 1. List available entities
entities = get_entities()

# 2. Search for specific functionality
results = search_entities("what I need")

# 3. Get detailed information
details = get_entity_detail(["entity1", "entity2"])

# 4. Access usage instructions
instructions = get_kit_instructions("entity1")
```

### Learning Workflow
```python
# 1. Load system fundamentals
fundamentals = get_kit_instructions("system_fundamentals")

# 2. Explore available entities
entities = get_entities()

# 3. Get entity documentation
docs = get_entity_detail(["key_entity"])

# 4. Find code examples
examples = search_kit_code_examples("entity usage")
```

### Implementation Workflow
```python
# 1. Search for similar examples
examples = search_kit_code_examples("what I want to build")

# 2. Get entity details
entity_docs = get_entity_detail(["required_entities"])

# 3. Check relationships/dependencies
relationships = get_entity_relationships("main_entity")

# 4. Load styling/configuration docs
config = get_configuration_docs(["relevant_section"])
```

---

## Conclusion

The Standard MCP Pattern provides a consistent, proven architecture for building AI-powered knowledge access systems. By following this pattern, new MCP implementations benefit from:

- **Consistency**: Familiar structure for developers and AI models
- **Reusability**: Common services and utilities across systems
- **Scalability**: Proven performance optimizations
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to add new entity types or search capabilities

Both Kit Functions MCP and Omni UI Functions MCP demonstrate this pattern's effectiveness in providing comprehensive, intelligent access to complex technical ecosystems.

---

**Version**: 1.0.0
**Last Updated**: 2025-01-10
**Applies To**: Kit Functions MCP, Omni UI Functions MCP, and future MCP systems
