# Isaac Sim MCP Architecture - Comprehensive Tool Definition and Implementation Plan

## Overview

The Isaac Sim MCP (Model Context Protocol) provides a hierarchical, developer-focused toolkit for working with NVIDIA Omniverse Isaac Sim applications. This architecture defines a comprehensive MCP server that enables AI-powered robotics and simulation development workflows.

The Isaac Sim MCP provides intelligent assistance for Isaac Sim developers by offering complete access to Isaac Sim extensions, APIs, documentation, and code examples through 5 specialized tools.

## Architecture Principles

### 1. Hierarchical Information Retrieval
Tools are organized in a scaffolded manner, allowing progressive discovery:
- Start with high-level system instructions for Isaac Sim and robotics
- Drill down into specific extensions and implementations
- Access detailed code examples and settings as needed

### 2. Flexible Input Handling
Following established MCP patterns, all tools support multiple input formats:
- Native arrays for batch processing
- JSON strings for compatibility
- Comma-separated values for convenience
- Single values for simple queries

### 3. Batch Processing Optimization
Tools are designed to handle multiple items efficiently:
- Single API call for multiple extensions
- Reduced context window usage
- 70% faster than sequential queries

### 4. Error Handling and Validation
Comprehensive error handling with clear messages:
- Input validation at wrapper level
- Graceful degradation for missing items
- Detailed error context for debugging
- Fuzzy matching for typo tolerance

## Isaac Sim MCP - Tool Definitions

### 1. System Instructions and Documentation

#### `get_isaac_sim_instructions`
**Purpose**: Retrieve comprehensive Isaac Sim framework documentation and best practices

**Input Schema**:
```python
class GetIsaacSimInstructionsInput(BaseModel):
    instruction_sets: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Instruction sets to retrieve. Accepts flexible formats:
        - Single string: "isaacsim_system"
        - Native array: ["isaacsim_system", "robot_setup", "sensors"]
        - JSON string: '["isaacsim_system", "physics"]'
        - Comma-separated: "isaacsim_system, robot_setup, sensors"
        - Empty/null: Lists all available instruction sets

        Available sets (49 total):
        Getting Started: isaacsim_system, what_is_isaac_sim, quick_install,
            installation, workflows, quick_tutorials, examples
        Programming: python_scripting_and_tutorials, omnigraph,
            development_tools, debugging_profiling, templates, application_template
        Robotics: robot_setup, robot_simulation, sensors,
            synthetic_data_generation, physics, ros_2, isaac_lab, digital_twin
        USD & Assets: omniverse_and_usd, asset_structure, isaac_sim_assets,
            browsers, importers_and_exporters
        UI & Reference: gui_reference, user_interface_reference,
            keyboard_shortcuts_reference
        Extensions: adding_and_updating_extensions_guide, api_documentation,
            renaming_extensions_in_isaac_sim_4_5
        Performance: isaac_sim_conventions,
            isaac_sim_performance_optimization_handbook, isaac_sim_benchmarks,
            reference_architecture_and_task_groupings
        Other: release_notes, glossary, help_faq, licenses,
            data_collection_usage, community_project_highlights,
            omniverse_feedback_and_forums"""
    )
```

**Returns**: Formatted documentation with metadata, use cases, and examples

**Usage Patterns**:
- Load `isaacsim_system` when starting Isaac Sim development for framework fundamentals
- Load `robot_setup` for robot building with Wizard, editors, and assembler
- Load `robot_simulation` for robot control, motion, and policy-based examples
- Load `sensors` for camera and sensor configuration
- Load `physics` for PhysX simulation setup
- Load `ros_2` for ROS 2 integration and bridging
- Call without parameters to list all 49 available instruction sets

### 2. Extension Discovery and Analysis

#### `search_isaac_sim_extensions`
**Purpose**: Semantic search across Isaac Sim extensions using RAG techniques

**Input Schema**:
```python
class SearchExtensionsInput(BaseModel):
    query: str = Field(
        description="Search query for finding relevant extensions"
    )
    top_k: Optional[int] = Field(
        default=10,
        description="Number of results to return"
    )
```

**Returns**: Ranked list of relevant extensions with scores and brief descriptions

**Implementation Details**:
- Uses NVIDIA embedding models for semantic search
- Pre-indexed extension metadata and documentation
- Includes relevance scoring and category filtering
- Fuzzy matching for extension name suggestions

#### `get_isaac_sim_extension_details`
**Purpose**: Retrieve comprehensive information about specific extensions

**Input Schema**:
```python
class GetExtensionDetailsInput(BaseModel):
    extension_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Extension IDs to retrieve. Accepts:
        - Single ID: 'omni.isaac.core'
        - Array: ['omni.isaac.core', 'omni.isaac.sensor']
        - Comma-separated: 'omni.isaac.core, omni.isaac.sensor'
        - Empty/null: Lists available extensions"""
    )
```

**Returns**: Detailed extension information including: (2-4k token max)
- Key features and objectives
- Dependencies and requirements
- Configuration options
- API availability and symbols
- Code Atlas metadata

### 3. Code Examples and Patterns

#### `search_isaac_sim_code_examples`
**Purpose**: Find relevant code examples using semantic search

**Input Schema**:
```python
class SearchCodeExamplesInput(BaseModel):
    query: str = Field(
        description="Description of desired code functionality"
    )
    top_k: Optional[int] = Field(
        default=10,
        description="Number of results to return"
    )
```

**Returns**: Formatted code examples with:
- Complete implementation code
- Relative file paths and line numbers
- Extension IDs and context
- Descriptions and use cases
- Relevance scores
- Associated tags

**Search Capabilities**:
- Robot control and manipulation examples
- Sensor integration patterns
- Physics simulation setup
- USD scene creation
- Articulation configuration

### 4. Settings Discovery and Configuration

#### `search_isaac_sim_settings`
**Purpose**: Search Isaac Sim configuration settings using semantic search

**Input Schema**:
```python
class SearchSettingsInput(BaseModel):
    query: str = Field(
        description="Natural language search query for settings"
    )
    top_k: Optional[int] = Field(
        default=20,
        description="Number of results to return"
    )
    prefix_filter: Optional[str] = Field(
        None,
        description="Filter by prefix: 'exts', 'app', 'persistent', 'rtx'"
    )
    type_filter: Optional[str] = Field(
        None,
        description="Filter by type: 'bool', 'int', 'float', 'string', 'array', 'object'"
    )
```

**Returns**: Settings information with:
- Full setting paths
- Data types and default values
- Documentation (when available)
- Extensions using each setting
- Usage counts across codebase
- Source file locations

**Setting Prefixes**:

Supported `prefix_filter` values:
- `/exts/` - Extension-specific settings (filter: `exts`)
- `/app/` - Application settings (filter: `app`)
- `/persistent/` - Settings saved between sessions (filter: `persistent`)
- `/rtx/` - Rendering settings (filter: `rtx`)

Additional path prefixes that exist in the data but are not available as filter values:
- `/physics/` - Physics simulation settings
- `/isaac/` - Isaac Sim specific settings
- `/renderer/` - General renderer settings

## Implementation Architecture

### Registration Pattern
Following established MCP patterns, each tool follows this structure:

```python
# 1. Input Schema Definition
class ToolNameInput(BaseModel):
    # Flexible input handling
    parameter: Optional[Union[str, List[str]]] = Field(...)

# 2. Configuration Class
class ToolNameConfig(FunctionBaseConfig, name="tool_name"):
    verbose: bool = Field(default=False)
    # Tool-specific configuration

# 3. Registration Function
@register_function(config_type=ToolNameConfig)
async def register_tool_name(config: ToolNameConfig, builder: Builder):
    # Wrapper with input validation
    # Usage logging
    # Error handling
    # Return FunctionInfo
```

### Data Flow Architecture

```
User Query → MCP Server → NAT Framework → Tool Function
    ↓                                           ↓
Response ← Format Output ← Process Result ← Service Layer
                                              ↓
                                         Data Layer (FAISS/JSON)
```

### Service Layer Architecture

```python
┌─────────────────────────────────────────────────────────────┐
│                   Tool Functions Layer                       │
│  • get_isaac_sim_instructions                                          │
│  • search_isaac_sim_extensions                                         │
│  • get_isaac_sim_extension_details                                     │
│  • search_isaac_sim_code_examples                                      │
│  • search_isaac_sim_settings                                           │
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

### Caching Strategy
- Instructions and API docs: 15-minute cache
- Extension metadata: 5-minute cache
- Code examples: No cache (always fresh)
- Settings: 5-minute cache

### Error Handling Hierarchy
1. Input validation errors (immediate)
2. Data retrieval errors (with retry)
3. Processing errors (with fallback)
4. Format errors (with raw data option)

## Data Collection Pipeline

Isaac Sim MCP relies on a comprehensive automated data collection pipeline that processes Isaac Sim extensions to generate all necessary data files.

### Pipeline Stages

1. **Extension Data Collection**
   - Processes `extension.toml` files for metadata
   - Generates Code Atlas for Python APIs
   - Creates public API documentation
   - Output: `extensions_database.json`, Code Atlas files

2. **Code Examples Extraction**
   - Analyzes Code Atlas to find interesting methods
   - Filters by size, complexity, and patterns
   - Extracts complete source code with context
   - Output: Extracted method files with metadata

3. **Settings Discovery**
   - Extracts settings from TOML files
   - Scans Python/C++ source for settings usage
   - Tracks usage counts and locations
   - Output: Settings summary and statistics

4. **Embeddings Generation**
   - Generates semantic embeddings using NVIDIA models
   - Processes extensions, code examples, and settings
   - Batch processing for efficiency
   - Output: Embedding JSON files

5. **FAISS Database Creation**
   - Builds FAISS vector databases from embeddings
   - Creates searchable indexes for similarity search
   - Stores metadata for result retrieval
   - Output: FAISS index files

### Versioned Output Structure

```
src/isaacsim_fns/data/{version}/
├── extensions/
│   ├── extensions_database.json
│   ├── extensions_summary.json
│   ├── codeatlas/              # Per-extension Code Atlas
│   ├── api_docs/               # Public API documentation
│   └── extensions_faiss/       # FAISS search database
├── code_examples/
│   ├── extension_analysis_summary.json
│   ├── extracted_methods/      # Extracted code examples
│   └── code_examples_faiss/    # FAISS search database
└── settings/
    ├── setting_summary.json
    ├── setting_summary_simple.json
    ├── setting_statistics.json
    └── settings_faiss/         # FAISS search database
```

## Performance Considerations

### Batch Processing Benefits
- 70% reduction in API calls
- Optimized embedding computations
- Reduced network overhead
- Better context window utilization

### Optimization Techniques
1. **Lazy Loading**: Load data only when requested
2. **Smart Caching**: Cache immutable data, refresh dynamic data
3. **Parallel Processing**: Execute independent queries concurrently
4. **FAISS Optimization**: Use flat L2 index for exact search

### Performance Metrics
- **FAISS Queries**: ~50-200ms per query
- **Extension Details**: ~10-50ms per extension
- **Code Examples**: ~100-300ms per search
- **Settings Search**: ~50-150ms per query

## Security and Permissions

### Tool Permission Levels
All tools are **Read-Only** (Default):
- Documentation retrieval
- Extension discovery
- Code example search
- Settings search

### Audit and Logging
- All tool invocations logged via telemetry
- Parameter sanitization
- Usage analytics for optimization
- Error tracking for improvement

## Integration with NAT Framework

The Isaac Sim MCP tools integrate with NVIDIA's NAT (NeMo Agent Toolkit) framework following these patterns:

1. **Function Registration**: All tools registered as NAT functions (with AIQ backwards-compatible entry points)
2. **Workflow Composition**: Tools can be combined in complex workflows
3. **LLM Framework Support**: Compatible with LangChain, CrewAI, etc.
4. **Telemetry Integration**: Built-in usage tracking via Redis
5. **Error Recovery**: Automatic retry and fallback mechanisms

## Tool Description Templates

### Documentation Tool Template
```
GET_[TOOL]_DESCRIPTION = """Retrieve [specific content] for Isaac Sim development.

WHAT IT DOES:
- [Primary function]
- [Secondary benefits]
- [Integration points]

PARAMETERS:
- [param_name]: [Type and description]
  * Format options and examples
  * Default behaviors

RETURNS:
- [Return format and structure]
- [Metadata included]

USAGE EXAMPLES:
[tool_name](param="value")
[tool_name](param=["value1", "value2"])

TIPS FOR BETTER RESULTS:
- [Usage recommendation 1]
- [Usage recommendation 2]
"""
```

### Search Tool Template
```
SEARCH_[TOOL]_DESCRIPTION = """Search for [content type] using semantic search.

QUERY MATCHING:
- [What the search compares against]
- [Types of content indexed]

RANKING:
- [How results are scored]
- [Reranking options if available]

FILTERS:
- [Available filtering options]
- [How to combine filters]
"""
```

## API Summary

### Isaac Sim MCP Tools
| Tool | Purpose | Input Type | Batch Support | Search Type |
|------|---------|------------|---------------|-------------|
| `get_isaac_sim_instructions` | System documentation | String/Array | ✓ | Direct |
| `search_isaac_sim_extensions` | Extension discovery | String | ✗ | Semantic |
| `get_isaac_sim_extension_details` | Extension info | String/Array | ✓ | Direct |
| `search_isaac_sim_code_examples` | Code search | String | ✗ | Semantic |
| `search_isaac_sim_settings` | Settings search | String | ✗ | Semantic |

## Technology Stack

- **NAT Framework (nvidia-nat >= 1.4.0)**: Function registration and MCP server
- **LangChain**: Vector store and embedding integrations
- **FAISS**: High-performance semantic search
- **NVIDIA Embeddings**: nv-embedqa-e5-v5 model
- **Pydantic**: Input validation and schema definition
- **Redis**: Distributed telemetry (optional)
- **Python 3.11+**: Core implementation

## Port Configuration

The Isaac Sim MCP server runs on port **9904** by default, allowing it to run alongside other MCP servers:
- Kit MCP: 9902
- Isaac Sim MCP: 9904

## Future Enhancements

### Planned Features
1. **Runtime Integration**: Live Isaac Sim instance interaction
2. **Scene Inspection**: USD scene graph analysis
3. **Robot Debugging**: Articulation and joint inspection
4. **Performance Profiling**: Simulation performance metrics
5. **Multi-Version Support**: Support for multiple Isaac Sim versions

### Extension Points
1. Custom tool registration API
2. Plugin architecture for data sources
3. Custom embedding models
4. Extended telemetry metrics
5. Reranking model integration

## Conclusion

This architecture provides a comprehensive, scalable foundation for AI-powered Isaac Sim development workflows. By combining semantic search, structured data access, and intelligent caching, it enables AI models to efficiently discover and utilize Isaac Sim's extensive robotics and simulation capabilities.

The system's flexible input handling, batch processing support, and hierarchical information retrieval make it an essential tool for AI-assisted Isaac Sim development, from initial learning to advanced robotics application creation.
