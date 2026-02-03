# Kit MCP Architecture - Comprehensive Tool Definition and Implementation Plan

## Overview

The Kit MCP (Model Context Protocol) provides a hierarchical, developer-focused toolkit for working with NVIDIA Omniverse Kit applications. This architecture defines two primary MCP servers that work together to enable AI-powered development workflows:

1. **Kit Docs MCP** - Documentation, API discovery, and code examples
2. **Kit Runtime MCP** - Live Kit instance interaction and debugging

## Architecture Principles

### 1. Hierarchical Information Retrieval
Tools are organized in a scaffolded manner, allowing progressive discovery:
- Start with high-level system instructions and extension discovery
- Drill down into specific APIs and implementations
- Access detailed documentation and examples as needed

### 2. Flexible Input Handling
Following the OmniUI MCP pattern, all tools support multiple input formats:
- Native arrays for batch processing
- JSON strings for compatibility
- Comma-separated values for convenience
- Single values for simple queries

### 3. Batch Processing Optimization
Tools are designed to handle multiple items efficiently:
- Single API call for multiple extensions/APIs
- Reduced context window usage
- 60-80% faster than sequential queries

### 4. Error Handling and Validation
Comprehensive error handling with clear messages:
- Input validation at wrapper level
- Graceful degradation for missing items
- Detailed error context for debugging

## Kit Docs MCP - Tool Definitions

### 1. System Instructions and Documentation

#### `get_kit_instructions`
**Purpose**: Retrieve comprehensive Kit framework documentation and best practices

**Input Schema**:
```python
class GetInstructionsInput(BaseModel):
    instruction_sets: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Instruction sets to retrieve. Accepts:
        - 'kit_system': Core Kit framework fundamentals and architecture
        - 'extensions': Extension development guidelines and patterns
        - 'testing': Test writing best practices and framework usage
        - 'usd': USD integration and scene description patterns
        - 'ui': UI development with Kit widgets and layouts
        - Empty/null: Lists all available instruction sets"""
    )
```

**Returns**: Formatted documentation with metadata, use cases, and examples

**Usage Patterns**:
- Load `kit_system` when starting Kit development
- Load `extensions` for extension architecture guidance
- Load `testing` for test framework documentation
- Call without parameters to list all available instructions

### 2. Extension Discovery and Analysis

#### `search_kit_extensions`
**Purpose**: Semantic search across 400+ Kit extensions using RAG techniques

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
    categories: Optional[List[str]] = Field(
        None,
        description="Filter by extension categories (ui, rendering, physics, etc.)"
    )
```

**Returns**: Ranked list of relevant extensions with scores and brief descriptions

**Implementation Details**:
- Uses NVIDIA embedding models for semantic search
- Pre-indexed extension metadata and documentation
- Includes relevance scoring and category filtering

#### `get_kit_extension_details`
**Purpose**: Retrieve comprehensive information about specific extensions

**Input Schema**:
```python
class GetExtensionDetailsInput(BaseModel):
    extension_ids: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Extension IDs to retrieve. Accepts:
        - Single ID: 'omni.kit.window.console'
        - Array: ['omni.ui', 'omni.kit.window.file']
        - Comma-separated: 'omni.ui, omni.kit.window.file'
        - Empty/null: Lists available extensions"""
    )
```

**Returns**: Detailed extension information including: (2-4k token max)
- Key features and objectives
- Dependencies and requirements
- Configuration options
- Basic usage patterns

#### `get_kit_extension_dependencies`
**Purpose**: Analyze and visualize extension dependency graphs

**Input Schema**:
```python
class GetExtensionDependenciesInput(BaseModel):
    extension_id: str = Field(
        description="Extension ID to analyze dependencies for"
    )
    depth: Optional[int] = Field(
        default=2,
        description="Dependency tree depth to explore"
    )
    include_optional: Optional[bool] = Field(
        default=False,
        description="Include optional dependencies"
    )
```

**Returns**: Dependency tree with version requirements and conflict analysis

### 3. API Reference and Documentation

#### `get_kit_extension_apis`
**Purpose**: List all APIs provided by extensions

**Input Schema**:
```python
class GetExtensionAPIsInput(BaseModel):
    extension_ids: Optional[Union[str, List[str]]] = Field(
        description="Extension IDs to get APIs for"
    )
```

**Returns**: Structured API listing with:
- Classes and their methods
- Functions and parameters
- Effectively python_api.md

#### `get_kit_api_details`
**Purpose**: Get detailed documentation for specific APIs

**Input Schema**:
```python
class GetAPIDetailsInput(BaseModel):
    api_references: Optional[Union[str, List[str]]] = Field(
        description="""API references in format 'extension_id@symbol'. Accepts:
        - Single: 'omni.ui@Window'
        - Array: ['omni.ui@Window.__init__', 'omni.ui@Button.clicked_fn']
        - Multiple classes: 'omni.ui@Window, omni.ui@Button'"""
    )
```

**Returns**: Complete API documentation with:
- Full docstrings
- Parameter specifications
- Return types

### 4. Code Examples and Patterns

#### `search_kit_code_examples`
**Purpose**: Find relevant code examples using semantic search

**Input Schema**:
```python
class SearchCodeExamplesInput(BaseModel):
    query: str = Field(
        description="Description of desired code functionality"
    )
```

**Returns**: Formatted code examples with:
- Complete implementation code
- Relative file paths and line numbers
- extensions ids

#### `search_kit_test_examples`
**Purpose**: Find test implementations and patterns

**Input Schema**:
```python
class SearchTestExamplesInput(BaseModel):
    query: str = Field(
        description="Test scenario or functionality to find examples for"
    )
```

**Returns**: Test examples with setup, execution, and validation patterns

## Kit Runtime MCP - Tool Definitions

### 1. Debugging and Monitoring

#### `get_kit_logs`
**Purpose**: Retrieve Kit application logs for debugging

**Input Schema**:
```python
class GetKitLogsInput(BaseModel):
    since_timestamp: Optional[float] = Field(
        None,
        description="Unix timestamp to get logs from (default: last 5 minutes)"
    )
    log_level: Optional[str] = Field(
        None,
        description="Filter by level: 'ERROR', 'WARNING', 'INFO', 'DEBUG'"
    )
    filter_pattern: Optional[str] = Field(
        None,
        description="Regex pattern to filter log messages"
    )
    max_lines: Optional[int] = Field(
        default=1000,
        description="Maximum number of log lines to return"
    )
```

**Returns**: Formatted log entries with timestamps, levels, and messages

### 2. Visual Validation

#### `capture_window_screenshot`
**Purpose**: Capture screenshot of specific Kit window

**Input Schema**:
```python
class CaptureWindowScreenshotInput(BaseModel):
    window_name: str = Field(
        description="Name of the window to capture"
    )
    format: Optional[str] = Field(
        default="png",
        description="Image format: 'png', 'jpg', 'base64'"
    )
```

**Returns**: Image data or base64-encoded string

#### `capture_viewport_screenshot`
**Purpose**: Capture viewport rendering

**Input Schema**:
```python
class CaptureViewportScreenshotInput(BaseModel):
    viewport_index: Optional[int] = Field(
        default=0,
        description="Viewport index for multi-viewport applications"
    )
```

**Returns**: Viewport image with rendered scene

#### `capture_application_screenshot`
**Purpose**: Capture entire Kit application window

**Input Schema**:
```python
class CaptureApplicationScreenshotInput(BaseModel):
    pass
    # no inputs for now
```

**Returns**: Full application screenshot

### 3. Runtime Inspection and Execution

#### `execute_code`
**Purpose**: Execute Python code in Kit runtime (requires explicit approval)

**Input Schema**:
```python
class ExecuteCodeInput(BaseModel):
    code: str = Field(
        description="Python code to execute in Kit context"
    )
    timeout: Optional[float] = Field(
        default=10.0,
        description="Execution timeout in seconds"
    )
    capture_output: Optional[bool] = Field(
        default=True,
        description="Capture and return output"
    )
```

**Returns**: Execution result with output, errors, and return values

**Security Notes**:
- Disabled by default
- Requires explicit user approval per execution
- Sandboxed execution environment
- Audit logging of all executions

## Implementation Architecture

### Registration Pattern
Following the OmniUI MCP pattern, each tool follows this structure:

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
User Query → MCP Server → Input Validation → Tool Function
    ↓                                              ↓
Response ← Format Output ← Process Result ← Data Retrieval
```

### Caching Strategy
- Instructions and API docs: 15-minute cache
- Extension metadata: 5-minute cache
- Code examples: No cache (always fresh)
- Screenshots: No cache (real-time)
- Logs: Stream with buffering

### Error Handling Hierarchy
1. Input validation errors (immediate)
2. Data retrieval errors (with retry)
3. Processing errors (with fallback)
4. Format errors (with raw data option)

## Performance Considerations

### Batch Processing Benefits
- 60-80% reduction in API calls
- Optimized embedding computations
- Reduced network overhead
- Better context window utilization

### Optimization Techniques
1. **Lazy Loading**: Load data only when requested
2. **Incremental Retrieval**: Support pagination for large results
3. **Smart Caching**: Cache immutable data, refresh dynamic data
4. **Parallel Processing**: Execute independent queries concurrently

## Security and Permissions

### Tool Permission Levels
1. **Read-Only** (Default):
   - All documentation tools
   - Log retrieval
   - Screenshot capture

2. **Elevated** (Requires Approval):
   - Code execution
   - File system access
   - Network requests

### Audit and Logging
- All tool invocations logged
- Parameter sanitization
- Rate limiting per tool
- Usage analytics for optimization

## Integration with AIQ Framework

The Kit MCP tools integrate with NVIDIA's AIQ framework following these patterns:

1. **Function Registration**: All tools registered as AIQ functions
2. **Workflow Composition**: Tools can be combined in complex workflows
3. **LLM Framework Support**: Compatible with LangChain, CrewAI, etc.
4. **Profiling Integration**: Built-in performance profiling
5. **Error Recovery**: Automatic retry and fallback mechanisms

## Tool Description Templates

### Documentation Tool Template
```
GET_[TOOL]_DESCRIPTION = """Retrieve [specific content] for Kit development.

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

### Kit Docs MCP Tools
| Tool | Purpose | Input Type | Batch Support |
|------|---------|------------|---------------|
| `get_kit_instructions` | System documentation | String/Array | ✓ |
| `search_kit_extensions` | Extension discovery | String | ✗ |
| `get_kit_extension_details` | Extension info | String/Array | ✓ |
| `get_kit_extension_dependencies` | Dependency analysis | String | ✗ |
| `get_kit_extension_apis` | API listing | String/Array | ✓ |
| `get_kit_api_details` | API documentation | String/Array | ✓ |
| `search_kit_code_examples` | Code search | String | ✗ |
| `search_kit_test_examples` | Test search | String | ✗ |

### Kit Runtime MCP Tools
| Tool | Purpose | Permission | Real-time |
|------|---------|------------|-----------|
| `get_kit_logs` | Log retrieval | Read-only | ✓ |
| `capture_window_screenshot` | Window capture | Read-only | ✓ |
| `capture_viewport_screenshot` | Viewport capture | Read-only | ✓ |
| `capture_application_screenshot` | App capture | Read-only | ✓ |
| `execute_code` | Code execution | Elevated | ✓ |

## Future Enhancements

### Planned Features
1. **Interactive Debugging**: Step-through debugging support
2. **Performance Profiling**: Detailed performance metrics
3. **USD Scene Manipulation**: Direct USD graph editing
4. **Multi-Instance Support**: Connect to multiple Kit instances
5. **Collaborative Features**: Shared debugging sessions

### Extension Points
1. Custom tool registration API
2. Plugin architecture for data sources
3. Webhook support for events
4. Custom authentication providers
5. Extended profiling metrics

## Conclusion

This architecture provides a comprehensive, scalable foundation for AI-powered Kit development workflows. By following established patterns from OmniUI MCP and leveraging the AIQ framework, we ensure consistency, maintainability, and optimal performance across all tools.