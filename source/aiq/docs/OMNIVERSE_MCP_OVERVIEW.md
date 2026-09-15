# Omniverse MCP Tools Overview

## Introduction

The **Omniverse MCP (Model Context Protocol) Tools** are two complementary AI-powered systems that provide intelligent access to the entire NVIDIA Omniverse Kit development ecosystem. These tools enable AI models to discover, learn, and implement Omniverse applications through semantic search, structured knowledge retrieval, and comprehensive documentation access.

### What is MCP?

Model Context Protocol (MCP) is a standardized interface for connecting AI models to external tools and knowledge sources. The Omniverse MCP tools implement this protocol to provide:

- **Semantic Search**: Natural language queries for finding relevant information
- **Structured Knowledge**: API documentation, code examples, and implementation patterns
- **Context-Aware Access**: Intelligent filtering and relevance ranking
- **Batch Operations**: Efficient bulk data retrieval
- **Telemetry**: Usage tracking for continuous improvement

### The Two Systems

The Omniverse ecosystem is served by two specialized MCP systems, each optimized for different aspects of development:

1. **Kit Functions MCP**: Complete access to the Kit framework ecosystem (400+ extensions, APIs, settings)
2. **Omni UI Functions MCP**: Specialized access to the OmniUI framework (150+ classes, styling, patterns)

Together, these systems provide comprehensive coverage of the entire Omniverse development stack.

---

## Kit Functions MCP

### Purpose

The Kit Functions MCP provides AI models with intelligent access to NVIDIA Omniverse Kit's extensive ecosystem of **400+ extensions**, **thousands of APIs**, configuration settings, code examples, and application templates.

### Core Capabilities

**Extension Discovery**:
- Search across 400+ Kit extensions using natural language
- Get detailed extension metadata, features, and dependencies
- Analyze dependency trees and relationships
- Discover extensions by functionality, category, or use case

**API Documentation**:
- Access detailed API documentation with signatures and parameters
- List all APIs provided by specific extensions
- Get complete docstrings and usage information
- Symbol resolution with format `extension_id@symbol`

**Code and Test Examples**:
- Semantic search for production code patterns
- Find test implementations and testing patterns
- Locate working examples across the Kit codebase
- Tagged categorization (async, ui, usd, etc.)

**Configuration Discovery**:
- Search across 1,000+ Kit settings
- Filter by category (app, exts, rtx, physics)
- Type-based filtering (bool, int, float, string, array, object)
- Find extension-specific configuration options

**Application Templates**:
- Search for appropriate application templates
- Access complete README and .kit configuration files
- Templates: kit_base_editor, usd_composer, usd_explorer, usd_viewer

**Development Guidance**:
- System instructions for Kit framework architecture
- Extension development patterns and best practices
- Testing strategies and framework usage
- USD integration and UI development guides

### Technology Stack

- **FAISS**: High-performance semantic search
- **NVIDIA Embeddings**: nemotron-3-embed-1b for query understanding
- **Code Atlas**: Per-extension code analysis
- **Extensions Database**: 400+ extensions with complete metadata
- **Settings Database**: 1,000+ configuration options

### Available Tools (11 Functions)

1. `search_kit_extensions` - Find extensions by functionality
2. `get_kit_extension_details` - Complete extension information
3. `get_kit_extension_dependencies` - Dependency tree analysis
4. `get_kit_extension_apis` - List APIs for extensions
5. `get_kit_api_details` - Detailed API documentation
6. `search_kit_code_examples` - Find code implementations
7. `search_kit_test_examples` - Find test patterns
8. `search_kit_settings` - Configuration discovery
9. `search_kit_app_templates` - Find application templates
10. `get_kit_app_template_details` - Complete template details
11. `get_{scope}_instructions` - Development guidance

### Key Benefits

- **Comprehensive Coverage**: Access to 400+ extensions, thousands of APIs
- **Smart Search**: NVIDIA-powered semantic search finds relevant results
- **Dependency Understanding**: Analyze and visualize extension relationships
- **Code Discovery**: Find working examples instead of starting from scratch
- **Template-Based Start**: Begin with proven application architectures

### Example Workflow

```python
# 1. Find relevant extensions
results = search_kit_extensions("viewport and camera management")

# 2. Get detailed information
details = get_kit_extension_details("omni.kit.viewport.window")

# 3. Check dependencies
deps = get_kit_extension_dependencies("omni.kit.viewport.window", depth=2)

# 4. Explore APIs
apis = get_kit_extension_apis("omni.kit.viewport.window")

# 5. Get API documentation
docs = get_kit_api_details(["omni.kit.viewport.window@ViewportWindow"])

# 6. Find implementation examples
examples = search_kit_code_examples("create viewport window")
```

---

## Omni UI Functions MCP

### Purpose

The Omni UI Functions MCP provides AI models with specialized access to the OmniUI framework, covering **150+ UI classes**, comprehensive styling documentation, code examples, and implementation patterns for building Omniverse user interfaces.

### Core Capabilities

**API Discovery**:
- List all 150+ OmniUI classes and 50+ modules
- Get detailed class information with methods and docstrings
- Explore module contents and relationships
- Access method signatures with parameters and return types
- Fuzzy matching for typo-tolerant lookups

**Semantic Code Search**:
- Find OmniUI implementation patterns using natural language
- Specialized window and dialog example search
- NVIDIA embeddings + FAISS + reranking pipeline
- Production-quality code examples from real implementations

**Structured Learning**:
- **System Instructions**: 4 comprehensive guides (framework fundamentals, classes, 3D UI, core UI)
- **Class Instructions**: 61+ per-class usage guides organized in 10 categories
- **Style Documentation**: 37,820+ tokens covering complete styling system

**Categories** (61+ Classes):
- Models (3): Data models and delegates
- Shapes (12): Geometric primitives
- Widgets (11): Interactive UI components
- Containers (7): Layout containers
- Layouts (3): Layout managers
- Inputs (7): Input controls
- Windows (6): Windows and dialogs
- Scene (9): 3D UI components
- Units (3): Measurement types
- System (1): Style system

**Styling System**:
- Complete CSS-like styling reference
- Color palettes and theme management
- Widget-specific styling (buttons, sliders, containers)
- State selectors (hover, pressed, disabled)
- Typography and measurement systems

### Technology Stack

- **FAISS**: Semantic search for code examples
- **NVIDIA Embeddings**: nemotron-3-embed-1b for query understanding
- **NVIDIA Reranking**: llama-nemotron-rerank-vl-1b-v2 for relevance scoring
- **UI Atlas Database**: Complete API reference (150+ classes, 1,000+ methods)
- **RAG Collection**: Curated code example corpus
- **Instruction Library**: System and per-class usage guides

### Available Tools (10 Functions)

1. `search_ui_code_examples` - Semantic code search with reranking
2. `search_ui_window_examples` - Specialized window pattern search
3. `list_ui_classes` - List all OmniUI classes
4. `list_ui_modules` - List all OmniUI modules
5. `get_ui_class_detail` - Detailed class information
6. `get_ui_module_detail` - Detailed module information
7. `get_ui_method_detail` - Method signatures and docs
8. `get_ui_instructions` - System-level documentation
9. `get_ui_class_instructions` - Per-class usage guides
10. `get_ui_style_docs` - Comprehensive styling reference

### Key Benefits

- **Complete UI Coverage**: 150+ classes, 50+ modules, 1,000+ methods fully documented
- **Advanced RAG Pipeline**: NVIDIA embeddings + FAISS + reranking for highest relevance
- **Organized Learning**: 61+ class guides in 10 categories for structured exploration
- **Comprehensive Styling**: 37,820+ tokens covering all theming and customization aspects
- **Flexible Inputs**: Supports strings, arrays, JSON, comma-separated values for maximum compatibility
- **Batch Operations**: 70-80% faster when retrieving multiple entities

### Example Workflow

```python
# 1. Load framework fundamentals
system = get_ui_instructions("omni_ui_system")

# 2. List available widgets
classes = list_ui_classes()

# 3. Get class documentation
button_docs = get_ui_class_detail(["Button", "Label", "TreeView"])

# 4. Learn usage patterns
guides = get_ui_class_instructions(["Button", "Label"])

# 5. Find implementation examples
examples = search_ui_code_examples("button with custom style and click handler")

# 6. Get styling reference
styles = get_ui_style_docs(["buttons", "widgets"])

# 7. Find window patterns
window = search_ui_window_examples("modal dialog with buttons")
```

---

## How They Complement Each Other

### Division of Responsibility

**Kit Functions MCP** focuses on:
- Extension ecosystem (what exists, what it does)
- Extension APIs and dependencies
- Kit configuration and settings
- Application architecture and templates
- Cross-extension code patterns

**Omni UI Functions MCP** focuses on:
- UI framework specifics (widgets, layouts, styling)
- OmniUI API details and usage patterns
- Window and dialog implementations
- Theming and customization
- UI-specific code examples

### Combined Usage Example

Building a custom Omniverse application with UI:

```python
# === Kit Functions MCP: Application Setup ===

# 1. Find UI-related extensions
ui_extensions = search_kit_extensions("UI development tools")

# 2. Choose application template
template = get_kit_app_template_details("kit_base_editor")

# 3. Understand dependencies
deps = get_kit_extension_dependencies("omni.ui", depth=2)

# 4. Configure settings
settings = search_kit_settings("UI window", prefix_filter="app")


# === Omni UI Functions MCP: UI Implementation ===

# 5. Learn UI fundamentals
fundamentals = get_ui_instructions("omni_ui_system")

# 6. Find widget documentation
widgets = get_ui_class_detail(["Button", "Label", "VStack", "Window"])

# 7. Get implementation examples
window_example = search_ui_window_examples("application main window")
button_example = search_ui_code_examples("button click handler")

# 8. Apply styling
theme = get_ui_style_docs(["overview", "widgets", "buttons"])


# === Kit Functions MCP: Extension Integration ===

# 9. Find extension APIs to integrate
extension_apis = get_kit_extension_apis("omni.kit.window.console")
api_docs = get_kit_api_details(["omni.kit.commands@execute"])

# 10. Study integration patterns
integration = search_kit_code_examples("extension service usage")
```

### When to Use Each

**Use Kit Functions MCP when**:
- Starting a new Omniverse project
- Searching for existing extensions to use
- Understanding Kit architecture and configuration
- Analyzing extension dependencies
- Finding application templates
- Locating APIs across different extensions

**Use Omni UI Functions MCP when**:
- Building user interfaces
- Customizing UI appearance and theming
- Learning OmniUI widgets and containers
- Finding UI implementation patterns
- Creating windows and dialogs
- Understanding layout systems
- Applying styles and themes

---

## Value Proposition

### For AI Models

**Comprehensive Knowledge Access**:
- Complete coverage of Omniverse Kit and OmniUI ecosystems
- Structured, machine-readable documentation
- Semantic search for natural language queries
- Context-aware results through reranking

**Efficient Context Usage**:
- Batch operations reduce API calls by 70-80%
- Relevant examples instead of entire documentation
- Incremental learning path from basics to advanced
- Flexible input formats reduce integration complexity

**Production-Ready Information**:
- Real code examples from working implementations
- Complete API signatures with types and parameters
- Validated patterns and best practices
- Up-to-date documentation synchronized with framework

### For Developers

**Accelerated Development**:
- Find relevant extensions and APIs in seconds, not hours
- Copy working code examples instead of trial-and-error
- Understand dependencies before integration
- Start with proven application templates

**Comprehensive Learning**:
- Structured learning paths from basics to advanced
- Per-class usage guides with examples
- Complete styling reference for customization
- System instructions for framework understanding

**Better Code Quality**:
- Learn from production-quality examples
- Understand best practices and patterns
- Access complete API documentation
- Discover testing patterns and strategies

### For Organizations

**Reduced Onboarding Time**:
- New developers find information quickly
- Standardized patterns across team
- Self-service documentation access
- AI-assisted development support

**Increased Productivity**:
- Less time searching, more time building
- Reuse proven patterns and templates
- Faster prototyping and experimentation
- Efficient debugging with example patterns

**Better Maintainability**:
- Discover dependencies before they become problems
- Understand configuration options comprehensively
- Access API documentation instantly
- Find test patterns for quality assurance

---

## Getting Started

### Prerequisites

**Environment Setup**:
```bash
# Required: NVIDIA API key for embeddings and reranking
export NVIDIA_API_KEY="REPLACE_WITH_NVIDIA_API_KEY"

# Optional: Disable usage logging if needed
export OMNI_UI_DISABLE_USAGE_LOGGING="false"
```

**Installation**:
```bash
# Kit Functions MCP
pip install kit-fns

# Omni UI Functions MCP
pip install omni-ui-fns
```

### Quick Start: Kit Functions MCP

```python
from kit_fns import (
    search_kit_extensions,
    get_kit_extension_details,
    get_kit_api_details,
    search_kit_code_examples
)

# Find extensions
results = await search_kit_extensions("viewport tools", top_k=5)

# Get details
details = await get_kit_extension_details(["omni.kit.viewport.window"])

# Explore APIs
apis = await get_kit_api_details(["omni.kit.viewport.window@ViewportWindow"])

# Find examples
examples = await search_kit_code_examples("viewport creation")
```

### Quick Start: Omni UI Functions MCP

```python
from omni_ui_fns import (
    list_ui_classes,
    get_ui_class_detail,
    search_ui_code_examples,
    get_ui_class_instructions,
    get_ui_style_docs
)

# List classes
classes = await list_ui_classes()

# Get class documentation
docs = await get_ui_class_detail(["Button", "Label"])

# Find examples
examples = await search_ui_code_examples("button with icon")

# Get usage guides
guides = await get_ui_class_instructions(["Button"])

# Get styling
styles = await get_ui_style_docs(["buttons"])
```

### Integration with AIQ Workflows

Both MCP systems integrate seamlessly with AIQ workflows:

```yaml
# config.yaml
functions:
  # Kit Functions
  search_kit_extensions:
    _type: kit_fns/search_kit_extensions
    verbose: false

  get_kit_api_details:
    _type: kit_fns/get_kit_api_details
    verbose: false

  # Omni UI Functions
  search_ui_code_examples:
    _type: omni_ui_fns/search_ui_code_examples
    verbose: false
    enable_rerank: true
    rerank_k: 10

  get_ui_class_detail:
    _type: omni_ui_fns/get_ui_class_detail
    verbose: false

workflow:
  tool_names:
    - search_kit_extensions
    - get_kit_api_details
    - search_ui_code_examples
    - get_ui_class_detail
```

---

## Performance Overview

### Latency Characteristics

| Operation Type | Kit Functions | Omni UI Functions |
|----------------|---------------|-------------------|
| Semantic Search | 50-200ms | 2-3s (with reranking) |
| API Lookups | 50-100ms | 50-100ms |
| Batch Operations | 200ms+ | 200ms+ |
| Document Retrieval | <100ms | <100ms |

**Note**: Semantic search in Omni UI Functions includes NVIDIA embedding + FAISS + reranking for maximum relevance.

### Optimization Tips

1. **Use Batch Operations**: Retrieve multiple items in single calls (70-80% faster)
2. **Cache Static Data**: Classes, modules, instructions rarely change
3. **Specific Queries**: Better queries yield better semantic search results
4. **Disable Reranking**: Skip reranking when speed is critical (Omni UI Functions)

---

## Documentation References

### Comprehensive Documentation

- **Kit Functions MCP**: [`KIT_FNS_MCP_DOCUMENTATION.md`](../kit_fns/docs/KIT_FNS_MCP_DOCUMENTATION.md)
- **Omni UI Functions MCP**: [`OMNI_UI_FNS_MCP_DOCUMENTATION.md`](../omni_ui_fns/docs/OMNI_UI_FNS_MCP_DOCUMENTATION.md)

### Additional Resources

**Kit Functions MCP**:
- Extension database structure
- API reference format
- Settings hierarchy
- Application template details
- Usage patterns and examples

**Omni UI Functions MCP**:
- System instructions (agent_system, classes, omni_ui_scene_system, omni_ui_system)
- Tools reference with all 10 functions
- Class instruction categories
- Complete styling guide
- Performance optimization guide

---

## Telemetry and Analytics

Both systems include comprehensive telemetry:

- **Call Tracking**: Function usage patterns and frequency
- **Performance Metrics**: Execution times and latency
- **Success Rates**: Error tracking and failure analysis
- **Parameter Analysis**: Common query patterns
- **Session Analytics**: User session grouping

**Storage**: Redis Streams for scalable, distributed telemetry

**Privacy**: Telemetry can be disabled via environment variables

---

## Conclusion

The Omniverse MCP Tools represent a transformative approach to AI-assisted development for the NVIDIA Omniverse platform. By providing intelligent, context-aware access to comprehensive documentation, code examples, and API references, these systems enable:

- **Faster Development**: Find what you need in seconds
- **Better Quality**: Learn from production-proven patterns
- **Easier Learning**: Structured paths from basics to advanced
- **AI Empowerment**: Tools designed for AI model consumption

Together, **Kit Functions MCP** and **Omni UI Functions MCP** cover the complete Omniverse development stack, from application architecture and extension discovery to UI implementation and styling customization.

**Start building smarter with Omniverse MCP Tools.**

---

**Version**: 1.0.0
**Last Updated**: 2025-01-10
**Maintained by**: Omniverse GenAI Team
**Repository**: https://github.com/NVIDIA-Omniverse/kit-usd-agents
