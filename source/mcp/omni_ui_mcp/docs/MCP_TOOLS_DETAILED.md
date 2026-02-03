# OmniUI MCP Tools - Detailed Documentation

## Overview

The OmniUI MCP (Model Context Protocol) Server provides a comprehensive set of tools for accessing and leveraging OmniUI framework documentation, code examples, and API information. This document provides detailed information about each tool, its parameters, return types, and usage patterns.

## Tool Categories

The MCP tools are organized into the following functional categories:

1. **Discovery Tools** - Finding available classes and modules
2. **Detail Tools** - Getting detailed information about specific components
3. **Instruction Tools** - Accessing comprehensive documentation
4. **Code Example Tools** - Retrieving working code examples

---

## 1. Discovery Tools

### 1.1 list_ui_classes

**Purpose**: Retrieves a complete list of all available OmniUI classes from the Atlas database.

**Parameters**:
- None (zero-argument function)

**Return Type**: `Dict[str, Any]`
```python
{
    "success": bool,
    "result": str,  # JSON string containing class data
    "error": Optional[str]
}
```

**Result Structure** (when successful):
```json
{
    "class_full_names": ["FilterButton", "OptionsMenu", "SearchField", ...],
    "total_count": 150,
    "description": "OmniUI classes from Atlas data"
}
```

**Use Cases**:
- Discovering available UI components
- Getting a comprehensive list of all widgets
- Finding class names for further detail queries
- Understanding the breadth of the OmniUI framework

**Example Usage**:
```python
result = await list_ui_classes()
```

---

### 1.2 list_ui_modules

**Purpose**: Retrieves a complete list of all OmniUI modules from the Atlas database.

**Parameters**:
- None (zero-argument function)

**Return Type**: `Dict[str, Any]`
```python
{
    "success": bool,
    "result": str,  # JSON string containing module data
    "error": Optional[str]
}
```

**Result Structure** (when successful):
```json
{
    "module_names": ["omni.ui", "omni.ui.scene", "omni.ui.workspace", ...],
    "total_count": 25,
    "description": "OmniUI modules from Atlas data"
}
```

**Use Cases**:
- Understanding module organization
- Finding module namespaces
- Exploring framework structure
- Identifying feature areas

**Example Usage**:
```python
result = await list_ui_modules()
```

---

## 2. Detail Tools

### 2.1 get_ui_class_detail

**Purpose**: Retrieves comprehensive information about a specific OmniUI class.

**Parameters**:
- `class_name` (str): Name of the class to look up (can be partial or full name)

**Return Type**: `Dict[str, Any]`
```python
{
    "success": bool,
    "result": str,  # JSON string containing detailed class information
    "error": Optional[str]
}
```

**Result Structure** (when successful):
```json
{
    "full_name": "omni.ui.Button",
    "short_name": "Button",
    "module": "omni.ui",
    "description": "A clickable button widget",
    "methods": [
        {
            "name": "__init__",
            "parameters": ["text", "clicked_fn", "width", "height"],
            "description": "Initialize button"
        }
    ],
    "properties": [...],
    "parent_classes": ["Widget"],
    "examples": [...]
}
```

**Use Cases**:
- Getting method signatures
- Understanding class inheritance
- Finding available properties
- Learning class-specific patterns
- Accessing constructor parameters

**Example Usage**:
```python
result = await get_ui_class_detail("Button")
result = await get_ui_class_detail("omni.ui.scene.Line")
```

---

### 2.2 get_ui_module_detail

**Purpose**: Retrieves detailed information about a specific OmniUI module.

**Parameters**:
- `module_name` (str): Name of the module to look up (can be partial or full name)

**Return Type**: `Dict[str, Any]`
```python
{
    "success": bool,
    "result": str,  # JSON string containing detailed module information
    "error": Optional[str]
}
```

**Result Structure** (when successful):
```json
{
    "full_name": "omni.ui.scene",
    "classes": ["Line", "Rectangle", "Arc", ...],
    "functions": [...],
    "description": "3D scene UI components",
    "submodules": [...]
}
```

**Use Cases**:
- Understanding module contents
- Finding related classes
- Exploring module hierarchy
- Discovering utility functions

**Example Usage**:
```python
result = await get_ui_module_detail("omni.ui.scene")
```

---

### 2.3 get_ui_method_detail

**Purpose**: Retrieves detailed information about a specific method from any OmniUI class.

**Parameters**:
- `method_name` (str): Name of the method to look up (can be partial or full name)

**Return Type**: `Dict[str, Any]`
```python
{
    "success": bool,
    "result": str,  # JSON string containing detailed method information
    "error": Optional[str]
}
```

**Result Structure** (when successful):
```json
{
    "full_name": "Button.set_clicked_fn",
    "class_name": "Button",
    "method_name": "set_clicked_fn",
    "parameters": [
        {
            "name": "fn",
            "type": "Callable",
            "description": "Callback function"
        }
    ],
    "return_type": "None",
    "description": "Sets the click callback function",
    "examples": [...]
}
```

**Use Cases**:
- Understanding method signatures
- Finding parameter types
- Learning callback patterns
- Getting usage examples

**Example Usage**:
```python
result = await get_ui_method_detail("set_clicked_fn")
```

---

## 3. Instruction Tools

### 3.1 get_ui_instructions

**Purpose**: Retrieves comprehensive system-level documentation for OmniUI.

**Parameters**:
- `name` (str): The name of the instruction set to retrieve

**Valid Values**:
- `"agent_system"` - Core system prompt and framework basics
- `"classes"` - Class API reference and model patterns
- `"omni_ui_scene_system"` - 3D UI system documentation
- `"omni_ui_system"` - Core widgets, layouts and styling

**Return Type**: `Dict[str, Any]`
```python
{
    "success": bool,
    "result": str,  # Formatted markdown documentation
    "metadata": {
        "name": str,
        "description": str,
        "use_cases": List[str],
        "filename": str,
        "content_length": int,
        "line_count": int
    }
}
```

**Instruction Set Details**:

#### agent_system
- Core Omniverse UI Assistant system prompt
- Framework fundamentals
- Widget filters and options menus
- Searchable comboboxes
- General code writing guidelines

#### classes
- Comprehensive API reference
- AbstractValueModel patterns
- SimpleStringModel, SimpleBoolModel, SimpleFloatModel, SimpleIntModel
- Custom model implementations
- Model-view patterns

#### omni_ui_scene_system
- 3D shapes (Line, Curve, Rectangle, Arc)
- SceneView and camera controls
- Transform containers and matrices
- Gestures and mouse interactions
- Manipulators and custom 3D controls
- USD camera sync

#### omni_ui_system
- Basic UI shapes and widgets
- Labels, Buttons, Fields, Sliders
- HStack, VStack, ZStack, Grid layouts
- Window creation and management
- Styling with selectors
- Drag & drop functionality
- Model-Delegate-View pattern

**Example Usage**:
```python
result = await get_ui_instructions("agent_system")
result = await get_ui_instructions("omni_ui_scene_system")
```

---

### 3.2 get_ui_class_instructions

**Purpose**: Retrieves detailed documentation for specific OmniUI classes organized by category.

**Parameters**:
- `class_name` (str): The name of the class to retrieve

**Accepted Formats**:
- Simple name: `"Button"`, `"Label"`, `"TreeView"`
- Scene class: `"scene.Line"`, `"scene.Rectangle"`
- Full name: `"omni.ui.Button"`, `"omni.ui.scene.Line"`

**Return Type**: `Dict[str, Any]`
```python
{
    "success": bool,
    "result": str,  # Complete class documentation in markdown
    "metadata": {
        "class_name": str,
        "normalized_name": str,
        "category": str,
        "category_description": str,
        "file_path": str,
        "content_length": int,
        "line_count": int
    }
}
```

**Class Categories**:

1. **models** - Data models and delegates
   - AbstractValueModel, AbstractItemModel, AbstractItemDelegate

2. **shapes** - Basic shapes and geometric primitives
   - Rectangle, Circle, Ellipse, Triangle, Line, BezierCurve

3. **widgets** - Interactive UI widgets and controls
   - Button, RadioButton, CheckBox, ComboBox, Label, TreeView

4. **containers** - Layout containers and frames
   - Frame, CanvasFrame, ScrollingFrame, HStack, VStack, ZStack

5. **layouts** - Layout management and positioning
   - VGrid, HGrid, Placer

6. **inputs** - Input controls and field widgets
   - FloatSlider, IntSlider, FloatDrag, IntDrag, MultiFloatField

7. **windows** - Windows, dialogs, and menus
   - Window, MainWindow, Menu, MenuBar, Tooltip

8. **scene** - 3D scene UI components
   - Line, Curve, Rectangle, Arc, Image, Points, PolygonMesh

9. **units** - Unit and measurement types
   - Pixel, Percent, Fraction

10. **system** - System and styling components
    - Style

**Example Usage**:
```python
result = await get_ui_class_instructions("Button")
result = await get_ui_class_instructions("scene.Line")
result = await get_ui_class_instructions("omni.ui.TreeView")
```

---

## 4. Code Example Tools

### 4.1 get_omni_ui_code_example

**Purpose**: Retrieves relevant OmniUI code examples using semantic vector search and optional reranking.

**Parameters**:
- `request` (str): Query describing the desired OmniUI code example

**Configuration Parameters** (set in config.yaml):
- `rerank_k` (int): Number of documents to keep after reranking (default: 10)
- `enable_rerank` (bool): Whether to enable reranking (default: true)
- `embedding_model` (str): Embedding model to use (default: "nvidia/nv-embedqa-e5-v5")
- `embedding_endpoint` (str): Embedding service endpoint
- `embedding_api_key` (str): API key for embedding service
- `reranking_model` (str): Reranking model (default: "nvidia/llama-3.2-nv-rerankqa-1b-v2")
- `reranking_endpoint` (str): Reranking service endpoint
- `reranking_api_key` (str): API key for reranking service

**Return Type**: `Dict[str, Any]`
```python
{
    "success": bool,
    "result": str,  # Formatted code examples with metadata
    "error": Optional[str]
}
```

**Result Structure** (when successful):
```
File: /path/to/example.py
Method: build_search_ui

```python
def build_search_ui():
    with ui.VStack():
        search_field = ui.SearchField(
            placeholder="Search...",
            on_change_fn=handle_search
        )
        # ... more code
```

**How It Works**:
1. Converts query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
2. Performs semantic similarity search against pre-indexed code examples
3. Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2
4. Returns formatted code examples with metadata

**Query Matching Against**:
- SearchField, SearchWordButton implementations
- Widget styling and theming functions
- UI component building patterns
- Event handling and callback patterns
- Layout utilities (ZStack, VStack, HStack)
- Custom widget implementations

**Tips for Better Results**:
- Use specific OmniUI terminology (e.g., "SearchField", "ZStack")
- Include UI operations (e.g., "build_ui", "style", "event handling")
- Reference widget types (e.g., "Button", "Label", "Rectangle")
- Ask about patterns (e.g., "callback", "subscription", "model binding")

**Example Usage**:
```python
result = await get_omni_ui_code_example("How to create a search field?")
result = await get_omni_ui_code_example("Button styling with themes")
result = await get_omni_ui_code_example("VStack and HStack layout")
result = await get_omni_ui_code_example("event handling callbacks")
```

---

## Error Handling

All tools follow a consistent error handling pattern:

**Success Response**:
```python
{
    "success": True,
    "result": "...",  # Tool-specific result
    "error": None
}
```

**Error Response**:
```python
{
    "success": False,
    "result": "",
    "error": "Descriptive error message"
}
```

**Common Error Scenarios**:
- Atlas data not available
- Class/module/method not found
- Invalid instruction name
- FAISS index not found (for code examples)
- API key configuration issues

---

## Performance Characteristics

### Response Times (Typical)
- **list_ui_classes**: ~100ms (cached after first call)
- **list_ui_modules**: ~100ms (cached after first call)
- **get_ui_class_detail**: ~200ms (depends on class complexity)
- **get_ui_module_detail**: ~200ms
- **get_ui_method_detail**: ~150ms
- **get_ui_instructions**: ~50ms (file read)
- **get_ui_class_instructions**: ~50ms (file read)
- **get_omni_ui_code_example**: ~500-2000ms (vector search + reranking)

### Resource Usage
- **Memory**: ~500MB for Atlas data and FAISS index
- **CPU**: Minimal except during vector search
- **Network**: API calls for embeddings and reranking (code examples only)

---

## Integration Patterns

### Sequential Discovery Pattern
```python
# 1. Discover available classes
classes = await list_ui_classes()

# 2. Get details for specific class
details = await get_ui_class_detail("Button")

# 3. Get comprehensive documentation
docs = await get_ui_class_instructions("Button")

# 4. Find code examples
examples = await get_omni_ui_code_example("Button with custom styling")
```

### Documentation-First Pattern
```python
# 1. Get system documentation
system_docs = await get_ui_instructions("omni_ui_system")

# 2. Get specific class docs
button_docs = await get_ui_class_instructions("Button")

# 3. Find working examples
examples = await get_omni_ui_code_example("Button implementation")
```

### Search-First Pattern
```python
# 1. Search for code examples
examples = await get_omni_ui_code_example("create custom widget")

# 2. Get details about classes found in examples
for class_name in extract_classes(examples):
    details = await get_ui_class_detail(class_name)
```

---

## Best Practices

1. **Use Discovery Tools First**: Start with `list_ui_classes()` or `list_ui_modules()` to understand what's available

2. **Leverage Caching**: The Atlas service caches data after first load, so repeated calls are fast

3. **Be Specific with Queries**: For code examples, use specific OmniUI terminology for better results

4. **Combine Tools**: Use multiple tools together for comprehensive understanding

5. **Handle Errors Gracefully**: Always check the `success` field before using results

6. **Use Appropriate Detail Level**:
   - Quick reference: `get_ui_class_detail()`
   - Comprehensive docs: `get_ui_class_instructions()`
   - Working code: `get_omni_ui_code_example()`

7. **Optimize API Calls**: The code example tool makes external API calls; batch queries when possible

---

## Version Information

- **MCP Server Version**: 0.4.5
- **AIQ Toolkit Version**: 1.1.0
- **Embedding Model**: nvidia/nv-embedqa-e5-v5
- **Reranking Model**: nvidia/llama-3.2-nv-rerankqa-1b-v2
- **Atlas Data Version**: Latest from omni.ui framework