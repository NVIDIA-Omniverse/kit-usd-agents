# OmniUI MCP Server - API Reference

## Functions

### get_omni_ui_code_example

Retrieves relevant OmniUI code examples using semantic vector search and optional reranking.

#### Function Signature

```python
async def get_omni_ui_code_example(
    request: str,
    rerank_k: int = 10,
    enable_rerank: bool = True,
    embedding_config: Optional[Dict[str, Any]] = None,
    reranking_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

#### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `request` | `str` | Yes | - | Natural language query describing the desired OmniUI code example |
| `rerank_k` | `int` | No | 10 | Number of documents to keep after reranking |
| `enable_rerank` | `bool` | No | True | Whether to enable reranking of search results |
| `embedding_config` | `Dict` | No | None | Configuration for embedding service |
| `reranking_config` | `Dict` | No | None | Configuration for reranking service |

#### Embedding Configuration

```python
embedding_config = {
    'model': 'nvidia/nv-embedqa-e5-v5',     # Embedding model
    'endpoint': None,                        # Custom endpoint (None for NVIDIA API)
    'api_key': 'your_api_key'               # API key for embedding service
}
```

#### Reranking Configuration

```python
reranking_config = {
    'model': 'nvidia/llama-3.2-nv-rerankqa-1b-v2',  # Reranking model
    'endpoint': None,                                # Custom endpoint (None for NVIDIA API)
    'api_key': 'your_api_key'                       # API key for reranking service
}
```

#### Returns

```python
{
    "success": bool,      # True if operation succeeded
    "result": str,        # Formatted code examples or message
    "error": str | None   # Error message if operation failed
}
```

#### Example Usage

```python
import asyncio
from omni_ui_mcp.functions.get_omni_ui_code_example import get_omni_ui_code_example

async def search_examples():
    # Basic search
    result = await get_omni_ui_code_example("How to create a search field?")
    
    # Search with custom reranking
    result = await get_omni_ui_code_example(
        "Button styling",
        rerank_k=5,
        enable_rerank=True
    )
    
    # Search with custom endpoints
    result = await get_omni_ui_code_example(
        "VStack layout",
        embedding_config={
            'model': 'nvidia/nv-embedqa-e5-v5',
            'endpoint': 'http://localhost:8080',
            'api_key': 'custom_key'
        }
    )
    
    return result

# Run the async function
result = asyncio.run(search_examples())
```

---

### list_ui_classes

Returns a list of all OmniUI class names from the Atlas data.

#### Function Signature

```python
async def list_ui_classes() -> Dict[str, Any]
```

#### Returns

```python
{
    "success": bool,
    "result": str,  # JSON string containing class information
    "error": str | None
}
```

#### Result Structure

```json
{
    "class_full_names": ["FilterButton", "OptionsMenu", ...],
    "total_count": 150,
    "description": "OmniUI classes from Atlas data"
}
```

---

### list_ui_modules

Returns a list of all OmniUI modules from the Atlas data.

#### Function Signature

```python
async def list_ui_modules() -> Dict[str, Any]
```

#### Returns

```python
{
    "success": bool,
    "result": str,  # JSON string containing module information
    "error": str | None
}
```

---

### get_ui_class_detail

Gets detailed information about a specific OmniUI class.

#### Function Signature

```python
async def get_ui_class_detail(class_name: str) -> Dict[str, Any]
```

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `class_name` | `str` | Yes | Name of the class (supports fuzzy matching) |

#### Returns

Detailed class information including:
- Full class name
- Module name
- Docstring
- Parent classes
- Methods list
- Method details

---

### get_ui_module_detail

Gets detailed information about a specific OmniUI module.

#### Function Signature

```python
async def get_ui_module_detail(module_name: str) -> Dict[str, Any]
```

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `module_name` | `str` | Yes | Name of the module (supports fuzzy matching) |

#### Returns

Detailed module information including:
- Full module name
- Classes in the module
- Functions in the module
- Module path

---

### get_ui_method_detail

Gets detailed information about specific OmniUI methods.

#### Function Signature

```python
async def get_ui_method_detail(
    method_name: str,
    class_name: Optional[str] = None
) -> Dict[str, Any]
```

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `method_name` | `str` | Yes | Name of the method (supports fuzzy matching) |
| `class_name` | `str` | No | Optional class name to narrow search |

#### Returns

Detailed method information including:
- Method signature
- Docstring
- Parameters
- Return type
- Parent class

---

## MCP Protocol Interface

### Tool Registration

All functions are exposed through MCP with the following names:
- `get_omni_ui_code_example`
- `list_ui_classes`
- `list_ui_modules`
- `get_ui_class_detail`
- `get_ui_module_detail`
- `get_ui_method_detail`

### Calling Tools via MCP

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_omni_ui_code_example",
    "arguments": {
      "request": "How to create a search field?"
    }
  },
  "id": 1
}
```

### Listing Available Tools

```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "params": {},
  "id": 0
}
```

---

## Service Classes

### Retriever

Handles FAISS vector search operations.

```python
from omni_ui_mcp.services.retrieval import Retriever

retriever = Retriever(
    endpoint_url=None,      # Custom embedding endpoint
    api_key="your_key",     # API key
    load_path="path/to/index",  # FAISS index path
    top_k=20,               # Default number of results
    embedding_config={}     # Embedding configuration
)

# Search for documents
docs = retriever.search("search query", top_k=10)
```

### Reranker

Improves search result relevance.

```python
from omni_ui_mcp.services.reranking import Reranker

reranker = Reranker(
    endpoint_url="https://api.endpoint",
    api_key="your_key",
    model="nvidia/llama-3.2-nv-rerankqa-1b-v2"
)

# Rerank passages
results = reranker.rerank(
    query="search query",
    passages=["passage1", "passage2"],
    top_k=5
)
```

---

## Configuration Objects

### GetOmniUICodeExampleConfig

Pydantic configuration model for the function.

```python
from pydantic import BaseModel, Field

class GetOmniUICodeExampleConfig(BaseModel):
    name: str = "get_omni_ui_code_example"
    verbose: bool = False
    rerank_k: int = 10
    enable_rerank: bool = True
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = "${NVIDIA_API_KEY}"
    reranking_model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    reranking_endpoint: Optional[str] = None
    reranking_api_key: str = "${NVIDIA_API_KEY}"
```

---

## Error Codes and Messages

| Error | Description | Solution |
|-------|-------------|----------|
| `FAISS index not found` | Index files missing | Ensure index files exist in data directory |
| `Authorization failed` | Invalid API key | Check NVIDIA_API_KEY environment variable |
| `Retriever not initialized` | Failed to load index | Check index path and file permissions |
| `Invalid request parameters` | MCP parameter error | Ensure request is sent as `{"request": "query"}` |
| `No relevant examples found` | No matching results | Try different query terms |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NVIDIA_API_KEY` | Yes | - | API key for NVIDIA services |
| `MCP_PORT` | No | 9901 | Server port |
| `OMNI_UI_DISABLE_USAGE_LOGGING` | No | false | Disable usage analytics |

---

## Data Structures

### Code Example Format

```python
{
    "file_name": "widget.py",
    "file_path": "omni.kit.widget.search_delegate/omni/kit/widget/search_delegate/widget.py",
    "method_name": "omni.kit.widget.search_delegate.SearchField.build_ui",
    "source_code": "def build_ui(self):\n    ..."
}
```

### Response Format

```
### Example 1
File: widget.py
Path: omni.kit.widget.search_delegate/omni/kit/widget/search_delegate/widget.py
Method: omni.kit.widget.search_delegate.SearchField.build_ui

```python
def build_ui(self):
    self._container = ui.ZStack(**self._container_args)
    # ... code continues
```
"""

---

## Performance Considerations

- **Embedding Generation**: ~50-100ms per query
- **FAISS Search**: <10ms for 3,594 documents
- **Reranking**: ~100-200ms for 10 documents
- **Total Response Time**: ~200-400ms with reranking

---

## Limitations

- Maximum query length: 512 tokens
- Maximum response size: 30,000 characters
- Rerank_k maximum: 100
- Concurrent requests: Limited by server resources

---

*For more details, see the [full documentation](FEATURE_DOCUMENTATION.md).*