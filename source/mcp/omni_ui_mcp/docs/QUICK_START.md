# OmniUI MCP Server - Quick Start Guide

## Prerequisites
- Python 3.11+
- NVIDIA API key from [ngc.nvidia.com](https://ngc.nvidia.com)
- Poetry (for development)

## Installation

### Option 1: Using Poetry (Recommended for Development)

```bash
# Clone the repository
cd source/mcp/omni_ui_mcp

# Install dependencies
poetry install

# Set your API key
set NVIDIA_API_KEY=your_api_key_here  # Windows
export NVIDIA_API_KEY=your_api_key_here  # Linux/Mac

# Run the server
poetry run python -m omni_ui_mcp
```

### Option 2: Using Docker

```bash
# Build the Docker image
cd source/mcp/omni_ui_mcp
build-docker.bat  # Windows
./build-docker.sh  # Linux/Mac

# Run the container
docker run --rm -e NVIDIA_API_KEY=your_api_key -p 9901:9901 omni-ui-mcp:latest
```

## Quick Test

### Test the Code Example Search

```python
# test_example.py
import asyncio
import os

# Make sure NVIDIA_API_KEY is set
os.environ['NVIDIA_API_KEY'] = 'your_api_key_here'

import sys
sys.path.insert(0, 'src')

from omni_ui_mcp.functions.get_omni_ui_code_example import get_omni_ui_code_example

async def test():
    # Search for search field examples
    result = await get_omni_ui_code_example("How to create a search field?")
    
    if result["success"]:
        print("Found examples:")
        print(result["result"][:500])  # Print first 500 chars
    else:
        print(f"Error: {result['error']}")

asyncio.run(test())
```

## Common Queries

Try these example queries:

```python
# UI Components
"How to create a button with callback?"
"SearchField implementation"
"Checkbox with value changed handler"

# Layout
"VStack and HStack examples"
"Using spacers for layout"
"ZStack for overlapping elements"

# Styling
"Button styling with themes"
"Custom style overrides"
"Rectangle background color"

# Event Handling
"Click event handlers"
"Model subscription patterns"
"Callback functions in UI"
```

## Integration with Cursor IDE

1. Add to Cursor settings:
```json
{
  "mcpServers": {
    "omni-ui-mcp": {
      "type": "mcp",
      "url": "http://localhost:9901/mcp"
    }
  }
}
```

2. Use in agent mode:
- Open a new chat in agent mode
- Ask: "Show me OmniUI code examples for creating a search field"
- The assistant will use the `get_omni_ui_code_example` tool

## Verify Installation

Check that everything is working:

```bash
# 1. Check API key
python -c "import os; print('API key set:', bool(os.getenv('NVIDIA_API_KEY')))"

# 2. Check FAISS index
python -c "from pathlib import Path; p=Path('src/omni_ui_mcp/data/faiss_index_omni_ui'); print('FAISS index exists:', p.exists())"

# 3. Test server startup
poetry run python -m omni_ui_mcp
# Should see: "Starting MCP server on port 9901..."
```

## Next Steps

- Read the [full documentation](FEATURE_DOCUMENTATION.md)
- Explore the [API reference](API_REFERENCE.md)
- Check [troubleshooting guide](FEATURE_DOCUMENTATION.md#troubleshooting) if you encounter issues

## Support

For help or issues:
- Check the logs for debug information
- Ensure NVIDIA_API_KEY is correctly set
- Verify FAISS index files are present
- Contact the Omniverse GenAI team