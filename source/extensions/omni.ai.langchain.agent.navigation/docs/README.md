# Navigation Agent Extension

## Introduction

The Navigation Agent extension (`omni.ai.langchain.agent.navigation`) provides scene navigation capabilities for Chat USD, enabling users to interact with their USD scenes using natural language for camera control and viewpoint management.

## Table of Contents

1. [Introduction](introduction.md)
2. [Architecture](architecture.md)
3. [Usage Guide](usage_guide.md)
4. [API Reference](api_reference.md)
7. [Developer Guide](extending_chat_usd_with_custom_agents.md)

## Extension Purpose

The "ChatUSD with navigation" agent is a perfect example of how to extend Chat USD with custom functionality. It:

- **Preserves all base Chat USD capabilities**:
  - USD code generation and modification
  - Scene information queries
  - Asset search functionality
  - All other standard Chat USD features

- **Adds navigation-specific features**:
  - Camera positioning and movement
  - Viewpoint saving and recall
  - Scene exploration tools
  - Points of interest management

This extension demonstrates how developers can create their own specialized Chat USD extensions while maintaining compatibility with the core system. The Navigation Agent serves as both a useful tool and a reference implementation for creating custom agents.

## Quick Start

To use the Navigation Agent:

1. Enable the extension in Omniverse Kit
2. Select "ChatUSD with navigation" in the Chat USD panel
3. Try these commands:
   - `List all points of interest` - Shows all viewpoints
   - `Navigate to [point]` - Moves to a saved location
   - `Save this view as [name]` - Saves current camera position

## Custom USDCode Functions

The Navigation Agent extension adds custom functions to USDCode to enhance USD scene manipulation capabilities:

This extension demonstrates how to extend USDCode's functionality by adding custom helper functions. This is done using the `add_functions_from_file` method:

```python
import usdcode
import os

helper_functions_path = os.path.join(os.path.dirname(__file__), "helpers", "helper_functions.py")
usdcode.setup.add_functions_from_file(helper_functions_path)
```

**Purpose:**
- Adds all functions from a specified Python file to USDCode's namespace
- Makes custom helper functions available directly through the `usdcode` module
- Enables users to access extension-specific functionality through USDCode

**Example:**
In this extension, we add the `create_camera_from_view` function, which creates a new camera at a specified path using the current camera's view and parameters. Users can then call this function directly through USDCode:

```python
# Create a camera at default path
new_camera_path = usdcode.create_camera_from_view(stage)
```

This mechanism allows extension developers to seamlessly integrate their custom functionality with USDCode, enhancing the capabilities available to users in scripts and through the Chat USD interface.

## Cursor Rules for Extension Development

There is a `.cursor` folder in the docs folder containing Cursor rules. If you copy this folder to the root of your project, Cursor will have access to the full documentation about LC-Agent and Chat USD, enabling it to assist you in creating an extension with a custom agent.

These rules provide Cursor with comprehensive knowledge about:
- The LC-Agent architecture
- Chat USD's core functionality
- How to extend Chat USD with custom agents
- Best practices for agent development

This makes it significantly easier to develop new extensions and agents with AI assistance.
