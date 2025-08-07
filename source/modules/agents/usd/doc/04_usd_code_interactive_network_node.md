# USDCodeInteractiveNetworkNode

## Overview

The `USDCodeInteractiveNetworkNode` is a specialized network node in the LC Agent USD module that provides real-time interactive code development capabilities for Universal Scene Description (USD). It enables users to modify USD stages in real-time and work with USD assets in an interactive manner, offering a more dynamic and responsive development experience compared to the standard code generation nodes.

## Purpose

The primary purpose of the `USDCodeInteractiveNetworkNode` is to:

1. Enable real-time modification of USD stages
2. Provide interactive code development with immediate feedback
3. Support importing and working with USD assets found through other tools
4. Offer metafunction capabilities for common USD operations
5. Enhance the development experience with specialized system messages and examples

This node is designed for users who need a more interactive and dynamic approach to USD development, allowing them to iteratively build and modify USD scenes with immediate feedback.

## Implementation Details

### Class Definition

```python
class USDCodeInteractiveNetworkNode(NetworkNode):
    """
    "USD Code Interactive" node. Use it to modify USD stage in real-time and import assets that was found with another tools.
    """

    default_node: str = "USDCodeInteractiveNode"
    code_interpreter_hide_items: Optional[List[str]] = None

    def __init__(
        self,
        enable_code_atlas=True,
        enable_metafunctions=True,
        retriever_name=None,
        usdcode_retriever_name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        rag_top_k = self.find_metadata("rag_top_k")
        rag_max_tokens = self.find_metadata("rag_max_tokens")

        if enable_code_atlas or retriever_name:
            self.add_modifier(
                USDCodeGenRagModifier(
                    code_atlas_for_human=enable_code_atlas,
                    code_atlas_for_errors=enable_code_atlas,
                    retriever_name=retriever_name,
                    top_k=rag_top_k,
                    max_tokens=rag_max_tokens,
                )
            )

        # Metafunctions
        if enable_metafunctions:
            from ..modifiers.mf_rag_modifier import MFRagModifier

            args = {}
            if usdcode_retriever_name:
                args["retriever_name"] = usdcode_retriever_name

            self.add_modifier(
                MFRagModifier(
                    top_k=rag_top_k,
                    max_tokens=rag_max_tokens,
                    **args,
                )
            )
```

The implementation:

1. Extends `NetworkNode` from the core LC Agent framework
2. Sets `USDCodeInteractiveNode` as its default node type
3. Provides configuration options for code atlas and metafunctions
4. Adds specialized modifiers for RAG and metafunction support

### Configuration Options

The `USDCodeInteractiveNetworkNode` accepts several configuration options:

1. **enable_code_atlas**: Whether to enable code atlas for human messages and errors
2. **enable_metafunctions**: Whether to enable metafunction support
3. **retriever_name**: Name of the retriever for general RAG functionality
4. **usdcode_retriever_name**: Name of the retriever for USD code-specific RAG

These options allow customization of the interactive experience based on specific requirements.

### Default Node

The `USDCodeInteractiveNode` serves as the default processing node for this network. It:

1. Extends `USDCodeGenNode` with interactive capabilities
2. Loads specialized system messages for interactive development
3. Supports configuration of default prim path, up axis, and selection
4. Provides access to metafunctions for common USD operations

## Metafunctions

One of the key features of the `USDCodeInteractiveNetworkNode` is its support for metafunctions. These are pre-defined functions that:

1. Provide high-level operations for common USD tasks
2. Simplify complex USD operations into easy-to-use functions
3. Enable rapid development without writing boilerplate code
4. Offer consistent interfaces for USD operations

The metafunctions are extracted from the `usdcode` module and made available to the node through the `MFRagModifier`.

## System Messages

The node uses specialized system messages to guide its behavior:

### Identity

The identity system message establishes the node as an interactive USD coding assistant that can modify USD stages in real-time.

### Code Structure

The code structure message provides guidance on how to structure USD code for interactive development, including:

1. Proper import statements
2. Stage creation and access
3. Prim manipulation
4. Material and shader setup
5. Animation and transformation

### Selection

The selection message provides information about the current selection in the USD stage, allowing the node to operate on specific prims.

### Examples

The examples message provides comprehensive examples of interactive USD code, demonstrating:

1. Creating and modifying prims
2. Setting up materials and shaders
3. Working with transformations
4. Creating animations
5. Using metafunctions for common tasks

## RAG Integration

The `USDCodeInteractiveNetworkNode` uses two types of RAG integration:

1. **Code Atlas RAG**: Enhances responses with general USD documentation and examples
2. **Metafunction RAG**: Provides information about available metafunctions and their usage

This dual RAG approach ensures that the node has access to both general USD knowledge and specific information about metafunctions.
