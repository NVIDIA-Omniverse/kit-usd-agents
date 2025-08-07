# USDKnowledgeNetworkNode

## Overview

The `USDKnowledgeNetworkNode` is a specialized network node in the LC Agent USD module that provides knowledge-based assistance for Universal Scene Description (USD). It serves as a comprehensive information source for USD concepts, API usage, best practices, and general questions about USD functionality.

## Purpose

The primary purpose of the `USDKnowledgeNetworkNode` is to:

1. Answer factual questions about USD concepts and terminology
2. Provide explanations of USD API functions and classes
3. Offer guidance on USD best practices and workflows
4. Explain USD file formats and structure
5. Assist with understanding USD's role in 3D pipelines

This node is designed to be the knowledge foundation of the USD agent system, focusing on providing accurate information rather than generating or executing code.

## Implementation Details

### Class Definition

```python
class USDKnowledgeNetworkNode(NetworkNode):
    default_node: str = "USDKnowledgeNode"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_modifier(USDKnowledgeRagModifier())
```

The implementation is intentionally simple, leveraging the power of the LC Agent framework to handle most of the complexity. The class:

1. Extends `NetworkNode` from the core LC Agent framework
2. Sets `USDKnowledgeNode` as its default node type
3. Adds a `USDKnowledgeRagModifier` to enhance responses with retrieval-augmented generation

### Default Node

The `USDKnowledgeNode` serves as the default processing node for this network. It:

1. Processes queries using a system message to generate appropriate responses
2. Handles knowledge-based questions about USD
3. Provides factual information about USD concepts and API

### RAG Modifier

The `USDKnowledgeRagModifier` enhances the node's responses by:

1. Retrieving relevant information from a knowledge base of USD documentation
2. Injecting this information into the context before generating responses
3. Ensuring responses are grounded in accurate USD documentation
4. Using a specialized retriever designed for USD knowledge
5. Filtering and ranking retrieved information for relevance

This modifier is essential for providing accurate and comprehensive information about USD concepts and API.

## Knowledge Sources

The `USDKnowledgeNetworkNode` draws information from several sources through its RAG modifier:

1. **USD Documentation**: Official documentation from Pixar and other USD contributors
2. **Python API References**: Documentation of the USD Python API

This information is accessed through the RAG system, which retrieves relevant content based on the user's query.

## Integration with Other Components

The `USDKnowledgeNetworkNode` is designed to work seamlessly with other components of the LC Agent USD module:

1. It can be used as a standalone knowledge source
2. It can be integrated into a multi-agent system alongside code generation nodes
3. It can provide context for code generation and validation
