# LC Agent Overview

Important: LC Agent is not a framework. It's written on top of langchain.

LC Agent provides a modular system for building complex language model interactions. Here's a high-level overview of its core components:

## Core Components

### RunnableNode
The fundamental building block. A `RunnableNode` represents a single unit of processing that can:
- Handle message management (system messages, user inputs, model responses)
- Interact with language models
- Track execution state and metrics
- Connect with other nodes to form networks

### RunnableNetwork
A container and manager for `RunnableNode` instances that:
- Manages the execution flow between connected nodes
- Maintains network state and context
- Enables dynamic modification through modifiers
- Provides synchronous, asynchronous, and streaming execution options

### NetworkNode
A hybrid class that combines `RunnableNode` and `RunnableNetwork`, allowing it to:
- Function as both a single node and a self-contained subnetwork
- Integrate seamlessly with parent networks
- Inherit properties (like chat model settings) from parent networks
- Maintain its own execution context

### MultiAgentNetworkNode
A specialized `NetworkNode` that acts as a conversation coordinator by:
- Routing queries to appropriate specialized "agent" sub-networks
- Managing complex multi-domain problem solving
- Coordinating between different specialized nodes
- Synthesizing responses from multiple agents

### NetworkModifier
A middleware component that provides controlled ways to modify network behavior by:
- Offering hooks into different stages of network execution
- Enabling safe network structure modifications
- Managing state changes during execution
- Supporting dynamic network evolution

### NodeFactory
A centralized registry for node types that:
- Manages node registration and creation
- Handles configuration and default arguments
- Ensures type safety
- Provides debugging capabilities

### Profiling System
A comprehensive performance profiling system that:
- Tracks execution times across the entire network hierarchy
- Captures detailed timing for nodes, modifiers, chunks, and custom operations
- Provides hierarchical profiling data that reflects network structure
- Generates interactive HTML visualizations for performance analysis
- Integrates seamlessly without code modification

## Component Relationships

```
NodeFactory
    │
    ├── creates/manages ──► RunnableNode
    │                          │
    │                          ├── can be contained in ──► RunnableNetwork
    │                          │                               │
    │                          │                               ├── can be modified by ──► NetworkModifier
    │                          │                               │
    │                          │                               └── specialized by ──► NetworkNode
    │                          │                                                         │
    │                          │                                                         └── extended by ──► MultiAgentNetworkNode
    │                          │
    └── creates/manages ──────►┘
```

## Key Concepts

### Message Flow
1. Messages enter through nodes (typically `RunnableHumanNode`)
2. Flow through the network based on node connections
3. Get processed by language models or tools
4. Generate responses that continue through the network

### Execution Lifecycle
1. Network preparation and context setup
2. Node execution in topological order
3. Modifier application at various stages
4. Result collection and state updates

### Network Modification
1. Safe modifications through `NetworkModifier`
2. Automatic connection handling in `NetworkNode`
3. Dynamic routing in `MultiAgentNetworkNode`
4. Centralized management via `NodeFactory`

## Common Use Cases

### Simple Chatbot
```python
with RunnableNetwork(chat_model_name="gpt-4") as network:
    RunnableHumanNode("What is Python?")
    RunnableNode(inputs=[
        SystemMessage(content="You are a helpful assistant")
    ])
```

### Multi-Agent System
```python
class CustomMultiAgent(MultiAgentNetworkNode):
    route_nodes = [
        "CodeExpert",
        "DocumentationHelper",
        "ConceptExplainer"
    ]
```

### Network Modification
```python
class CustomModifier(NetworkModifier):
    def on_post_invoke(self, network, node):
        if node.invoked and "tool_call" in node.outputs:
            self._handle_tool_call(network, node)
```

### Performance Profiling
```python
# Enable profiling globally
from lc_agent.utils.profiling_utils import enable_profiling
enable_profiling()

# Or enable via environment variable
# export LC_AGENT_PROFILING=1

# Run your network normally - profiling is automatic
result = network.invoke({"query": "What is Python?"})

# Visualize results
from lc_agent.utils.profiling_html import create_profiling_html
create_profiling_html(network, "profiling_results.html")
```

For detailed information about each component, refer to their respective documentation pages.
