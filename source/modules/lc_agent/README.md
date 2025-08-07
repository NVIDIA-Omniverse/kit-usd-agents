# LC Agent

LC Agent is a modular system built on top of LangChain for creating complex language model interactions. It provides a flexible architecture for building AI applications through composable nodes and networks.

## Philosophy

Unlike traditional state machines or pre-defined workflow systems, LC Agent uses a **dynamic graph construction** approach:

- **Emergent Graphs**: The network starts empty and grows dynamically as execution proceeds
- **Immutable Nodes**: Each node represents a "baked" result - a snapshot of an LLM response, tool output, or user query
- **State as Structure**: The graph itself *is* the state - no separate mutable state object is passed between nodes
- **Reactive Flow**: NetworkModifiers inspect results and decide what nodes to create next, building the execution path on-the-fly

This creates a complete, chronological record of the entire thought process, making it ideal for complex, interactive workflows where the path cannot be predetermined.

## Key Components

- **RunnableNode** - The fundamental building block representing a single unit of processing
- **RunnableNetwork** - Container and manager for nodes, handling execution flow and state
- **NetworkNode** - Hybrid component that functions as both a node and a self-contained network
- **MultiAgentNetworkNode** - Specialized coordinator for routing queries to appropriate agent sub-networks
- **NetworkModifier** - Middleware for safely modifying network behavior during execution
- **NodeFactory** - Centralized registry for node types and their instantiation
- **Profiling System** - Comprehensive performance tracking with interactive HTML visualizations

## Quick Start

```python
from lc_agent import RunnableHumanNode
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode
from lc_agent import get_node_factory
import asyncio
from lc_agent_chat_models import register_all


async def main():
    register_all()

    get_node_factory().register(RunnableHumanNode)
    get_node_factory().register(RunnableNode)

    with RunnableNetwork(
        default_node="RunnableNode", chat_model_name="gpt-4"
    ) as network:
        RunnableHumanNode("Who are you?")

    print("Starting network...")
    async for c in network.astream():
        print(c.content, end="")
    print()

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

For detailed documentation on architecture, core concepts, and advanced usage, see the [`doc/`](doc/) directory:

- **[Overview](doc/overview.md)** - High-level overview and component relationships
- **[Core Concepts](doc/core_concepts/)** - Detailed documentation for each component
  - [RunnableNode](doc/core_concepts/runnable_node.md)
  - [RunnableNetwork](doc/core_concepts/runnable_network.md)  
  - [NetworkNode](doc/core_concepts/network_node.md)
  - [NetworkModifier](doc/core_concepts/network_modifier.md)
  - [NodeFactory](doc/core_concepts/node_factory.md)
- **[Multi-Agent Systems](doc/multi_agent/)** - Building complex multi-agent applications
- **[Profiling Guide](doc/profiling.md)** - Performance analysis and optimization
