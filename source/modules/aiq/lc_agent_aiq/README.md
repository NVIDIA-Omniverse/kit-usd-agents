# LC Agent AIQ

LC Agent plugin for AgentIQ integration.

This module provides utilities for integrating LC Agent with AgentIQ workflows.

## Overview

The `lc_agent_aiq` module bridges the gap between AIQ (AgentIQ) and LC Agent, enabling developers to leverage the power of LC Agent's network-based architecture within AIQ workflows. It provides two main approaches for building AI agents:

1. **Regular AIQ Functions** - Simple, stateless operations ideal for tools and basic integrations
2. **LC Agent Networks** - Complex, stateful agents with dynamic behavior and LLM integration

## Key Component: LCAgentFunction

`LCAgentFunction` is the core bridge between AIQ and LC Agent. It:

- Wraps LC Agent's `NetworkNode` and `RunnableNode` components for use in AIQ
- Manages the lifecycle of LC Agent networks within AIQ workflows
- Handles message conversion between AIQ and LC Agent formats
- Supports streaming responses and async execution
- Automatically registers and unregisters components with LC Agent's node factory

## Basic Usage

### Simple LC Agent Integration

```python
from lc_agent_aiq import LCAgentFunction
from aiq.cli.register_workflow import register_function

@register_function(config_type=MyAgentConfig)
async def my_agent_function(config: MyAgentConfig, builder: Builder):
    yield LCAgentFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=MyNetworkNode,
        lc_agent_node_gen_type=MyGeneratorNode,  # Optional
    )
```

### Multi-Agent Networks

The module also supports multi-agent configurations through `MultiAgentNetworkFunction`:

```python
from lc_agent_aiq import MultiAgentConfig

# In YAML config:
workflow:
  _type: MultiAgent
  tool_names:
    - agent1
    - agent2
    - tool1
```

## Documentation

For detailed guidance on when to use regular AIQ functions versus LC Agent networks, see:
- [AIQ vs LC Agent Guide](doc/aiq_vs_lc_agent_guide.md) - Comprehensive comparison and decision guide
