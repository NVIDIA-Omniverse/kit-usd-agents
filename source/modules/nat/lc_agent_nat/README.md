# LC Agent NAT

LC Agent plugin for NAT (NVIDIA AgentIQ Toolkit) integration.

This module provides utilities for integrating LC Agent with NAT workflows.

## Overview

The `lc_agent_nat` module bridges the gap between NAT (NVIDIA AgentIQ Toolkit) and LC Agent, enabling developers to leverage the power of LC Agent's network-based architecture within NAT workflows. It provides two main approaches for building AI agents:

1. **Regular NAT Functions** - Simple, stateless operations ideal for tools and basic integrations
2. **LC Agent Networks** - Complex, stateful agents with dynamic behavior and LLM integration

## Key Component: LCAgentFunction

`LCAgentFunction` is the core bridge between NAT and LC Agent. It:

- Wraps LC Agent's `NetworkNode` and `RunnableNode` components for use in NAT
- Manages the lifecycle of LC Agent networks within NAT workflows
- Handles message conversion between NAT and LC Agent formats
- Supports streaming responses and async execution
- Automatically registers and unregisters components with LC Agent's node factory

## Basic Usage

### Simple LC Agent Integration

```python
from lc_agent_nat import LCAgentFunction
from nat.cli.register_workflow import register_function

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
from lc_agent_nat import MultiAgentConfig

# In YAML config:
workflow:
  _type: MultiAgent
  tool_names:
    - agent1
    - agent2
    - tool1
```

## Documentation

For detailed guidance on when to use regular NAT functions versus LC Agent networks, see:
- [NAT vs LC Agent Guide](doc/aiq_vs_lc_agent_guide.md) - Comprehensive comparison and decision guide
