# Multi-Agent Architecture

The Chat USD system is built on a multi-agent architecture that enables it to handle a wide range of USD-related tasks by routing queries to specialized agents. This document explains the multi-agent architecture in detail, including its components, interactions, and benefits.

## Overview

The multi-agent architecture in Chat USD is implemented using the `MultiAgentNetworkNode` class from the LC Agent framework. This architecture allows the system to:

1. Analyze user queries to determine their intent
2. Route queries to specialized agents based on their intent
3. Coordinate responses from multiple agents
4. Present a unified response to the user

This approach enables each specialized agent to focus on a specific domain of expertise, resulting in more accurate and comprehensive responses.

## Key Components

### ChatUSDNetworkNode

The `ChatUSDNetworkNode` class is the main entry point for the multi-agent system. It extends `MultiAgentNetworkNode`.

### ChatUSDSupervisorNode

The `ChatUSDSupervisorNode` class is responsible for analyzing user queries and determining which specialized agent should handle them. It extends `RunnableNode` and is configured with a system message that guides its behavior.

The system message (`identity`) provides detailed instructions on how to analyze user queries and route them to the appropriate specialized agent.

### Specialized Agents

The multi-agent system includes several specialized agents, each focused on a specific domain:

1. **USDCodeInteractiveNetworkNode**: Handles USD code generation and execution
2. **USDSearchNetworkNode**: Handles USD asset search
3. **SceneInfoNetworkNode**: Handles scene information retrieval

Each agent is registered with the node factory and configured with specific parameters:

```python
get_node_factory().register(
    USDCodeInteractiveNetworkNode,
    name="ChatUSD_USDCodeInteractive",
    scene_info=False,
    enable_code_interpreter=enable_code_interpreter,
    code_interpreter_hide_items=code_interpreter_hide_items,
    enable_code_atlas=need_rags,
    enable_metafunctions=need_rags,
    enable_interpreter_undo_stack=True,
    max_retries=1,
    enable_code_promoting=True,
    hidden=True,
)
```

## Query Routing Process

The query routing process in Chat USD follows these steps:

1. **User Query Analysis**: The `ChatUSDSupervisorNode` analyzes the user's query to determine its intent
2. **Agent Selection**: Based on the intent, the supervisor selects the appropriate specialized agent
3. **Query Reformulation**: The supervisor reformulates the query for the selected agent
4. **Agent Execution**: The selected agent processes the reformulated query
5. **Response Integration**: The supervisor integrates the agent's response into a coherent reply
6. **Iterative Refinement**: If needed, the supervisor may route follow-up queries to other agents

This process ensures that each query is handled by the most appropriate agent, resulting in accurate and comprehensive responses.

## System Message Design

The system message for the `ChatUSDSupervisorNode` is a critical component of the multi-agent architecture. It provides detailed instructions on:

1. **Available Expert Functions**: Describes the capabilities of each specialized agent
2. **Function Calling Guidelines**: Explains how to call each function effectively
3. **Scene Operation Guidelines**: Provides guidance on scene-related operations
4. **Information Gathering Guidelines**: Explains how to gather scene information
5. **Code Integration Guidelines**: Describes how to integrate code from different agents

The system message is designed to ensure that the supervisor makes appropriate routing decisions and provides clear instructions to the specialized agents.

## Extending the Multi-Agent System

The multi-agent architecture in Chat USD can be extended in several ways:

1. **Adding New Agents**: New specialized agents can be added to handle additional domains
2. **Enhancing Routing Logic**: The routing logic can be improved to handle more complex queries
3. **Implementing Agent Collaboration**: Agents can be designed to collaborate on complex tasks

These extensions can enhance the capabilities of Chat USD and enable it to handle a wider range of USD development tasks.
