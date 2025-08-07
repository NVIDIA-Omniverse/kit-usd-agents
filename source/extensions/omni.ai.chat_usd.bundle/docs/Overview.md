# Chat USD Overview

## Introduction

Chat USD is a specialized AI assistant designed to facilitate Universal Scene Description (USD) development through natural language interaction. Built on the LC Agent framework, Chat USD provides a multi-agent system that enables users to interact with USD scenes, generate code, search for assets, and obtain information about scene elements using conversational language.

## Core Capabilities

Chat USD offers several key capabilities:

1. **USD Code Generation**: Creates and executes USD code based on natural language descriptions
2. **USD Asset Search**: Searches for USD assets based on natural language queries
3. **Scene Information Retrieval**: Analyzes and provides information about the current USD scene
4. **Interactive Development**: Enables real-time modification of USD scenes through conversation
5. **UI Integration**: Creates interactive UI elements with omni.ui (in the omni.ui variant)

## Architecture Overview

Chat USD is built on a multi-agent architecture that routes user queries to specialized agents based on the query's intent:

```text
                  ┌─────────────────────┐
                  │                     │
                  │  ChatUSDNetworkNode │
                  │                     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │                     │
                  │ChatUSDSupervisorNode│
                  │                     │
                  └──────────┬──────────┘
                             │
                             ▼
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼─────────┐ ┌───────▼────────┐ ┌────────▼─────────┐
│                  │ │                │ │                  │
│USDCodeInteractive│ │   USDSearch    │ │    SceneInfo     │
│     NetworkNode  │ │  NetworkNode   │ │   NetworkNode    │
│                  │ │                │ │                  │
└──────────────────┘ └────────────────┘ └──────────────────┘
```

The system uses a supervisor node to analyze user queries and route them to the appropriate specialized agent. Each agent is optimized for a specific task and can communicate with other agents to provide comprehensive responses.

## Key Components

### ChatUSDNetworkNode

The main entry point for user interactions, responsible for:
- Analyzing user queries
- Routing to appropriate specialized agents
- Coordinating responses from multiple agents
- Presenting final results to the user

### ChatUSDSupervisorNode

The orchestrator of the multi-agent system, responsible for:
- Determining which specialized agent should handle a query
- Formulating appropriate sub-queries for each agent
- Integrating responses from multiple agents
- Ensuring coherent and complete final responses

### USDCodeInteractiveNetworkNode

Specialized agent for USD code generation and execution, responsible for:
- Generating USD code based on natural language descriptions
- Executing code to modify the USD scene
- Validating and fixing code issues
- Providing feedback on code execution

### USDSearchNetworkNode

Specialized agent for USD asset search, responsible for:
- Interpreting natural language search queries
- Searching for relevant USD assets
- Presenting search results with previews
- Facilitating asset import into the scene

### SceneInfoNetworkNode

Specialized agent for scene information retrieval, responsible for:
- Analyzing the current USD scene
- Extracting relevant information about scene elements
- Providing context for other agents
- Answering queries about scene structure and properties

## Integration with LC Agent Framework

Chat USD is built on the LC Agent framework, leveraging its core components:

1. **RunnableNode**: Basic processing units for handling messages
2. **NetworkNode**: Container for specialized functionality
3. **MultiAgentNetworkNode**: Coordinator for multiple specialized agents
4. **NetworkModifier**: Middleware for extending node functionality

This integration allows Chat USD to benefit from the framework's flexibility, modularity, and extensibility.

## Use Cases

Chat USD is designed to support various USD development workflows:

1. **Rapid Prototyping**: Quickly create and modify USD scenes using natural language
2. **Asset Discovery**: Find and import USD assets based on descriptions
3. **Scene Analysis**: Understand the structure and properties of USD scenes
4. **Code Generation**: Generate USD code for complex operations
5. **Learning Tool**: Explore USD capabilities through conversation
