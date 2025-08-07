# Chat USD Architecture

This section provides a detailed overview of the Chat USD architecture, explaining how the different components work together to create a powerful USD development assistant.

## Table of Contents

- [Multi-Agent Architecture](./multi-agent-architecture.md) - Overview of the multi-agent system
- [Component Interactions](./component-interactions.md) - How components interact with each other
- [Message Flow](./message-flow.md) - How messages flow through the system
- [Extension Integration](./extension-integration.md) - How Chat USD integrates with Omniverse Kit

## Architecture Overview

Chat USD is built on a multi-agent architecture that routes user queries to specialized agents based on the query's intent. The system consists of several key components:

1. **ChatUSDNetworkNode**: The main entry point for user interactions
2. **ChatUSDSupervisorNode**: The orchestrator of the multi-agent system
3. **Specialized Agents**:
   - **USDCodeInteractiveNetworkNode**: For USD code generation and execution
   - **USDSearchNetworkNode**: For USD asset search
   - **SceneInfoNetworkNode**: For scene information retrieval

These components work together to provide a comprehensive USD development assistant that can handle a wide range of tasks, from answering knowledge-based questions to generating and executing USD code.

## System Diagram

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

## Key Architectural Principles

1. **Separation of Concerns**: Each component has a specific responsibility
2. **Message-Based Communication**: Components communicate through well-defined message interfaces
3. **Extensibility**: The architecture is designed to be easily extended with new capabilities
4. **Modularity**: Components can be replaced or modified without affecting the rest of the system
