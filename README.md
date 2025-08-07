# Kit USD Agents

This repository contains Chat USD and its supporting framework for AI-assisted Universal Scene Description (USD) development in NVIDIA Omniverse Kit.

## What is Chat USD?

Chat USD is a specialized AI assistant that enables natural language interaction with USD scenes. Built on top of LangChain, Chat USD provides a multi-agent system for USD development workflows.

### Core Capabilities
- **USD Code Generation & Execution**: Generate and execute USD code from natural language descriptions
- **Asset Search**: Search for USD assets using natural language queries
- **Scene Information**: Analyze and retrieve information about USD scenes
- **Interactive Development**: Real-time scene modification through conversation
- **Extensibility**: Add custom agents like navigation, UI generation, and more

## Repository Structure

### Extensions
- `omni.ai.chat_usd.bundle` - The main Chat USD extension bundle
- `omni.ai.langchain.agent.usd_code` - USD code generation and execution agent
- `omni.ai.langchain.agent.navigation` - Example custom agent for scene navigation
- `omni.ai.langchain.widget.core` - UI components for AI-powered interfaces
- `omni.ai.langchain.core` - Bridge between LangChain and Omniverse
- `omni.ai.langchain.aiq` - NVIDIA NeMo Agent Toolkit platform integration
- `omni.ai.aiq.agent.chat_usd` - Chat USD integration with NVIDIA NeMo Agent Toolkit

### Modules
- `lc_agent` - Core LC Agent built on LangChain
- `agents/usd` - USD-specific agent implementations
- `data_generation/usdcode` - USD meta-functions for optimized operations
- `rags` - Retrieval-augmented generation components
- `aiq` - NVIDIA NeMo Agent Toolkit integration utilities

## Getting Started

1. Build: `build.bat -r`
2. Run: `_build\windows-x86_64\release\omni.app.chat_usd.bat`

## Documentation

For detailed documentation on Chat USD architecture, components, and how to extend it with custom agents, see the documentation in source/extensions/omni.ai.chat_usd.bundle/docs/README.md
