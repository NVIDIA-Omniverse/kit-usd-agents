# Stage Builder - Multi-Agent USD Scene Builder

An intelligent multi-agent system for USD stage creation in Omniverse Kit that coordinates specialized agents to design, layout, and build complex scenes.

## Overview

Stage Builder is the initial extension for a new blueprint that will use multiple USD agents to design and build stages in Kit. This project addresses key challenges in the current Chat USD system and creates a practical tool for USD stage creation through agent coordination.

## Goals

Stage Builder has two main goals:

1. **Demonstrate Omniverse Agentic Workflow** - Show how multiple specialized USD agents can work together to solve complex tasks through agent coordination, progressive planning, and intelligent tool use.

2. **Improve Developer Experience** - Make it easier for developers to understand and extend the USD Agent System by establishing clear patterns and improving observability.

## Features

- **Multi-Agent Coordination**: Orchestrates multiple specialized agents to work together on complex stage building tasks
- **Progressive Planning**: Generates high-level plans that are executed step-by-step with detailed instructions
- **USD Code Execution**: Execute USD code interactively to create and modify scenes
- **Scene Analysis**: Analyze existing scenes to gather information about objects and their properties
- **Asset Search**: Search for USD assets and components within the scene or asset library
- **Intelligent Placement**: Future capability for smart object placement with collision detection and style matching

## Architecture

Stage Builder uses a supervisor-based multi-agent architecture:

- **Supervisor Agent**: Coordinates and routes tasks to specialized agents
- **Planning Agent**: Creates and manages execution plans for complex tasks
- **USD Code Agent**: Executes USD code for scene modification
- **Scene Info Agent**: Analyzes scene contents and object properties
- **USD Search Agent**: Searches for USD assets and components

## Installation

This extension is automatically installed with Omniverse Kit.

## Usage

1. Open Omniverse Kit
2. Enable the extension through the Extension Manager
3. Open the Chat USD panel where Stage Builder functionality is integrated
4. Start building scenes using natural language

### Example Prompts

Try queries like:
- "Create a living room with furniture"
- "Build a conference room setup"
- "Design an outdoor scene with trees and benches"
- "Set up a kitchen with appliances"

## Requirements

- Omniverse Kit 2023.1.0 or newer
- Chat USD extension
- NVIDIA Agent Intelligence Toolkit (AIQ)
- Planning Agent extension
- USD Code Agent extension

## Future Enhancements

The Stage Builder blueprint will evolve to include:

- Spatial management with voxel grids for collision detection
- Hierarchical zone system for semantic space organization
- Contact point system for physically valid object placement
- Style matching and consistency enforcement
- Advanced placement rules and clearance requirements

## Documentation

- **[Getting Started Guide](GETTING_STARTED.md)** - Installation, setup, and basic usage
- **[Prompts and Examples](PROMPTS.md)** - Detailed demonstration of capabilities and technical architecture
- **[MultiAgent AIQ Guide](MULTIAGENT_AIQ_GUIDE.md)** - How LC-Agent works inside AIQ and creating multi-agent systems

## Development

Stage Builder is designed to be extended by developers. The modular architecture allows for:

- Adding new specialized agents
- Creating custom planning strategies
- Implementing domain-specific placement rules
- Extending the zone and contact point systems

For more information on extending Stage Builder, see the developer documentation.