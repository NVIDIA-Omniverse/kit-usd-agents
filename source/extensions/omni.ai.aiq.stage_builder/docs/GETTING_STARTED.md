# Stage Builder - Getting Started

## Overview

Stage Builder is a multi-agent system for intelligent USD stage creation in Omniverse Kit. It uses coordinated agents for design, layout, and asset placement in USD scenes. The extension leverages AIQ (NVIDIA Agent Intelligence Toolkit) to orchestrate multiple specialized agents:

- **Planning Agent**: Creates detailed execution plans for complex tasks
- **USD Code Agent**: Executes USD code for scene modification
- **Scene Info Agent**: Analyzes scene contents and object properties
- **USD Search Agent**: Searches for USD assets and components

For a detailed demonstration of Stage Builder's capabilities and architecture, see [PROMPTS.md](PROMPTS.md).

## Prerequisites

- Omniverse Kit 2023.1.0 or newer
- Access to NVIDIA APIs for USD Search and embeddings
- Git for version control

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git
cd kit-usd-agents
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3. Build the Project

```bash
build.bat -r
```

This will build the project in release mode. The build process will:
- Download required dependencies
- Compile extensions
- Prepare the runtime environment

### 4. Set Environment Variables

Before running, you need to set the API keys for USD Search and NVIDIA services:

```bash
set USDSEARCH_API_KEY=your_usd_search_api_key
set NVIDIA_API_KEY=your_nvidia_api_key
```

## Running Stage Builder

Execute the following command to launch Kit with Stage Builder enabled:

```bash
_build\windows-x86_64\release\omni.app.chat_usd.bat --enable omni.ai.aiq.stage_builder
```

## Using Stage Builder

### 1. Select the Agent and Model

Once Kit launches:
2. In the dropdown of the Chat USD window, select **"Stage Builder AIQ"**
3. Select the chat model: **"meta/llama-4-maverick-..."**

### 2. Example Usage

Type the following example command:

```
find 3 different chairs and import them to the scene, place in line, not at the same point, make sure their scale is in meters so their size is not very different
```

The system will:
1. **Planning**: Create a detailed plan for finding and placing chairs
2. **Search**: Use USD Search to find appropriate chair assets
3. **Import**: Import the selected chairs into the scene
4. **Placement**: Position chairs in a line with appropriate spacing
5. **Scale**: Ensure all chairs are properly scaled in meters

### 3. More Examples and Technical Details

For a comprehensive demonstration of Stage Builder's capabilities, including:
- Scattering objects on surfaces
- Importing and stacking warehouse assets
- Camera focus control
- Detailed explanation of the multi-agent architecture

See [PROMPTS.md](PROMPTS.md) for in-depth examples and technical implementation details.
