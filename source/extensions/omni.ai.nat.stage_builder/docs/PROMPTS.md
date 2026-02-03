# Stage Builder - Prompts and Examples

## Overview

This document demonstrates Stage Builder's AI-powered scene automation capabilities through real examples. Stage Builder uses natural language prompts to create and execute USD (Universal Scene Description) scripts automatically, orchestrating specialized agents to build and modify 3D scenes.

## Demonstration

### Three Sequential Prompts

The following examples showcase intelligent scene manipulation through natural language:

1. **Scattering Objects on Surfaces**
   - **User Action**: Selects a barrel and a shelf in the viewport
   - **Prompt**: "randomly scatter 50 instances of selected barrel over a surface of the selected shelf"
   - **AI Response**:
     - Uses `usdcode.get_selection()` to identify selected objects
     - Calls `usdcode.scatter_prims()` function to create point instancer
     - Distributes 50 instances randomly on the shelf surface

   ![Scattering objects on surfaces](../data/screenshot01.png)

2. **Importing and Stacking Assets**
   - **Prompt**: "find and import 5 warehouse assets and stack them to the highest empty shelf in the scene. print the shelf path"
   - **AI Response**:
     - ChatUSD_USDSearch searches for warehouse assets
     - ChatUSD_SceneInfo finds shelves using `usdcode.list_prims_within_vertical_zone()` to check emptiness
     - ChatUSD_USDCodeInteractive imports assets and uses `usdcode.stack_objects()`
     - Prints the shelf path found

   ![Importing and stacking assets](../data/screenshot02.png)

3. **Camera Focus Control**
   - **Prompt**: "focus current camera on this shelf"
   - **AI Response**:
     - ChatUSD_SceneInfo to determine the name of the current camera
     - ChatUSD_USDCodeInteractive calls helper function to focus it on the shelf
     - Updates camera position and orientation

   ![Camera focus control](../data/screenshot03.png)

## How the System Works

### Architecture Overview

Stage Builder uses a **Multi-Agent Architecture** with specialized agents coordinated by a supervisor:

```
User Prompt
    ↓
Supervisor Agent (Orchestrator)
    ↓
┌─────────────────────────────────────────────────────┐
│  Planning Agent → Creates detailed execution plan   │
│  SceneInfo Agent → Gathers scene information        │
│  USDCodeInteractive Agent → Executes USD operations │
│  USDSearch Agent → Finds and imports assets         │
└─────────────────────────────────────────────────────┘
    ↓
Scene Modifications
```

### The Agents Explained

1. **Supervisor Agent**
   - Routes tasks to appropriate specialized agents
   - Manages conversation flow
   - Ensures proper execution sequence

2. **Planning Agent**
   - Creates comprehensive, step-by-step plans
   - Specifies exact object properties and positions
   - Handles spatial reasoning and collision avoidance

3. **ChatUSD_SceneInfo**
   - Analyzes current scene state
   - Retrieves object paths, positions, and properties
   - Provides bounding box information for spatial calculations
   - Essential for operations on existing scene elements

4. **ChatUSD_USDCodeInteractive**
   - Executes USD Python code
   - Implements scene modifications
   - Handles object creation, transformation, and material assignment
   - Uses specialized functions like `align_objects` and `stack_objects`

5. **ChatUSD_USDSearch**
   - Searches asset libraries (including SimReady content)
   - Filters by tags, categories, or keywords
   - Returns asset paths for importing

---

For installation and setup instructions, see the [Getting Started Guide](GETTING_STARTED.md).
