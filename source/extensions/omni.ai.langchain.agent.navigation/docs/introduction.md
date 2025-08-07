# Introduction to Navigation Agent

The Navigation Agent extension (`@omni.ai.langchain.agent.navigation`) enhances Chat USD with powerful scene navigation capabilities, allowing users to interact with USD scenes using natural language.

## Overview

Navigation Agent enables intuitive camera control and viewpoint management in USD scenes. It leverages the Language Chain (LC) Agent framework to process natural language commands and translate them into camera operations.

"ChatUSD with navigation" maintains all the standard Chat USD functionality while adding navigation-specific capabilities. This means you can:
- Generate and modify USD code
- Query scene information and structure
- Search for assets and references
- Use all standard Chat USD features
- PLUS navigate, save viewpoints, and explore your scene

This extension serves as an example of how to extend Chat USD with custom functionality using the LC Agent framework. The navigation capabilities demonstrate a pattern that can be applied to create other specialized agents.

### Key Features

- **Natural Language Camera Control**: Navigate scenes using simple commands
- **Points of Interest Management**: Save, name, and revisit important viewpoints
- **Scene Exploration**: Easily discover and navigate complex USD scenes
- **Seamless Integration**: Works with other Chat USD capabilities

## Basic Concepts

### Points of Interest (POIs)

Points of Interest (POIs) are named camera positions saved within a USD scene, capturing:
- Camera position and orientation
- Additional metadata (optional)

### Navigation Commands

The Navigation Agent recognizes three primary command types:

1. **LIST**: Displays all available points of interest
   ```
   LIST
   ```

2. **NAVIGATE**: Moves the camera to a specific point of interest
   ```
   NAVIGATE kitchen
   ```

3. **SAVE**: Creates a new point of interest
   ```
   SAVE front_door
   ```

These commands are automatically generated from natural language input.

### Behind the Scenes

When a user asks to "Show me the kitchen," the following occurs:
1. Chat USD routes the request to the Navigation Agent
2. The agent generates a `NAVIGATE kitchen` command
3. A modifier intercepts and executes the command
4. The camera moves to the saved position
5. A confirmation message is sent back to the user

## Next Steps

- [Architecture and Components](architecture.md)
- [Usage Guide](usage_guide.md)