# Usage Guide

This guide explains how to use the Navigation Agent extension to navigate USD scenes and manage viewpoints.

## Getting Started

1. Open Omniverse Kit
2. Enable the extension (Window > Extensions > @omni.ai.langchain.agent.navigation)
3. Open the Chat USD panel
4. Select "ChatUSD with navigation" from the agent dropdown
5. Open a USD scene

## Basic Commands

### Listing Points of Interest

Ask using natural language:
- "List all points of interest"
- "Show me all available viewpoints"
- "What viewpoints are saved in this scene?"

### Navigating to a Point of Interest

- "Navigate to the kitchen"
- "Show me the living room"
- "Take me to the front door"

### Saving a New Point of Interest

- "Save this view as kitchen"
- "Save current position as living room"
- "Remember this viewpoint as front door"

## Working with Points of Interest

### Naming Conventions

- Use descriptive names identifying the location
- Avoid special characters or very long names
- Use underscores instead of spaces if needed (e.g., "front_door")
- Consider consistent naming for related points (e.g., "kitchen_1", "kitchen_2")

Points of interest are stored as USD stage metadata, which means they are saved with the scene and persist when reopening the file.

## Integration with Other Agents

The Navigation Agent works with other Chat USD capabilities:

### Full Chat USD Functionality

"ChatUSD with navigation" is a complete extension of Chat USD rather than a replacement. You can use all standard Chat USD features, including:

- USD code generation (e.g., "Create a sphere at the current position")
- Code modification (e.g., "Change the color of the selected object")
- Scene queries (e.g., "List all meshes in this scene")
- Material adjustments (e.g., "Make this object more reflective")
- Asset management (e.g., "Find textures for wood")

The navigation capabilities are seamlessly integrated, allowing you to switch contexts naturally without changing agents.

### Scene Information

- "Navigate to the kitchen and tell me what objects are there"
- "Show me the living room and list all the furniture"

### USD Code Generation

- "Navigate to the kitchen and generate code to add a new light"
- "Show me the living room and write code to change the material of the couch"

## Advanced Use Cases

### Creating Tours

1. Navigate to important locations
2. Save each with a descriptive name
3. Use a naming convention for order (e.g., "tour_1_entrance", "tour_2_living_room")

### Scene Exploration

For complex scenes:
1. List all points of interest
2. Navigate to specific areas
3. Save new points as you discover interesting perspectives

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Point not found | Check spelling or list all points |
| Camera doesn't move | Ensure stage is loaded properly |
| Cannot save points | Check if stage is read-only |

## Next Steps

- [Examples](examples.md)
- [API Reference](api_reference.md)
- [Advanced Topics](advanced_topics.md)