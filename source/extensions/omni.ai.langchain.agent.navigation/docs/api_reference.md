# API Reference

This document provides a reference for the Navigation Agent extension's key classes and methods.

## Extension Design

The Navigation Agent API is designed to extend Chat USD functionality while preserving all core capabilities. This extension demonstrates how to:

- Create specialized nodes that integrate with the Chat USD framework
- Implement custom modifiers that intercept and process domain-specific commands
- Register new components that work alongside existing Chat USD systems
- Use system messages to define specialized capabilities

The API follows a consistent pattern that can be applied to create other domain-specific extensions beyond navigation.

## Extension

### NavigationExtension

```python
class NavigationExtension(omni.ext.IExt):
    """Extension for Scene Navigation Agent."""
```

#### Methods

| Method | Description |
|--------|-------------|
| `on_startup(ext_id)` | Registers all navigation agent components |
| `on_shutdown()` | Unregisters all navigation agent components |

## Nodes

### NavigationGenNode

```python
class NavigationGenNode(RunnableNode):
    """Node responsible for scene navigation operations."""
```

Generates navigation commands based on user input using a specialized system message.

### NavigationNetworkNode

```python
class NavigationNetworkNode(NetworkNode):
    """Tool for scene navigation in USD scenes."""
```

#### Key Properties

- `default_node`: "NavigationGenNode"
- `metadata`: Contains description and examples for the node

#### Methods

- `__init__(**kwargs)`: Initializes the node and adds the NavigationModifier

### ChatUSDNavigationSupervisorNode

```python
class ChatUSDNavigationSupervisorNode(ChatUSDSupervisorNode):
    """Supervisor node that combines USD capabilities with scene navigation."""
```

Extends the standard ChatUSDSupervisorNode with navigation-specific capabilities through the navigation supervisor identity message.

### ChatUSDNavigationNetworkNode

```python
class ChatUSDNavigationNetworkNode(ChatUSDNetworkNode):
    """Specialized network node that extends ChatUSDNetworkNode with navigation."""
```

#### Key Properties

- `default_node`: "ChatUSDNavigationSupervisorNode"
- `route_nodes`: List including "ChatUSD_Navigation"
- `function_calling`: False
- `generate_prompt_per_agent`: True
- `multishot`: True

## Modifiers

### NavigationModifier

```python
class NavigationModifier(NetworkModifier):
    """Modifier that handles navigation commands from the NavigationGenNode."""
```

#### Command Patterns

- `LIST_COMMAND`: "LIST"
- `NAVIGATE_PATTERN`: r"NAVIGATE\s+(.+)"
- `SAVE_PATTERN`: r"SAVE\s+(.+)"
- `DONE_COMMAND`: "DONE"

#### Methods

| Method | Description |
|--------|-------------|
| `on_post_invoke_async(network, node)` | Processes navigation commands |
| `process_command(command)` | Routes commands to appropriate handlers |
| `_handle_list_command()` | Lists all points of interest |
| `_handle_navigate_command(poi_name)` | Navigates to a specific point |
| `_handle_save_command(poi_name)` | Saves current camera position |

## System Messages

### NAVIGATION_GEN_SYSTEM

Located at `nodes/systems/navigation_gen_system.md`.

- Defines the Navigation Agent's role
- Specifies command formats (LIST, NAVIGATE, SAVE)
- Provides guidance on command usage

### NAVIGATION_SUPERVISOR_IDENTITY

Located at `nodes/systems/chat_usd_navigation_supervisor_identity.md`.

- Describes the Navigation Agent's capabilities to the supervisor
- Provides routing guidelines

## Utility Functions

### Stage Utilities

| Function | Description |
|----------|-------------|
| `get_current_stage()` | Gets the current USD stage |
| `get_stage_metadata(stage, key)` | Retrieves metadata from the stage |
| `set_stage_metadata(stage, key, value)` | Sets metadata on the stage |

### Camera Utilities

| Function | Description |
|----------|-------------|
| `get_current_camera_transform()` | Gets the current camera transform |
| `set_camera_transform(transform)` | Sets the camera transform |
| `save_camera_poi(name)` | Saves a point of interest |
| `get_camera_poi(name)` | Gets a saved camera position |
| `list_camera_pois()` | Lists all saved camera positions |

## Extension Points

### Custom Commands

Extend `NavigationModifier`:
1. Add new command patterns as class variables
2. Extend the `process_command` method
3. Implement handler methods

### Custom System Messages

Customize behavior by modifying:
1. `navigation_gen_system.md`: Change command formats
2. `chat_usd_navigation_supervisor_identity.md`: Change routing behavior

## Next Steps

- [Developer Guide](extending_chat_usd_with_custom_agents.md)