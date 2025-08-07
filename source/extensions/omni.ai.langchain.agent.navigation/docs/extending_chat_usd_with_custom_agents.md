# Extending Chat USD with Custom Agents

This guide explains how to extend Chat USD with custom agents, using the Navigation Agent (`@omni.ai.langchain.agent.navigation`) as a reference implementation.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Component Relationships](#component-relationships)
4. [Creating a Custom Agent](#creating-a-custom-agent)
   - [Extension Structure](#extension-structure)
   - [Node Implementation](#node-implementation)
   - [Modifier Implementation](#modifier-implementation)
   - [System Messages](#system-messages)
5. [Integration with Chat USD](#integration-with-chat-usd)
6. [Example: Navigation Agent](#example-navigation-agent)
7. [Best Practices](#best-practices)

## Introduction

Chat USD is a powerful AI assistant for Universal Scene Description (USD) development that can be extended with custom agents to add new capabilities. Custom agents allow Chat USD to perform specialized tasks, such as scene navigation, asset search, or custom operations on USD scenes.

This document explains how to create and integrate custom agents into the Chat USD framework, using the Navigation Agent as a reference implementation.

## Architecture Overview

Chat USD uses a modular architecture based on the Language Chain (LC) Agent framework. The key components for extending Chat USD with custom agents are:

1. **Extension**: Registers the agent with the system and manages its lifecycle.
2. **Nodes**: Implement the agent's functionality and define how it processes inputs and generates outputs.
3. **Modifiers**: Intercept and modify the behavior of nodes, allowing for custom processing of commands.
4. **System Messages**: Define the agent's capabilities, identity, and how it should respond to user queries.

## Component Relationships

Understanding the inheritance hierarchy and relationships between components is crucial for implementing a custom agent:

### Inheritance Hierarchy

1. **ChatUSDNavigationNetworkNode**: The main entry point for user interactions.
   - Derives from `ChatUSDNetworkNode` and `MultiAgentNetworkNode`, which handles routing between agents
   - Ultimately derives from `NetworkNode`, which is a container for subnodes
   - Does not interact with LLMs directly
   - Acts as a coordinator for the conversation between agents and supervisor
   - Registered with the user-friendly name "ChatUSD with navigation"

2. **ChatUSDNavigationSupervisorNode**: The orchestrator for agent interactions.
   - Derives from `ChatUSDSupervisorNode`
   - Interacts with LLMs to determine which agent to call based on the user query
   - Can transform user queries to match agent capabilities
   - Uses system messages to understand agent capabilities and make routing decisions
   - Combines the base ChatUSDSupervisorNode system message with additional navigation-specific instructions

3. **NavigationNetworkNode**: The navigation agent implementation.
   - Derives from `NetworkNode`
   - Does not interact with LLMs directly but contains nodes that do
   - Registered as "ChatUSD_Navigation" and referenced in the route_nodes list of ChatUSDNavigationNetworkNode
   - Provides its docstring to the supervisor to explain its capabilities
   - Contains modifiers that process command outputs

4. **NavigationGenNode**: The component that generates navigation commands.
   - Derives from `RunnableNode`
   - Directly interacts with LLMs to generate navigation commands
   - Uses a specific system message that instructs the LLM to output specific command formats
   - Its outputs are intercepted by NavigationModifier
   - Its outputs are not directly visible to the supervisor, only the final result is

### Modifier's Role in Command Execution

The NavigationModifier plays a critical role in the execution of navigation commands:

1. **Command Interception**: The modifier's `on_post_invoke_async` method is called after each node in the network is invoked. It carefully checks conditions to prevent infinite loops:
   ```python
   if (
       node.invoked
       and isinstance(node.outputs, AIMessage)
       and node.outputs.content
       and not network.get_children(node)
   ):
   ```

2. **Command Processing**: When NavigationGenNode generates a command like "LIST", "NAVIGATE", or "SAVE", the modifier intercepts it and executes the corresponding action.

3. **Result Injection**: After processing a command, the modifier creates a new `RunnableHumanNode` with the result:
   ```python
   with network:
       RunnableHumanNode(f"Assistant: {result}")
   ```

4. **Continued Execution**: Creating this new node continues invoking the network. The default modifier for NavigationNetworkNode automatically creates the default node (NavigationGenNode) after the RunnableHumanNode.

5. **Command Execution in USD Stage**: The modifier executes navigation commands directly in the current USD stage. For example:
   - LIST command retrieves points of interest from the stage metadata
   - NAVIGATE command sets the camera transform to a specific position
   - SAVE command stores the current camera position in the stage metadata

This execution chain is what enables the navigation commands to be processed and executed, allowing the agent to perform real actions in the USD scene. Without the modifier, the commands would be meaningless text that couldn't affect the scene.

The modifier is the bridge between the language model's text output and actual functionality in the application, making it one of the most critical components when extending Chat USD with custom capabilities.

## Creating a Custom Agent

### Extension Structure

A typical custom agent extension follows this structure:

```
omni.ai.langchain.agent.custom/
├── docs/                      # Documentation
├── examples/                  # Example usage
├── omni/
│   └── ai/
│       └── langchain/
│           └── agent/
│               └── custom/    # Your agent name
│                   ├── __init__.py
│                   ├── extension.py
│                   ├── nodes/
│                   │   ├── __init__.py
│                   │   ├── custom_node.py
│                   │   └── systems/
│                   │       └── custom_system.md
│                   └── modifiers/
│                       ├── __init__.py
│                       └── custom_modifier.py
└── premake5.lua              # Build configuration
```

### Node Implementation

Nodes are the core components that implement the agent's functionality. For a custom agent, you typically need to create:

1. **GenNode**: Handles the generation of responses based on user input.
2. **NetworkNode**: Provides the interface for the agent to be used in a network.
3. **SupervisorNode**: Extends the ChatUSDSupervisorNode to add your agent's capabilities.
4. **NetworkNode for Chat USD**: Extends the ChatUSDNetworkNode to integrate your agent.

Here's how the Navigation Agent implements these nodes:

```python
# 1. GenNode - Responsible for generating navigation commands
class NavigationGenNode(RunnableNode):
    """Node responsible for scene navigation operations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add system message that defines how to respond to navigation queries
        self.inputs.append(RunnableSystemAppend(system_message=NAVIGATION_GEN_SYSTEM))

# 2. NetworkNode - Provides the interface for the navigation agent
class NavigationNetworkNode(NetworkNode):
    """Tool for scene navigation in USD scenes."""

    # Specifies which node class to use for generation
    default_node: str = "NavigationGenNode"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add the modifier that will process navigation commands
        self.add_modifier(NavigationModifier())

        # Metadata for the node
        self.metadata["description"] = "Scene navigation operations for USD scenes."
        self.metadata["examples"] = [
            "List all points of interest",
            "Navigate to the kitchen",
            "Save this view as 'front entrance'",
        ]

# 3. SupervisorNode - Extends ChatUSDSupervisorNode with navigation capabilities
class ChatUSDNavigationSupervisorNode(ChatUSDSupervisorNode):
    """Supervisor node that combines USD capabilities with scene navigation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add system message that helps determine when to use navigation
        self.inputs.append(RunnableSystemAppend(system_message=NAVIGATION_SUPERVISOR_IDENTITY))

# 4. NetworkNode for Chat USD - Integrates navigation with Chat USD
class ChatUSDNavigationNetworkNode(ChatUSDNetworkNode):
    """Specialized network node that extends ChatUSDNetworkNode with navigation."""

    # Specifies which supervisor node to use
    default_node: str = "ChatUSDNavigationSupervisorNode"

    # Specifies which tools/agents are available
    route_nodes: List[str] = [
        "ChatUSD_USDCodeInteractive",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
        "ChatUSD_Navigation",  # This is the NavigationNetworkNode
    ]

    function_calling = False
    generate_prompt_per_agent = True
    multishot = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add navigation-specific examples
        self.metadata["examples"].append("Show me all the viewpoints in this scene")
```

### Modifier Implementation

Modifiers intercept and process the output of nodes, allowing for custom behavior. The most critical aspect of a modifier is the `on_post_invoke_async` method, which prevents infinite loops by carefully checking conditions before processing commands.

Here's the NavigationModifier implementation:

```python
class NavigationModifier(NetworkModifier):
    """Modifier that handles navigation commands from the NavigationGenNode."""

    # Command patterns
    LIST_COMMAND = "LIST"
    NAVIGATE_PATTERN = r"NAVIGATE\s+(.+)"
    SAVE_PATTERN = r"SAVE\s+(.+)"
    DONE_COMMAND = "DONE"

    async def on_post_invoke_async(self, network, node):
        """
        Post-invoke hook that processes navigation commands.

        This method is called for every node in the network after it's invoked.
        The critical if condition prevents infinite loops by ensuring:
        1. The node has been invoked (has output)
        2. The output is an AIMessage (contains content)
        3. The content is not empty
        4. The node has no children (is the end of the chain)

        Without these checks, the modifier would create new nodes in an infinite loop.
        """
        if (
            node.invoked
            and isinstance(node.outputs, AIMessage)
            and node.outputs.content
            and not network.get_children(node)
        ):
            # Extract the command from the node output
            command = node.outputs.content.strip()
            result = await self.process_command(command)
            if result:
                # Add a new node to the network with the result
                # This is safe because we've verified this is the end of the chain
                with network:
                    RunnableHumanNode(f"Assistant: {result}")

    async def process_command(self, command):
        """Process navigation commands and return appropriate responses."""
        # Process LIST command
        if command == self.LIST_COMMAND:
            return self._handle_list_command()

        # Process NAVIGATE command
        navigate_match = re.match(self.NAVIGATE_PATTERN, command)
        if navigate_match:
            poi_name = navigate_match.group(1).strip()
            return self._handle_navigate_command(poi_name)

        # Process SAVE command
        save_match = re.match(self.SAVE_PATTERN, command)
        if save_match:
            poi_name = save_match.group(1).strip()
            return await self._handle_save_command(poi_name)

        # If command is DONE, do nothing
        if command == self.DONE_COMMAND:
            return None

        # If command is not recognized, provide a helpful message
        return self._handle_unrecognized_command(command)
```

### System Messages

System messages define the agent's capabilities, identity, and how it should respond to user queries. These are typically stored as markdown files in the `nodes/systems/` directory.

The Navigation Agent uses two system messages:

1. **navigation_gen_system.md**: Instructs the NavigationGenNode how to respond to navigation queries with specific commands (LIST, NAVIGATE, SAVE, DONE).

2. **chat_usd_navigation_supervisor_identity.md**: Helps the supervisor node determine when to route queries to the navigation agent.

Example from the Navigation Agent's supervisor identity:

```markdown
# ChatUSD_Navigation Function

## Use Case: Scene Navigation

The ChatUSD_Navigation function enables natural language navigation within USD scenes.

## When to Call ChatUSD_Navigation:

Call this function when the user request involves:
1. Navigating to a specific location or viewpoint in the scene
2. Listing available points of interest in the scene
3. Saving the current camera position as a point of interest
4. Requesting information about scene navigation capabilities

## When NOT to Call ChatUSD_Navigation:

1. When the exact same navigation request was just processed in the previous turn
2. When a navigation operation has already been completed successfully
3. When the user is asking about the current view or location (use ChatUSD_SceneInfo instead)
```

## Integration with Chat USD

To integrate your custom agent with Chat USD, you need to register your nodes with the node factory in your extension's `on_startup` method:

```python
class NavigationExtension(omni.ext.IExt):
    """Extension for Scene Navigation Agent."""

    def on_startup(self, ext_id):
        """Called when the extension is started."""
        self._ext_id = ext_id
        self._register_navigation_agent()

    def on_shutdown(self):
        """Called when the extension is shut down."""
        self._unregister_navigation_agent()

    def _register_navigation_agent(self):
        """Register the navigation agent components."""
        # Register the GenNode (hidden from direct use)
        get_node_factory().register(NavigationGenNode, name="NavigationGenNode", hidden=True)

        # Register the NetworkNode (available as ChatUSD_Navigation)
        get_node_factory().register(NavigationNetworkNode, name="ChatUSD_Navigation", hidden=True)

        # Register the SupervisorNode (hidden from direct use)
        get_node_factory().register(
            ChatUSDNavigationSupervisorNode, name="ChatUSDNavigationSupervisorNode", hidden=True
        )

        # Register the ChatUSD NetworkNode (available as "ChatUSD with navigation")
        get_node_factory().register(ChatUSDNavigationNetworkNode, name="ChatUSD with navigation")

    def _unregister_navigation_agent(self):
        """Unregister the navigation agent components."""
        get_node_factory().unregister(NavigationGenNode)
        get_node_factory().unregister(NavigationNetworkNode)
        get_node_factory().unregister(ChatUSDNavigationSupervisorNode)
        get_node_factory().unregister(ChatUSDNavigationNetworkNode)
```

Note that:
- The GenNode, NetworkNode, and SupervisorNode are registered with `hidden=True` because they are not meant to be used directly by users.
- The ChatUSDNavigationNetworkNode is registered with a user-friendly name ("ChatUSD with navigation") because it's the entry point for users.

## Example: Navigation Agent

The Navigation Agent (`omni.ai.langchain.agent.navigation`) is a reference implementation that demonstrates how to extend Chat USD with custom capabilities. It provides the following features:

- Listing points of interest (POIs) in a USD scene
- Navigating to specific POIs
- Saving camera positions as new POIs

## Best Practices

When creating custom agents for Chat USD, follow these best practices:

1. **Prevent infinite loops**: Carefully implement the `on_post_invoke_async` method in your modifier to check all necessary conditions before processing commands.

2. **Clear component relationships**: Understand how your nodes relate to each other and to the Chat USD framework.

3. **Descriptive system messages**: Clearly define when your agent should be used and what it can do.

4. **Error handling**: Provide helpful error messages when commands cannot be executed.

5. **Documentation**: Document your agent's capabilities and provide examples.

By following this guide, you can create custom agents that extend Chat USD with new capabilities, enhancing its usefulness for USD development and scene manipulation.