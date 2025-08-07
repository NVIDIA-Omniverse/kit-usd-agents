# Extending the Chat USD System

This document provides guidance on how to extend the Chat USD system to add new functionality, customize existing components, or integrate with other systems.

## Overview

The Chat USD system is designed to be extensible, allowing developers to add new functionality, customize existing components, or integrate with other systems. This extensibility is achieved through a modular architecture that separates concerns and provides clear extension points.

The main extension points in the Chat USD system include:

1. **Adding New Network Nodes**: Create specialized agents for specific tasks
2. **Adding New Modifiers**: Extend the functionality of network nodes
3. **Customizing System Messages**: Modify the behavior of existing nodes
4. **Integrating with Other Systems**: Connect the Chat USD system with other systems
5. **Customizing the UI**: Modify the user interface to suit specific needs

Each extension point is detailed in the following sections.

## Adding New Network Nodes

Network nodes are the building blocks of the Chat USD system. They are responsible for processing user queries and generating responses. To add a new network node, follow these steps:

1. **Create a New Node Class**: Create a new Python class that extends the `NetworkNode` class from the LC Agent framework.

```python
from lc_agent import NetworkNode

class MyCustomNetworkNode(NetworkNode):
    """My Custom Network Node"""

    default_node: str = "MyCustomNode"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add modifiers or other initialization code here
```

2. **Configure the Node**: Configure the node with appropriate parameters and modifiers.

```python
def __init__(self, custom_param=True, **kwargs):
    super().__init__(**kwargs)
    self.custom_param = custom_param

    # Add modifiers
    self.add_modifier(MyCustomModifier())
```

3. **Register the Node**: Register the node with the node factory in the extension's initialization code.

```python
def _register_nodes(self):
    """Register the nodes with the node factory"""
    # Get the node factory
    node_factory = get_node_factory()

    # Register existing nodes
    # ...

    # Register the custom node
    node_factory.register(
        MyCustomNetworkNode,
        name="ChatUSD_MyCustom",
        custom_param=True,
        hidden=True,
    )
```

4. **Update the Route Nodes**: Update the `route_nodes` dictionary in the `ChatUSDNetworkNode` class to include the new node.

```python
class ChatUSDNetworkNode(MultiAgentNetworkNode):
    """Chat USD Network Node"""

    default_node: str = "ChatUSDSupervisorNode"
    route_nodes: List[str] = [
        "ChatUSD_USDCodeInteractive",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
        "ChatUSD_MyCustom"
    ]

    # ...
```

5. **Update the System Message**: Update the system message of the `ChatUSDSupervisorNode` to include instructions for the new node.

```python
USD_SUPERVISOR_SYSTEM = """
You are a USD expert that can help users with USD-related tasks.

You have access to the following expert functions:

1. ChatUSD_USDCodeInteractive
   - Generates and executes USD code
   - ...

2. ChatUSD_USDSearch
   - Searches for USD assets
   - ...

3. ChatUSD_SceneInfo
   - Retrieves information about the current USD scene
   - ...

4. ChatUSD_MyCustom
   - Performs custom functionality
   - Use this function when the user asks for [specific task]
   - Required for:
     * [specific use case 1]
     * [specific use case 2]
   - Cannot [limitation]
"""
```

## Adding New Modifiers

Modifiers extend the functionality of network nodes by intercepting and modifying messages, executing code, processing results, and more. To add a new modifier, follow these steps:

1. **Create a New Modifier Class**: Create a new Python class that extends the `NetworkModifier` class from the LC Agent framework.

```python
from lc_agent import NetworkModifier

class MyCustomModifier(NetworkModifier):
    """My Custom Modifier"""

    def __init__(self, custom_param=True):
        self.custom_param = custom_param

    def on_pre_invoke(self, network, inputs):
        """Called before the network is invoked"""
        # Modify the node before they are processed by the network

    def on_post_invoke(self, network, result):
        """Called after the network is invoked"""
        # Modify the result after it is processed by the network
```

2. **Add the Modifier to a Node**: Add the modifier to a node in the node's initialization code.

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Add the custom modifier
    self.add_modifier(MyCustomModifier(custom_param=True))
```

## Customizing System Messages

System messages are used to provide instructions to the language model that powers the Chat USD system. They define the behavior of the system and guide the model in generating appropriate responses. To customize a system message, follow these steps:

1. **Identify the System Message**: Identify the system message you want to customize. System messages are typically defined as constants in the node's module.

```python
# In chat_usd_supervisor_node.py
USD_SUPERVISOR_SYSTEM = """
You are a USD expert that can help users with USD-related tasks.

You have access to the following expert functions:
...
"""
```

2. **Modify the System Message**: Modify the system message to suit your needs. Be careful to maintain the overall structure and intent of the message.

```python
# Modified system message
USD_SUPERVISOR_SYSTEM = """
You are a USD expert that can help users with USD-related tasks.

You have access to the following expert functions:
...

Additional instructions:
- Prioritize [specific task] when the user asks for [specific condition]
- Always include [specific information] in your responses
- Avoid [specific behavior]
"""
```
