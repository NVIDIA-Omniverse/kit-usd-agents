# NodeFactory

## Overview

`NodeFactory` is a core component in the LC Agent framework that provides a centralized registry for node types and handles their instantiation. It implements the Factory pattern to manage the creation and registration of different types of nodes in a type-safe and maintainable way.

## Purpose

The NodeFactory solves several key problems in node management:

1. **Centralized Registration**: Provides a single point of registration for all node types
2. **Dynamic Node Creation**: Enables creation of nodes by name without direct class references
3. **Configuration Management**: Handles default arguments and configuration for node types
4. **Type Safety**: Ensures proper node type creation and validation
5. **Debug Support**: Provides debugging capabilities for node creation tracking

## Core Components

### Class Definition

```python
class NodeFactory:
    def __init__(self):
        self._registered_nodes = {}  # Registry of node types
```

### Registration Methods

```python
def register(self, node_type: Type, *args, **kwargs):
    """
    Registers a new node type with optional default arguments.
    
    Args:
        node_type (Type): The class of the node
        *args: Default positional arguments for node creation
        **kwargs: Default keyword arguments for node creation
            - name (str): Optional custom name for the node type
            - hidden (bool): Whether the node should be hidden from listings
    """

def unregister(self, node_type: Union[Type, str]):
    """
    Removes a node type from the registry.
    
    Args:
        node_type (Union[Type, str]): Node class or name to unregister
    """
```

### Node Creation

```python
def create_node(self, node_name: str, *args, **kwargs) -> "RunnableNode":
    """
    Creates a new instance of a registered node type.
    
    Args:
        node_name (str): Name of the node type to create
        *args: Additional arguments for node creation
        **kwargs: Additional keyword arguments for node creation
    
    Returns:
        RunnableNode: New instance of the requested node type
    """
```

## Usage Examples

### Basic Registration and Creation

```python
from lc_agent import get_node_factory

# Register a node type
get_node_factory().register(RunnableNode)

# Create an instance
node = get_node_factory().create_node("RunnableNode")
```

### Custom Named Registration

```python
# Register with custom name
get_node_factory().register(
    WeatherKnowledgeNode,
    name="WeatherKnowledge",
    hidden=True
)

# Create using custom name
weather_node = get_node_factory().create_node("WeatherKnowledge")
```

### Registration with Default Arguments

```python
# Register with default configuration
get_node_factory().register(
    RunnableNode,
    inputs=[SystemMessage(content="Default system message")],
    chat_model_name="gpt-4"
)

# Create with defaults
node = get_node_factory().create_node("RunnableNode")
```

### Multiple Node Registration

```python
# Register multiple node types
get_node_factory().register(RunnableNode)
get_node_factory().register(RunnableHumanNode)
get_node_factory().register(LocationSupervisorNode)
get_node_factory().register(
    WeatherKnowledgeNode,
    name="WeatherKnowledge"
)
```

## Node Type Management

### Type Checking and Validation

```python
# Check if node type is registered
if get_node_factory().has_registered("RunnableNode"):
    # Create node
    node = get_node_factory().create_node("RunnableNode")

# Get registered node type
node_type = get_node_factory().get_registered_node_type("RunnableNode")
```

### Node Type Listing

```python
# Get all registered node names
all_nodes = get_node_factory().get_registered_node_names()

# Get only visible node names
visible_nodes = get_node_factory().get_registered_node_names(hidden=False)

# Get base node names for a type
base_nodes = get_node_factory().get_base_node_names("CustomNode")
```

## Configuration Management

### Argument Merging

The factory implements smart argument merging when creating nodes:

1. **Positional Arguments**:
   - Base arguments from registration
   - Additional arguments from creation call
   - Combined in order: base + additional

2. **Keyword Arguments**:
   - Deep merge of dictionaries
   - Creation arguments take precedence
   - Nested dictionary merging supported

```python
# Registration with base metadata
get_node_factory().register(
    RunnableNode,
    metadata={"base": True}
)

# Creation with additional metadata
node = get_node_factory().create_node(
    "RunnableNode",
    metadata={"custom": True}
)

# Result:
# node.metadata == {"base": True, "custom": True}
```

## Integration Examples

### Network Integration

```python
# Network setup with factory
def setup_network():
    factory = get_node_factory()
    
    # Register network nodes
    factory.register(RunnableNode)
    factory.register(RunnableHumanNode)
    
    # Create network with nodes
    with RunnableNetwork() as network:
        factory.create_node("RunnableHumanNode")
        factory.create_node("RunnableNode")
    
    return network
```
