# NetworkNode

## Overview

`NetworkNode` is a specialized class in the LC-Agent framework that represents a subnetwork. It is designed to combine the capabilities of both `RunnableNode` and `RunnableNetwork` by inheriting from both. This hybrid approach allows a NetworkNode to function as a self-contained subnetwork while still participating in the overarching network’s execution lifecycle.

## Purpose & Role

- **Subnetwork Representation:**  
  A NetworkNode encapsulates a subnetwork within the overall agent system. This allows parts of a conversation (or different tasks) to be processed in isolation while still being fully integrated into the main network.

- **Parent-Child Integration:**  
  By inheriting from both `RunnableNode` and `RunnableNetwork`, the NetworkNode facilitates seamless communication between the parent network and its subnetwork. It automatically connects its root nodes to their appropriate parents via its built-in modifier.

- **Modifier-Driven Behavior:**  
  NetworkNode automatically adds a `NetworkNodeModifier` upon initialization. This modifier helps connect the subnetwork to its parent network by ensuring that root nodes are properly linked, providing automatic fallback and inheritance of properties (such as the chat model).

- **Lifecycle Management:**  
  It integrates into the network lifecycle by overriding key methods like `_pre_invoke_network` and `_post_invoke_network`. These methods ensure that the subnetwork inherits necessary settings (for example, the chat model name from the active parent network) before processing begins, and they provide a hook for additional post-processing as needed.

## Key Features

- **Dual Inheritance:**  
  NetworkNode extends both `RunnableNode` and `RunnableNetwork`, thereby merging node-level processing with network-level behaviors. This makes it possible to treat it either as a single processing element or as a full-fledged network.

- **Modifier Integration:**  
  The class registers an instance of `NetworkNodeModifier` during initialization. This modifier handles the connection between the subnetwork and the parent network:
  - In the `on_begin_invoke` hook, it creates a default node if the subnetwork is empty.
  - In the `on_pre_invoke` hook, it connects root nodes to their parent networks when no parents are present.

- **Lifecycle Method Overrides:**  
  - `_pre_invoke_network`: Called before the network is invoked. If an active parent network exists (via `RunnableNetwork.get_active_network()`), this method inherits the chat model name if not already set.
  - `_post_invoke_network`: A placeholder for post-processing the subnetwork after execution. It can be customized for additional cleanup.
  
- **Delegated Model Invocation:**  
  NetworkNode provides the following methods to call the chat model:
  - `_ainvoke_chat_model`: Asynchronously invokes the chat model using the subnetwork context.
  - `_invoke_chat_model`: Synchronously invokes the chat model.
  - `_astream_chat_model`: Streams chat model responses.
  
  These methods delegate to the underlying `RunnableNetwork` methods after performing pre- and post-invocation tasks.

## Lifecycle & Integration

### Preparation Phase

- **Pre-Invocation:**  
  The `_pre_invoke_network` method is called before invoking the network. It checks for an active parent network; if one exists and if the network's chat model name isn’t set, it copies the parent model name. This ensures consistency across nested networks.

### Execution Phase

- **Model Invocation Delegation:**  
  Methods `_invoke_chat_model`, `_ainvoke_chat_model`, and `_astream_chat_model` wrap around the corresponding methods from `RunnableNetwork`. They call `_pre_invoke_network` first, then execute the desired chat model call, and finally call `_post_invoke_network`.

### Post-Execution Phase

- **Post-Invocation:**  
  The `_post_invoke_network` method is a placeholder intended for additional clean-up or final adjustments after the subnetwork completes its execution. It can be overridden as needed.

### Modifier Usage

NetworkNode adds a `NetworkNodeModifier` automatically. This modifier:
- Ensures that if the subnetwork is empty, a default child node (as specified by the `default_node` property) is created.
- Connects nodes that have no parents to the root of the subnetwork, ensuring proper inheritance of properties.

## How to Use NetworkNode

### 1. Registration

Register your NetworkNode (and any specialized subclasses) with the node factory so they can be used within your application. For example, in a multi-agent setting, NetworkNode types might be registered to represent different sub-networks.

```python
from lc_agent import get_node_factory, NetworkNode

# Register a custom network node
get_node_factory().register(NetworkNode, name="CustomNetworkNode")
```

### 2. Instantiation & Configuration

Create an instance of a NetworkNode (or a subclass) and configure its properties:

- Set the `default_node` property to specify the fallback node for the network.
- Configure any modifiers or additional properties, such as the chat model name.

```python
from lc_agent import NetworkNode

# Create a subnetwork instance
network_node = NetworkNode(default_node="RunnableNode", chat_model_name="gpt-4")
```

### 3. Running in Context

Use a `with` statement to activate a NetworkNode. This provides a context in which the node is considered active, and any nodes created inside the context are tracked by the NetworkNode (and connected automatically via the NetworkNodeModifier).

```python
from lc_agent import RunnableHumanNode, RunnableNetwork

with network_node:
    # Create nodes that automatically join this subnetwork
    RunnableHumanNode("Hello from the subnetwork!")
```

### 4. Invocation

Invoke the network using the provided methods. The NetworkNode will handle pre-invocation (inheriting settings), execution (via delegated chat model invocation), and post-invocation tasks.

```python
# Synchronously invoke using the default node settings
result = network_node.invoke()

# Asynchronously or with streaming:
async for chunk in network_node.astream():
    print(chunk.content)
```

### 5. Integration within Multi-Agent Systems

In multi-agent scenarios (see testing_fc_routing.py for an example), NetworkNode can act as a subnetwork within a larger setup. It may be registered as part of the multi-agent network via the node factory, and its modifiers ensure that its behavior integrates seamlessly with the parent network (e.g., via shared chat models and event callbacks).

## Real-World Example

Consider the `testing_fc_routing.py` example, where multiple node types—including multi-agent network nodes—are registered. In that example:

- The NetworkNode is registered along with other node types.
- It is used as part of a larger network (`RunnableNetwork`) with a specified default node.
- When a user query is sent via `RunnableHumanNode`, the NetworkNode (as a subnetwork) activates and uses its modifiers to connect to parent and child nodes, and to propagate the appropriate chat model settings.
- It utilizes the environment settings (via `register_llama_model`) to ensure that the subnetwork is correctly configured.

## Best Practices

- **Always Use Context Managers:**  
  Activate NetworkNode using a `with` statement to ensure that all node creations are properly tracked and integrated.
  
- **Leverage Modifier Integration:**  
  Do not attempt to modify network structures directly. Instead, rely on the subclassed `NetworkNodeModifier` to connect subnetworks to their parents. This avoids issues like inconsistent state or infinite loops.
  
- **Inherit Chat Model Settings:**  
  Allow NetworkNode to inherit the chat model name from its parent network when not provided, ensuring consistency across the application.
  
- **Override Lifecycle Hooks When Needed:**  
  If you need specialized behavior before or after network invocation, override `_pre_invoke_network` and `_post_invoke_network` in your subclass.
  
- **Keep Modifiers Clean and Focused:**  
  Use modifiers to connect subnetwork nodes automatically and track node states to prevent unnecessary reprocessing.
