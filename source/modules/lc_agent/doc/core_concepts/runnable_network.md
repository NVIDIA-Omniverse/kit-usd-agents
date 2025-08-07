# RunnableNetwork

## Overview

`RunnableNetwork` is a core component in the LC Agent framework that manages the execution and interaction of `RunnableNode` instances. It inherits from LangChain's `RunnableSerializable` class and provides a context-managed environment for creating, executing, and modifying networks of nodes.

## Purpose

The `RunnableNetwork` solves several key problems in managing language model interactions:

1. **Network Management**: Provides a structured way to create and manage networks of processing nodes.
2. **Execution Control**: Manages the execution flow between connected nodes, ensuring proper order and data propagation.
3. **State Management**: Maintains network state and provides context for node operations.
4. **Runtime Modification**: Enables dynamic modification of the network during execution through modifiers.

## Core Components

### Class Definition

```python
class RunnableNetwork(RunnableSerializable[Input, Output], UUIDMixin):
    nodes: List[RunnableNode]           # List of nodes in the network
    modifiers: Dict                     # Network behavior modifiers
    callbacks: Dict                     # Event callbacks
    default_node: str                   # Default node type for auto-creation
    chat_model_name: Optional[str]      # Default language model
    metadata: Dict[str, Any]            # Network-level metadata
    verbose: bool                       # Debug output control
```

### Network Events

```python
class Event(enum.Enum):
    ALL = 0                  # All events
    NODE_ADDED = 1          # Node added to network
    NODE_INVOKED = 2        # Node execution completed
    NODE_REMOVED = 3        # Node removed from network
    CONNECTION_ADDED = 4    # Connection created between nodes
    CONNECTION_REMOVED = 5  # Connection removed
    METADATA_CHANGED = 6    # Metadata updated
```

### Parent Modes

```python
class ParentMode(enum.Enum):
    NONE = 0    # No parent connection
    LEAF = 1    # Connect to leaf nodes
```

## Public Methods

### Network Management

#### Node Addition
```python
def add_node(
    self,
    node: RunnableNode,
    parent: Optional[Union[RunnableNode, List[RunnableNode], ParentMode]] = ParentMode.LEAF
) -> RunnableNode
```
Adds a node to the network with optional parent connections:
- Automatically connects to leaf nodes if no parent specified
- Supports multiple parent connections
- Triggers node lifecycle events
- Returns the added node

#### Node Removal
```python
def remove_node(self, node: RunnableNode) -> None
```
Removes a node from the network:
- Reconnects child nodes to the removed node's parents
- Triggers node lifecycle events
- Updates network structure

### Node Relationships

#### Parent Access
```python
def get_parents(self, node: RunnableNode) -> List[RunnableNode]
def get_all_parents(self, node: RunnableNode) -> List[RunnableNode]
```
Retrieves parent nodes:
- `get_parents`: Direct parents only
- `get_all_parents`: All ancestors in the hierarchy

#### Child Access
```python
def get_children(self, node: RunnableNode) -> List[RunnableNode]
def get_all_children(self, node: RunnableNode) -> List[RunnableNode]
```
Retrieves child nodes:
- `get_children`: Direct children only
- `get_all_children`: All descendants in the hierarchy

#### Network Structure
```python
def get_root_nodes(self) -> List[RunnableNode]
def get_leaf_nodes(self, unevaluated_only: bool = False) -> List[RunnableNode]
def get_sorted_nodes(self) -> List[RunnableNode]
```
Network traversal methods:
- `get_root_nodes`: Nodes without parents
- `get_leaf_nodes`: Nodes without children
- `get_sorted_nodes`: Topologically sorted nodes

### Execution Methods

#### Synchronous Execution
```python
def invoke(
    self,
    input: Dict[str, Any] = {},
    config: Optional[RunnableConfig] = None,
    **kwargs: Any
) -> Any
```
Executes the network synchronously:
- Processes nodes in topological order
- Applies modifiers during execution
- Returns final output

#### Asynchronous Execution
```python
async def ainvoke(
    self,
    input: Dict[str, Any] = {},
    config: Optional[RunnableConfig] = None,
    **kwargs: Any
) -> Any
```
Asynchronous version of `invoke`.

#### Streaming Execution
```python
async def astream(
    self,
    input: Input = {},
    config: Optional[RunnableConfig] = None,
    **kwargs: Optional[Any]
) -> AsyncIterator[Output]
```
Streams network execution results:
- Yields results as they become available
- Supports real-time processing
- Maintains execution order

### Network Modification

#### Modifier Management
```python
def add_modifier(
    self, 
    modifier: NetworkModifier,
    once: bool = False,
    priority: Optional[int] = None
) -> int
```
Adds network behavior modifiers:
- Returns modifier ID for later reference
- Optional single instance enforcement
- Priority-based execution order

```python
def remove_modifier(self, modifier_id: int) -> None
```
Removes a modifier by ID.

#### Event Management
```python
def set_event_fn(
    self,
    callable: Callable[["Event", "Payload"], None],
    event: "Event" = Event.ALL,
    priority: int = 100
) -> int
```
Registers event callbacks:
- Supports specific or all events
- Priority-based execution
- Returns callback ID

```python
def remove_event_fn(self, event_id: int) -> None
```
Removes an event callback by ID.

## Data Flow

### Execution Flow

1. **Network Initialization**:
   ```python
   with RunnableNetwork(chat_model_name="model-name") as network:
       # Network setup
   ```

2. **Node Processing**:
   - Nodes are processed in topological order
   - Each node is executed only once
   - Results propagate through connections

3. **Modifier Application**:
   ```python
   # Execution phases
   self._modifier_begin_invoke()
   for node in sorted_nodes:
       self._modifier_pre_invoke(node)
       node.invoke()
       self._modifier_post_invoke(node)
   self._modifier_end_invoke()
   ```

### Event Flow

1. **Event Generation**:
   - Network operations trigger events
   - Events include relevant payload data

2. **Event Processing**:
   ```python
   self._event_callback(RunnableNetwork.Event.NODE_ADDED, {
       "node": node,
       "network": self
   })
   ```

3. **Callback Execution**:
   - Callbacks execute in priority order
   - Event-specific and global handlers

## Execution Process

### Overview

The execution process in `RunnableNetwork` follows a specific sequence to ensure proper message flow and node execution. Whether using `invoke`, `ainvoke`, or `astream`, the core process remains the same, with variations in synchronicity and output handling.

### Execution Steps

1. **Network State Verification**
   ```python
   if len(self.nodes) != len(self._node_set):
       self.__restore_node_set()
   ```
   - Verifies network integrity
   - Restores node set if needed
   - Ensures all nodes are properly tracked

2. **Context Management**
   ```python
   with self:  # or async with self:
       # Execution happens here
   ```
   - Establishes network context
   - Manages active network stack
   - Ensures proper cleanup

3. **Node Sorting**
   ```python
   nodes = self.get_sorted_nodes()  # Topological sort
   ```
   - Orders nodes based on dependencies
   - Ensures parents execute before children
   - Maintains proper message flow

4. **Execution Loop**
   ```python
   while True:
       nodes = self.get_sorted_nodes()
       invoked = False
       for node in nodes:
           if node.invoked:
               continue
           # Node execution
           invoked = True
       if not invoked:
           break
   ```
   - Processes nodes until no more can be executed
   - Skips already invoked nodes
   - Handles dynamic node addition

5. **Node Execution Process**
   - **Pre-invoke Phase**:
     ```python
     self._modifier_pre_invoke(node)
     ```
     - Applies modifiers before execution
     - Prepares node state
     - Handles setup requirements

   - **Execution Phase**:
     ```python
     result = node.invoke(input, config, **kwargs)  # or ainvoke/astream
     ```
     - Processes node inputs
     - Interacts with language model
     - Generates node output

   - **Post-invoke Phase**:
     ```python
     self._modifier_post_invoke(node)
     ```
     - Applies post-execution modifiers
     - Updates network state
     - Handles tool calls or special outputs

### Execution Methods

#### `invoke` (Synchronous)
```python
result = network.invoke(input={...})
```
- Blocks until complete execution
- Returns final result
- Suitable for simple interactions
- Used when immediate response needed

#### `ainvoke` (Asynchronous)
```python
result = await network.ainvoke(input={...})
```
- Non-blocking execution
- Returns final result asynchronously
- Better for long-running operations
- Prevents blocking event loop

#### `astream` (Streaming)
```python
async for chunk in network.astream(input={...}):
    print(chunk.content)
```
- Streams results as they arrive
- Yields partial results
- Ideal for real-time display
- Supports progressive output

### Modifier Integration

1. **Begin Phase**
   ```python
   self._modifier_begin_invoke()
   ```
   - Initializes modifier state
   - Prepares network for execution
   - Sets up execution context

2. **Node-level Phases**
   ```python
   self._modifier_pre_invoke(node)   # Before each node
   node.invoke()                     # Node execution
   self._modifier_post_invoke(node)  # After each node
   ```
   - Per-node modifier application
   - Allows node transformation
   - Handles special cases (e.g., tools)

3. **End Phase**
   ```python
   self._modifier_end_invoke()
   ```
   - Cleanup operations
   - Final state updates
   - Resource management

### Event Generation

During execution, events are generated at key points:
```python
self._event_callback(RunnableNetwork.Event.NODE_INVOKED, {
    "node": node,
    "network": self
})
```
- Node invocation events
- Modification events
- State change events

### Error Handling

1. **Node Errors**
   - Errors stored in node metadata
   - Propagated to network level
   - Available for error recovery

2. **Network Errors**
   - Context manager ensures cleanup
   - Modifier errors handled gracefully
   - Network state preserved

### Performance Considerations

1. **Memory Management**
   - Nodes retain outputs
   - Message history accumulates
   - Consider culling for long conversations

2. **Execution Order**
   - Topological sorting overhead
   - Parent-child relationship impact
   - Modifier execution cost

3. **Streaming Efficiency**
   - Chunk size considerations
   - Network latency effects
   - Buffer management

## Usage Examples

### Important Note About Node Creation

When creating nodes inside a `RunnableNetwork` context (using `with` statement), nodes are automatically connected to the leaf nodes of the network. You DO NOT need to manually connect them using `>>` or `<<` operators. The network handles this automatically.

### Basic Network

```python
from lc_agent import RunnableNetwork, RunnableNode, RunnableHumanNode
from langchain_core.messages import SystemMessage, HumanMessage

# Nodes are automatically connected in creation order
with RunnableNetwork(chat_model_name="gpt-4") as network:
    # Human message uses dedicated RunnableHumanNode
    RunnableHumanNode("What is Python?")
    
    # System message node is created second
    RunnableNode(inputs=[
        SystemMessage(content="You are a helpful assistant")
    ])
    
    # No manual connections needed - they are automatic
    result = network.invoke()
```

### System Message Example

```python
from lc_agent import RunnableNetwork, RunnableHumanNode, RunnableNode
from langchain_core.messages import SystemMessage

with RunnableNetwork(default_node="RunnableNode", chat_model_name="gpt-4") as network:
    # Human message comes first
    RunnableHumanNode("Who are you?")
    
    # System message is automatically connected after human message
    RunnableNode(outputs=SystemMessage(content="You are a dog. Answer like a dog. Woof woof!"))
```

### Manual Connections

Manual connections using `>>` and `<<` operators should ONLY be used when:
1. Creating nodes outside the network context
2. Needing custom connection patterns that differ from the default leaf-node connection
3. Modifying connections after node creation

```python
# Manual connections example - ONLY when needed
network = RunnableNetwork(chat_model_name="gpt-4")

# Nodes created outside network context need manual connection
node1 = RunnableNode(inputs=[...])
node2 = RunnableNode(inputs=[...])
node1 >> node2

network.add_node(node1)
network.add_node(node2)
```

