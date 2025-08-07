# NetworkModifier

## Overview

`NetworkModifier` is a core component in the LC Agent framework that provides a safe and controlled way to modify network behavior and structure during execution. It acts as a middleware layer that can intercept and modify network execution at specific points in the execution lifecycle.

## Purpose

The `NetworkModifier` solves several key problems in network execution:

1. **Safe Network Modification**: Provides the only valid way to modify network structure and behavior during execution, preventing conflicts and race conditions.
2. **Execution Lifecycle Hooks**: Offers hooks into different stages of network execution for custom behavior.
3. **State Management**: Enables tracking and modification of network state in a controlled manner.
4. **Dynamic Network Evolution**: Allows networks to evolve and adapt based on execution results.

## Core Components

### Class Definition

```python
class NetworkModifier:
    def on_begin_invoke(self, network: "RunnableNetwork") -> None:
        """Called at the start of network execution"""
        pass

    def on_pre_invoke(self, network: "RunnableNetwork", node: "RunnableNode") -> None:
        """Called before each node execution"""
        pass

    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode") -> None:
        """Called after each node execution"""
        pass

    def on_end_invoke(self, network: "RunnableNetwork") -> None:
        """Called at the end of network execution"""
        pass

    async def on_begin_invoke_async(self, network: "RunnableNetwork") -> None:
        """Async version of on_begin_invoke"""
        return self.on_begin_invoke(network)

    async def on_pre_invoke_async(self, network: "RunnableNetwork", node: "RunnableNode") -> None:
        """Async version of on_pre_invoke"""
        return self.on_pre_invoke(network, node)

    async def on_post_invoke_async(self, network: "RunnableNetwork", node: "RunnableNode") -> None:
        """Async version of on_post_invoke"""
        return self.on_post_invoke(network, node)

    async def on_end_invoke_async(self, network: "RunnableNetwork") -> None:
        """Async version of on_end_invoke"""
        return self.on_end_invoke(network)
```

## Execution Lifecycle

### 1. Begin Invoke
- Called once at the start of network execution
- Used for network-wide initialization
- Can set up initial network state
- Example: Creating initial nodes in an empty network

```python
def on_begin_invoke(self, network: "RunnableNetwork"):
    if not network.nodes:
        # Initialize empty network with starting nodes
        with network:
            RunnableHumanNode("Initial prompt")
```

### 2. Pre-Invoke
- Called before each node execution
- Can modify node before execution
- Access to node's inputs and configuration
- Example: Adding RAG content to system prompts

```python
def on_pre_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
    if isinstance(node, RunnableAINode):
        # Add retrieval content to system prompt
        retrieval_results = self._get_relevant_content(node)
        node.inputs.insert(1, retrieval_results)
```

### 3. Post-Invoke
- Called after each node execution
- Can process node results
- Can create new nodes based on output
- Most common point for network modification
- Example: Processing tool calls from AI output

```python
def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
    if (node.invoked and 
        isinstance(node.outputs, AIMessage) and 
        node.outputs.tool_calls and 
        not network.get_children(node)):
        
        # Create tool nodes for each tool call
        for tool_call in node.outputs.tool_calls:
            tool_node = RunnableToolNode(tool_call)
            node >> tool_node
```

### 4. End Invoke
- Called once at the end of network execution
- Used for cleanup and final processing
- Can perform network-wide operations
- Example: Cleaning up temporary resources

```python
def on_end_invoke(self, network: "RunnableNetwork"):
    # Cleanup temporary resources
    self._cleanup_resources()
    # Reset network state
    self._reset_state()
```

## Common Modifier Patterns

### 1. Node Creation and Connection
```python
def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
    with network:
        # Create new node
        new_node = RunnableNode(...)
        # Connect to current node
        node >> new_node
```

### 2. Branch Creation
```python
def create_branch(self, network: "RunnableNetwork", node: "RunnableNode", data):
    with network:
        # Create branch starting node
        branch_start = RunnableHumanNode(data["prompt"])
        # Connect to parent
        node >> branch_start
        # Add branch nodes
        branch_start >> RunnableNode(data["system_message"])
```

### 3. Node Metadata Management
```python
def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
    # Update node metadata
    node.metadata["processed"] = True
    node.metadata["execution_time"] = time.time() - start_time
```

### 4. Conditional Network Modification
```python
def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
    if node.invoked and not network.get_children(node):
        if "error" in node.metadata:
            # Create error handling branch
            self._create_error_branch(network, node)
        else:
            # Create success branch
            self._create_success_branch(network, node)
```

## Common Use Cases

### 1. Tool Handling
Used to process tool calls from AI responses and create appropriate tool nodes:
```python
class ToolModifier(NetworkModifier):
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}

    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        if (node.invoked and 
            isinstance(node.outputs, AIMessage) and 
            node.outputs.tool_calls):
            
            for tool_call in node.outputs.tool_calls:
                tool = self.tools.get(tool_call["name"])
                if tool:
                    result = tool.invoke(tool_call["args"])
                    tool_node = RunnableToolNode(result)
                    node >> tool_node
```

### 2. RAG Integration
Adds retrieval-augmented generation content to nodes:
```python
class RAGModifier(NetworkModifier):
    def __init__(self, retriever):
        self.retriever = retriever

    def on_pre_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        if isinstance(node, RunnableAINode):
            # Get relevant content
            context = self.retriever.get_relevant_docs(node.inputs)
            # Add to node inputs
            node.inputs.insert(1, context)
```

### 3. Code Processing
Handles code execution and error management:
```python
class CodeInterpreterModifier(NetworkModifier):
    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        if node.invoked and isinstance(node.outputs, AIMessage):
            code = self._extract_code(node.outputs.content)
            result = self._execute_code(code)
            
            if "error" in result:
                self._create_error_node(network, node, result)
            else:
                self._create_success_node(network, node, result)
```

### 4. Classification and Routing
Routes messages based on classification results:
```python
class ClassificationModifier(NetworkModifier):
    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        if isinstance(node, ClassificationNode) and node.invoked:
            result = self._parse_classification(node.outputs)
            if result:
                self._route_to_appropriate_node(network, node, result)
```

## Best Practices

1. **Safe Network Modification**
   - Never modify network outside modifier methods
   - Keep modifications atomic and self-contained

2. **State Management**
   - Use node metadata for state tracking
   - Avoid global state in modifiers
   - Clean up temporary state in `on_end_invoke`

3. **Error Handling**
   - Handle errors gracefully in modifiers
   - Create appropriate error branches
   - Maintain network stability during errors

4. **Performance**
   - Minimize expensive operations in modifiers
   - Cache results when possible
   - Use async methods for I/O operations

5. **Modifier Design**
   - Keep modifiers focused on single responsibility
   - Make modifiers composable
   - Document modifier behavior and requirements

## Common Pitfalls

1. **Race Conditions**
   ```python
   # BAD: Accessing shared state without synchronization
   self.global_counter += 1
   
   # GOOD: Use node metadata
   node.metadata["counter"] = node.metadata.get("counter", 0) + 1
   ```

2. **Infinite Loops**
   ```python
   # BAD: Creating nodes without termination condition
   def on_post_invoke(self, network, node):
       node >> RunnableNode()  # Will create nodes infinitely
   
   # GOOD: Check conditions
   def on_post_invoke(self, network, node):
       if not network.get_children(node) and some_condition:
           node >> RunnableNode()
   ```

## Integration with RunnableNetwork

### Modifier Registration

Modifiers are registered with the network using the `add_modifier` method:

```python
network.add_modifier(modifier: NetworkModifier, once: bool = False, priority: Optional[int] = None) -> int
```

- `once`: When True, prevents duplicate modifiers of the same type
- `priority`: Determines execution order (lower numbers execute first)
- Returns modifier ID for later removal

Example:
```python
# Add a modifier with priority
network.add_modifier(RAGModifier(retriever), priority=100)

# Add a singleton modifier
network.add_modifier(CodeInterpreterModifier(), once=True)
```

### Modifier Execution Order

1. Modifiers are executed in priority order (lower numbers first)
2. For each execution phase (begin/pre/post/end), all modifiers are called in sequence
3. New modifiers added during execution are included in subsequent phases
4. Default modifier (priority -100) is always added to new networks

### Network State Management

1. **Active Network Stack**
   ```python
   # Get current active network
   current = RunnableNetwork.get_active_network()
   
   # Get all active networks (most recent first)
   for network in RunnableNetwork.get_active_networks():
       # Process network
   ```

2. **Network Context**
   ```python
   # Network context ensures proper modifier execution
   with network:
       # Modifiers will be called for nodes created here
       node = RunnableNode(...)
   ```

## Modifier Composition

### Chaining Modifiers

Multiple modifiers can work together to create complex behaviors:

```python
# Add modifiers in desired execution order
network.add_modifier(RAGModifier(retriever), priority=100)      # Add context first
network.add_modifier(ToolModifier(tools), priority=200)         # Process tools second
network.add_modifier(CodeInterpreterModifier(), priority=300)   # Execute code last
```

### Modifier Communication

Modifiers can communicate through node metadata:

```python
class FirstModifier(NetworkModifier):
    def on_post_invoke(self, network, node):
        node.metadata["processed_by_first"] = True
        node.metadata["shared_data"] = {"key": "value"}

class SecondModifier(NetworkModifier):
    def on_post_invoke(self, network, node):
        if node.metadata.get("processed_by_first"):
            shared_data = node.metadata.get("shared_data", {})
            # Use shared data
```

### Modifier Dependencies

When modifiers depend on each other:

```python
class DependentModifier(NetworkModifier):
    def __init__(self, required_modifier_type):
        self.required_type = required_modifier_type

    def on_begin_invoke(self, network):
        # Check if required modifier exists
        modifier_id = network.get_modifier_id(self.required_type)
        if modifier_id is None:
            # Add required modifier
            network.add_modifier(self.required_type(), priority=self._get_priority() - 1)
```

## Advanced Patterns

### Conditional Execution

```python
class ConditionalModifier(NetworkModifier):
    def on_post_invoke(self, network, node):
        if not self._should_process(node):
            return
        
        # Process node
        self._process_node(network, node)

    def _should_process(self, node):
        return (node.invoked and 
                not node.metadata.get("processed") and
                isinstance(node.outputs, AIMessage))
```

**Important:** All modifications to the network's structure and node state must be performed exclusively via `NetworkModifier` implementations. Direct modifications—such as manually changing the contents of `RunnableNetwork.nodes`, altering node connections with operators (e.g. using >>) outside of modifier hooks, or directly updating node metadata from outside modifier methods—are strictly prohibited. This design ensures consistency, prevents race conditions, and avoids unexpected behavior during execution.

## Critical: Preventing Infinite Loops

### Understanding Modifier Execution Cycle

**Important:** Modifiers are called for EVERY node during EACH execution cycle. This means:
1. Each modifier's hooks are called repeatedly
2. The same node can be processed multiple times
3. Without proper conditions, modifiers can create infinite loops

### Node Targeting Conditions

To prevent infinite loops, modifiers must use specific conditions to target exactly when they should act. Common patterns from production modifiers:

```python
# Basic node targeting pattern
if (node.invoked and                     # Node has been executed
    isinstance(node, RunnableAINode) and # Specific node type
    not network.get_children(node)):     # Node has no children yet
    # Safe to modify network here

# More complex targeting
if (node.invoked and 
    isinstance(node.outputs, AIMessage) and    # Check output type
    node.outputs.tool_calls and               # Has specific data
    not network.get_children(node) and        # No children
    "processed" not in node.metadata):        # Not already processed
    # Safe to modify network here

# Tool node handling
if (node.invoked and 
    isinstance(node, RunnableToolNode) and    # Tool node
    not network.get_children(node) and        # No children
    isinstance(network, MultiAgentNetworkNode) and  # Specific network
    network.multishot):                       # Network feature flag
    # Safe to handle tool node
```

### Common Targeting Conditions

1. **Execution State**
   - `node.invoked`: Node has been executed
   - `not network.get_children(node)`: Node has no children yet

2. **Node Type**
   - `isinstance(node, RunnableAINode)`
   - `isinstance(node, RunnableToolNode)`
   - `isinstance(node, RunnableHumanNode)`

3. **Output Type**
   - `isinstance(node.outputs, AIMessage)`
   - `isinstance(node.outputs, HumanMessage)`

4. **Metadata Checks**
   - `"processed" not in node.metadata`
   - `"interpreter_code" not in node.metadata`
   - `node.metadata.get("contribute_to_history", True)`

5. **Network Type**
   - `isinstance(network, MultiAgentNetworkNode)`
   - `isinstance(network, NetworkNode)`

### Real-World Examples

1. **Default Modifier** - Adds default node after human messages:
```python
class DefaultModifier(NetworkModifier):
    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        if (node.invoked and 
            isinstance(node.outputs, HumanMessage) and 
            not network.get_children(node)):  # Prevent infinite loop
            
            default_node = network.default_node
            if default_node:
                node >> get_node_factory().create_node(default_node)
```

2. **Code Interpreter** - Executes code and handles results:
```python
class CodeInterpreterModifier(NetworkModifier):
    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        if (node.invoked and 
            isinstance(node, RunnableAINode) and 
            not network.get_children(node) and
            "interpreter_code" not in node.metadata and  # Prevent reprocessing
            "stop_reason" not in node.metadata):
            
            code = self._extract_code(node.outputs.content)
            # Process code...
```

3. **Tool Handler** - Processes tool calls:
```python
class ToolModifier(NetworkModifier):
    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        if (node.invoked and 
            isinstance(node.outputs, AIMessage) and 
            node.outputs.tool_calls and 
            not network.get_children(node)):  # Prevent infinite loop
            
            for tool_call in node.outputs.tool_calls:
                # Process tool calls...
```

### Preventing Reprocessing

1. **Using Metadata Flags**
```python
def on_post_invoke(self, network, node):
    if (node.invoked and 
        not node.metadata.get("processed")):  # Check if already processed
        
        # Process node
        node.metadata["processed"] = True     # Mark as processed
```

2. **Using Type Checks**
```python
def on_post_invoke(self, network, node):
    if (node.invoked and 
        isinstance(node, TargetNodeType) and  # Only specific node types
        not isinstance(node, ProcessedNodeType)):  # Exclude processed types
        
        # Process node
```

3. **Using Network State**
```python
def on_post_invoke(self, network, node):
    if (node.invoked and 
        network.get_leaf_nodes()[-1] is node):  # Only process leaf nodes
        
        # Process node
```

### Common Mistakes

1. **Missing Node State Check**
```python
# BAD: Will create nodes even if not executed
def on_post_invoke(self, network, node):
    node >> RunnableNode()  # Creates infinite nodes

# GOOD: Check node state
def on_post_invoke(self, network, node):
    if node.invoked and not network.get_children(node):
        node >> RunnableNode()
```

2. **Missing Child Check**
```python
# BAD: Will keep creating children
def on_post_invoke(self, network, node):
    if isinstance(node, TargetType):
        node >> RunnableNode()  # Infinite loop

# GOOD: Check for existing children
def on_post_invoke(self, network, node):
    if isinstance(node, TargetType) and not network.get_children(node):
        node >> RunnableNode()
```

3. **Missing Metadata Check**
```python
# BAD: Will reprocess same node
def on_post_invoke(self, network, node):
    if node.invoked:
        process_node(node)  # Infinite processing

# GOOD: Track processing state
def on_post_invoke(self, network, node):
    if node.invoked and "processed" not in node.metadata:
        process_node(node)
        node.metadata["processed"] = True
```

### Best Practices for Loop Prevention

1. **Always Check Node State**
   - `node.invoked`: Ensure node has been executed
   - `not network.get_children(node)`: Ensure no existing children

2. **Use Specific Type Checks**
   - Check node type: `isinstance(node, TargetType)`
   - Check output type: `isinstance(node.outputs, MessageType)`

3. **Track Processing State**
   - Use metadata flags: `node.metadata["processed"] = True`
   - Check for previous processing: `"processed" not in node.metadata`

4. **Validate Network State**
   - Check network type: `isinstance(network, NetworkType)`
   - Verify network features: `network.multishot`

5. **Implement Safety Checks**
   - Maximum iteration counters
   - Depth limits
   - State validation
