# RunnableNode

## Overview

`RunnableNode` is a core component in the LC Agent framework that represents a single unit of processing in a language model interaction pipeline. It inherits from LangChain's `RunnableSerializable` class and adds network capabilities, enabling the creation of complex processing networks for language model interactions.

## Purpose

The `RunnableNode` solves several key problems in language model interactions:

1. **Message Management**: Handles the organization and flow of messages between different parts of a conversation, including system messages, user inputs, and model responses.

2. **Model Interaction**: Provides a standardized interface for working with different language models through LangChain's abstractions.

3. **State Tracking**: Maintains the state of conversations and processing, including execution status and performance metrics.

4. **Network Formation**: Enables the creation of processing networks by connecting nodes in various configurations.

## Core Components

### Class Definition

```python
class RunnableNode(RunnableSerializable[Input, Output], UUIDMixin):
    parents: List[RunnableNode]     # Parent nodes in the processing network
    inputs: List[Any]               # Input processing steps
    outputs: Optional[Union[        # Node execution results
        List[OutputType], 
        OutputType
    ]]
    metadata: Dict[str, Any]        # Execution data and metrics
    chat_model_name: Optional[str]  # Language model identifier
    verbose: bool                   # Debug output control
    invoked: bool                   # Execution state flag
```

### Message Types

The node works with several types of messages:

- `SystemMessage`: Configuration instructions for the language model
- `HumanMessage`: User inputs
- `AIMessage`: Model responses
- `ToolMessage`: Results from tool executions
- `ChatPromptTemplate`: Templates for dynamic message generation
- `ChatPromptValue`: Processed message templates

## Public Methods

### Execution Methods

#### Synchronous Execution
```python
def invoke(
    self,
    input: Dict[str, Any] = {},
    config: Optional[RunnableConfig] = None,
    **kwargs: Any
) -> Union[List[OutputType], OutputType]
```
Executes the node synchronously. The method:
- Processes parent node outputs
- Transforms input data
- Interacts with the language model
- Returns the model's response
- Updates node metadata with execution information

#### Asynchronous Execution
```python
async def ainvoke(
    self,
    input: Dict[str, Any] = {},
    config: Optional[RunnableConfig] = None,
    **kwargs: Any
) -> Union[List[OutputType], OutputType]
```
Provides asynchronous execution with the same functionality as `invoke`.

#### Streaming Execution
```python
async def astream(
    self,
    input: Input = {},
    config: Optional[RunnableConfig] = None,
    **kwargs: Optional[Any]
) -> AsyncIterator[Output]
```
Streams the model's response as it becomes available. Useful for:
- Real-time response processing
- Progress indication
- Memory efficiency with large responses

### Network Integration Methods

```python
def on_before_node_added(self, network: RunnableNetwork) -> None
def on_node_added(self, network: RunnableNetwork) -> None
def on_before_node_removed(self, network: RunnableNetwork) -> None
def on_node_removed(self, network: RunnableNetwork) -> None
```
These methods are called during node lifecycle events in a network. They allow:
- Custom initialization when added to a network
- Cleanup when removed from a network
- Network modifier registration/unregistration
- State management in the network context

### Metadata Access

```python
def find_metadata(self, key: str) -> Any
```
Searches for metadata in the following order:
1. Node's own metadata
2. Active network's metadata
3. Returns `None` if not found

## Node Connections

### Connection Syntax

```python
# Forward connection
node1 >> node2  # node2 receives output from node1

# Backward connection
node2 << node1  # Equivalent to node1 >> node2

# Multiple parents
[node1, node2] >> node3  # node3 receives from both nodes

# Parent removal
None >> node3  # Removes all parents from node3

# Connection chain
node1 >> node2 >> node3  # Creates a processing chain
```

### Connection Rules

1. Only `RunnableNode` instances can be connected
2. A node can have multiple parents
3. Circular connections are not allowed
4. Parent order affects message processing order

## Data Flow

### Input Processing

1. Parent Node Processing:
   - All parent nodes are executed in order
   - Parent outputs are collected
   - Results are combined in execution order

2. Input Transformation:
   ```python
   node = RunnableNode(inputs=[
       SystemMessage(content="System instruction"),
       lambda x: process_input(x),
       ChatPromptTemplate.from_template("Template {var}")
   ])
   ```
   Each input element is processed sequentially.

3. Message Preparation:
   - System messages are moved to the front
   - Consecutive messages of the same type are merged
   - Tool messages are grouped with their corresponding AI messages

### Output Handling

1. Storage:
   ```python
   node.outputs = model_response  # Stores the execution result
   ```

2. Metadata Recording:
   ```python
   node.metadata["token_usage"] = {
       "total_tokens": count,
       "prompt_tokens": prompt_count,
       "completion_tokens": completion_count,
       "tokens_per_second": rate,
       "time_to_first_token": ttft
   }
   ```

3. Child Node Triggering:
   - Output becomes available to child nodes
   - Network triggers child node execution if configured

## Usage Examples

### Basic Usage

```python
from lc_agent import RunnableNode, RunnableNetwork
from langchain_core.messages import SystemMessage, HumanMessage

# Create a simple conversation node
with RunnableNetwork(chat_model_name="gpt-4") as network:
    node = RunnableNode(inputs=[
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="What is Python?")
    ])
    
    # Execute synchronously
    result = node.invoke()
    print(result.content)
```

### Complex Network

```python
# Create a processing network
with RunnableNetwork(chat_model_name="gpt-4") as network:
    # Context node
    context = RunnableNode(inputs=[
        SystemMessage(content="You are analyzing data.")
    ])
    
    # Data processing node
    processor = RunnableNode(inputs=[
        HumanMessage(content="Analyze this data: {data}")
    ])
    
    # Result formatting node
    formatter = RunnableNode(inputs=[
        SystemMessage(content="Format the response as JSON")
    ])
    
    # Connect nodes
    context >> processor >> formatter
    
    # Execute the network
    result = await network.ainvoke({"data": "sample data"})
```

## Metadata System

### Structure

```python
metadata = {
    # Execution metrics
    "token_usage": {
        "total_tokens": int,
        "prompt_tokens": int,
        "completion_tokens": int,
        "tokens_per_second": float,
        "time_to_first_token": float,
        "elapsed_time": float
    },
    
    # Execution data
    "chat_model_input": List[Dict],  # Processed input messages
    "error": Optional[str],          # Error information
    "invoke_input": Dict,            # Original input data
    
    # Custom data
    "contribute_to_history": bool,   # History inclusion flag
    "custom_field": Any             # User-defined metadata
}
```

### Metadata Usage

```python
# Access metadata after execution
node = RunnableNode()
result = node.invoke()

# Get execution metrics
token_usage = node.metadata["token_usage"]
print(f"Total tokens: {token_usage['total_tokens']}")

# Check for errors
if "error" in node.metadata:
    print(f"Error occurred: {node.metadata['error']}")

# Access network-level metadata
network_data = node.find_metadata("network_config")
```

## Best Practices

1. **Node Design**
   - Keep nodes focused on single responsibilities
   - Use appropriate message types
   - Handle errors through metadata
   - Document custom behavior

2. **Network Integration**
   - Use context managers (`with` statements)
   - Implement lifecycle hooks when needed
   - Clean up resources in removal hooks
   - Maintain clear parent-child relationships

3. **Performance**
   - Monitor token usage through metadata
   - Use streaming for long responses
   - Implement message culling for long conversations
   - Track execution metrics

4. **Error Handling**
   - Check metadata for errors
   - Implement recovery strategies
   - Use verbose mode for debugging
   - Maintain error context in metadata

## Execution Process

### Overview

The execution process in `RunnableNode` follows a carefully orchestrated sequence of steps to handle message processing, model interaction, and state management. Both synchronous (`invoke`) and asynchronous (`ainvoke`) methods follow the same core logic with their respective execution patterns.

### Execution Steps

1. **Pre-Invocation Setup**
   ```python
   def _pre_invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig], **kwargs):
       # Prepare node for invocation
       pass
   ```
   - Initializes execution state
   - Validates input parameters
   - Sets up execution context

2. **Cached Result Check**
   ```python
   if self.invoked or self.outputs is not None:
       self.invoked = True
       return self.outputs
   ```
   - Checks if node was previously executed
   - Returns cached results if available
   - Prevents redundant execution

3. **Token Counter Setup**
   ```python
   count_tokens = CountTokensCallbackHandler()
   config = self._get_config(config, count_tokens)
   ```
   - Initializes token counting
   - Configures callback handlers
   - Sets up performance tracking

4. **Parent Node Processing**
   ```python
   parents_result = self._process_parents(input, config, **kwargs)  # or _aprocess_parents for async
   ```
   - Executes parent nodes in sequence
   - Collects parent outputs
   - Maintains execution order

5. **Input Combination**
   ```python
   chat_model_input = self._combine_inputs(input, config, parents_result, **kwargs)
   ```
   - Merges parent results
   - Processes input transformations
   - Prepares model input

6. **Chat Model Setup**
   ```python
   chat_model_name = self._get_chat_model_name(chat_model_input, input, config)
   chat_model = self._get_chat_model(chat_model_name, chat_model_input, input, config)
   ```
   - Determines model to use
   - Retrieves model instance
   - Configures model parameters

7. **Message Sanitization**
   ```python
   chat_model_input = self._sanitize_messages_for_chat_model(chat_model_input, chat_model_name, chat_model)
   ```
   - Formats messages for model
   - Handles tool messages
   - Ensures compatibility

8. **Token Management**
   ```python
   max_tokens = get_chat_model_registry().get_max_tokens(chat_model_name)
   if max_tokens is not None and tokenizer is not None:
       chat_model_input = _cull_messages(chat_model_input, max_tokens, tokenizer)
   ```
   - Checks token limits
   - Culls messages if needed
   - Maintains context window

9. **Model Execution**
   ```python
   result = self._invoke_chat_model(chat_model, chat_model_input, input, config, **kwargs)
   # or
   result = await self._ainvoke_chat_model(chat_model, chat_model_input, input, config, **kwargs)
   ```
   - Executes language model
   - Handles responses
   - Processes outputs

10. **Metadata Recording**
    ```python
    self.metadata["token_usage"] = {
        "total_tokens": count_tokens.total_tokens,
        "prompt_tokens": count_tokens.prompt_tokens,
        "completion_tokens": count_tokens.completion_tokens,
        "tokens_per_second": count_tokens.tokens_per_second_with_ttf,
        "time_to_first_token": count_tokens.time_to_first_token,
        "elapsed_time": count_tokens.elapsed_time
    }
    ```
    - Records performance metrics
    - Stores token usage
    - Tracks timing information

11. **State Update**
    ```python
    self.outputs = result
    self.invoked = True
    ```
    - Caches execution result
    - Updates node state
    - Marks completion

### Execution Methods

#### `invoke` (Synchronous)
```python
def invoke(
    self,
    input: Dict[str, Any] = {},
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> Union[List[OutputType], OutputType]
```
- Blocks until completion
- Processes steps sequentially
- Returns final result
- Suitable for simple interactions

#### `ainvoke` (Asynchronous)
```python
async def ainvoke(
    self,
    input: Dict[str, Any] = {},
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> Union[List[OutputType], OutputType]
```
- Non-blocking execution
- Allows concurrent operations
- Returns final result asynchronously
- Better for long-running operations

### Error Handling

1. **Exception Capture**
   ```python
   try:
       # Execution steps
   except BaseException as e:
       self.metadata["error"] = str(e)
       raise
   ```
   - Captures all exceptions
   - Records error in metadata
   - Preserves stack trace

2. **State Preservation**
   - Maintains node state on error
   - Allows error recovery
   - Enables debugging

### Performance Tracking

1. **Token Usage**
   - Total tokens used
   - Prompt vs completion tokens
   - Token rate metrics

2. **Timing Metrics**
   - Time to first token
   - Total execution time
   - Tokens per second

3. **Model Performance**
   - Model-specific metrics
   - Response latency
   - Throughput statistics
