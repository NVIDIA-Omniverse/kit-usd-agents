# MultiAgentNetworkNode

## Overview

MultiAgentNetworkNode is a specialized network node within the LC-Agent framework designed to support a multi-agent conversational system. It extends the basic network functionality by introducing routing and coordination among several specialized "agents" (sub-networks). Unlike a simple network node, it acts as a conversation coordinator that intelligently routes user queries to appropriate specialized agents based on query context and content.

## Motivation & Use Cases

### 1. Complex Domain Problems
When your application needs to handle queries that span multiple domains or require different types of expertise. For example:

- **Documentation Systems (Doc Atlas)**:
  - Routing between course content, API documentation, and examples
  - Handling both high-level concepts and detailed implementation questions
  - Managing different documentation formats and sources

- **Educational Systems (USD Tutor)**:
  ```python
  class USDTutorNetworkNode(MultiAgentNetworkNode):
      route_nodes: List[str] = [
          "USDTutor_Course",     # Course content
          "USDTutor_Code",       # Code examples
          "USDTutor_Knowledge",  # General knowledge
      ]
  ```
  - Separating course content from code generation
  - Providing both theoretical knowledge and practical examples
  - Managing progressive learning paths

### 2. Specialized Processing
When different types of queries require different processing pipelines:

```python
# Example from USD Tutor
def register_usd_tutor_agent():
    # Register specialized nodes for different tasks
    get_node_factory().register(
        USDCourseNetworkNode,    # Handles course content
        name="USDTutor_Course",
        hidden=True,
    )
    get_node_factory().register(
        USDCodeGenNetworkNode,   # Handles code generation
        name="USDTutor_Code",
        hidden=True,
    )
```

### 3. Modular System Design
When you need to:
- Add or remove capabilities without affecting the core system
- Maintain specialized components independently
- Scale different components based on demand

## Architecture & Components

### 1. Core Properties

```python
class MultiAgentNetworkNode(NetworkNode):
    default_node: str = ""              # Fallback node type
    route_nodes: List[str] = []         # Available specialized agents
    multishot: bool = True              # Allow multiple interactions
    function_calling: bool = True       # Use function calling for routing
    classification_node: bool = True     # Use classification for routing
    generate_prompt_per_agent: bool = True  # Generate specific prompts
```

### 2. Identity System
The multi-agent system requires a clear identity and routing instructions. Example from USD Tutor:

```python
# Identity setup in usd_tutor_identity.md
first_routing_instruction = (
    '(Reminder to respond in one line only. Format: "<tool_name> <question>". '
    "The only available tools are: USDTutor_Course, USDTutor_Code, USDTutor_Knowledge)"
)

subsequent_routing_instruction = (
    '(Reminder: Either respond with "<tool_name> <question>" OR "FINAL <answer>". '
    'For FINAL response, provide comprehensive explanation including all tool responses.)'
)
```

### 3. Specialized Agents
Each route node represents a specialized agent:

```python
# Example from USD Course Network Node
class USDCourseNetworkNode(DocAtlasNetworkNode):
    """Specialized teaching assistant for OpenUSD curriculum"""
    default_node = "RunnableNode"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Configure with specific course materials
        self.metadata["description"] = "Educational content provider"
```

## Supervisor Node and Message Hierarchy

The MultiAgentNetworkNode implements a hierarchical system where a supervisor node orchestrates the interaction between specialized sub-nodes. This hierarchy is crucial for maintaining coherent conversations and ensuring proper information flow between components.

### Supervisor Node

The supervisor node, specified by the `default_node` property, serves as the central coordinator for the multi-agent system. It has several key responsibilities:

1. **Conversation Management**
   The supervisor maintains the overall conversation context and makes high-level decisions about routing. It acts as both the entry point for user queries and the final response synthesizer.

   ```python
   class USDTutorSupervisorNode(RunnableNode):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Add the supervisor's system message
           self.inputs.append(RunnableSystemAppend(
               system_message=identity
           ))
   ```

2. **System Message Hierarchy**
   Each node in the system can maintain its own system message, creating a layered approach to conversation:

   - The supervisor node's system message defines the overall behavior and routing rules
   - Sub-nodes can have their own specialized system messages for their specific tasks
   - Messages don't interfere with each other, allowing specialized behavior at each level

   For example, in the USD Tutor system:
   ```python
   # Supervisor level - defines routing and integration
   supervisor_message = """
   You are a USD tutor coordinator. Route queries to:
   - Course content (educational materials)
   - Code examples (implementation details)
   - Knowledge base (conceptual explanations)
   """

   # Specialized node level - focuses on specific task
   course_node_message = """
   You are a USD course content expert. Provide
   information specifically from course materials.
   """
   ```

3. **Visibility and Scope**
   The multi-agent system implements a carefully controlled visibility system:

   - **Supervisor Visibility**: The supervisor node has access to all responses from sub-nodes but maintains separation between them. This allows it to:
     - Synthesize comprehensive responses
     - Maintain conversation context
     - Make informed routing decisions
     - Prevent information leakage between sub-nodes

   - **Sub-node Visibility**: Each specialized node:
     - Only sees queries routed specifically to it
     - Has access to its own context and history
     - Cannot directly access other nodes' states or responses
     - Maintains focus on its specialized task

### Node Interaction Patterns

The interaction between supervisor and specialized nodes follows specific patterns:

1. **Query Flow**
   ```
   User Query → Supervisor Node → Classification → Specialized Node → Response → Supervisor → Final Answer
   ```

2. **Context Management**
   - The supervisor maintains the global conversation context
   - Each sub-node maintains its specialized context
   - Responses are integrated at the supervisor level

3. **Model Selection**
   Different nodes can use different language models:
   ```python
   class CourseNode(NetworkNode):
       def __init__(self):
           super().__init__(chat_model_name="gpt-4")  # Specialized model

   class CodeNode(NetworkNode):
       def __init__(self):
           super().__init__(chat_model_name="code-llama")  # Code-specific model
   ```

### Implementation Example

Here's a complete example showing the hierarchy and interaction:

```python
class TutorSystem(MultiAgentNetworkNode):
    def __init__(self):
        super().__init__()
        
        # Configure supervisor
        self.default_node = "SupervisorNode"
        self.route_nodes = ["Course", "Code", "Knowledge"]
```

### Best Practices for Node Hierarchy

1. **System Message Design**
   - Keep supervisor messages focused on routing and integration
   - Make specialized node messages task-specific
   - Avoid overlapping responsibilities
   - Maintain clear boundaries between nodes

2. **Model Selection**
   - Choose appropriate models for each node's task
   - Consider computational resources
   - Balance capability with efficiency
   - Use consistent models for related tasks

3. **Context Management**
   - Implement proper context isolation
   - Maintain clear data flow paths
   - Handle state transitions carefully
   - Monitor context size and performance

4. **Response Integration**
   - Implement clear response synthesis rules
   - Maintain conversation coherence
   - Handle conflicts between sub-nodes
   - Provide clear attribution in responses

## Advanced Features

### 1. Classification-based Routing
When `classification_node=True`:
- Uses a dedicated node to classify queries
- Routes based on semantic understanding
- Maintains conversation context

### 2. Function Calling Mode
When `function_calling=True`:
- Treats routing as function calls
- Provides structured routing decisions
- Enables tool-like interaction patterns

### 3. Multishot Processing
When `multishot=True`:
- Allows multiple agent interactions
- Supports follow-up questions
- Maintains conversation state

## Best Practices

### 1. Agent Design
- Keep agents focused on specific tasks
- Provide clear agent descriptions
- Implement proper metadata handling

### 2. Routing Strategy
- Use clear routing instructions
- Implement proper classification
- Handle edge cases gracefully

### 3. Response Integration
- Synthesize responses comprehensively
- Maintain context across agents
- Provide clear final answers
