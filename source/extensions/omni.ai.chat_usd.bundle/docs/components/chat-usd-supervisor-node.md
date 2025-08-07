# ChatUSDSupervisorNode

The `ChatUSDSupervisorNode` is the orchestrator of the multi-agent system in Chat USD. It extends the `RunnableNode` class from the LC Agent framework and is responsible for analyzing user queries and routing them to the appropriate specialized agent.

## Overview

The `ChatUSDSupervisorNode` serves as the brain of the Chat USD system, making decisions about which specialized agent should handle each user query. It uses a system message to guide its behavior, providing detailed instructions on how to analyze queries and route them to the appropriate agent.

## Implementation

The `ChatUSDSupervisorNode` is implemented as a Python class that extends `RunnableNode`:

```python
class ChatUSDSupervisorNode(RunnableNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs.append(RunnableSystemAppend(system_message=identity))

    def _sanitize_messages_for_chat_model(self, messages, chat_model_name, chat_model):
        """Sanitizes messages and adds metafunction expert type for USD operations."""
        messages = super()._sanitize_messages_for_chat_model(messages, chat_model_name, chat_model)
        return sanitize_messages_with_expert_type(messages, "knowledge", rag_max_tokens=0, rag_top_k=0)
```

The class initializes itself with a system message (`identity`) that provides detailed instructions on how to analyze queries and route them to the appropriate agent. It also overrides the `_sanitize_messages_for_chat_model` method to add metafunction expert type information to the messages.

## System Message

The system message (`identity`) is a critical component of the `ChatUSDSupervisorNode`. It provides detailed instructions on:

1. **Available Expert Functions**: Describes the capabilities of each specialized agent
2. **Function Calling Guidelines**: Explains how to call each function effectively
3. **Scene Operation Guidelines**: Provides guidance on scene-related operations
4. **Information Gathering Guidelines**: Explains how to gather scene information
5. **Code Integration Guidelines**: Describes how to integrate code from different agents

Here's an excerpt from the system message:

```markdown
You are an expert code orchestrator, specialized in coordinating multiple AI functions to create comprehensive software solutions. Your role is to break down user requests into specific tasks and delegate them to specialized functions, each with their distinct expertise:

# Available Expert Functions:

1. ChatUSD_USDCodeInteractive
   - Expert in USD (Universal Scene Description) implementation
   - Generates USD-specific code

2. ChatUSD_USDSearch
   - Specialized in searching and querying USD data
   - Provides USD-related information
   - Does not generate implementation code

3. ChatUSD_SceneInfo [CRITICAL FOR SCENE OPERATIONS]
   - Maintains current scene state knowledge
   - Must be consulted FIRST for any scene manipulation tasks
   - Required for:
     * Any operation where prim name is not explicitly provided
     * Any attribute manipulation without explicit values
     * Operations requiring knowledge of:
       - Prim existence or location
       - Prim properties (size, position, rotation, scale)
       - Prim hierarchy
       - Prim type or nature
       - Current attribute values
       - Scene structure
       - Available materials
       - Relationship between prims
       - Bounds or extents
       - Layer structure
       - Stage metadata
   - Provides scene context for other functions
   - Should be used before USD code generation for scene operations
   - Cannot generate complex code but provides essential scene data
```

This system message is designed to ensure that the supervisor makes appropriate routing decisions and provides clear instructions to the specialized agents.

## Query Analysis

The `ChatUSDSupervisorNode` analyzes user queries to determine their intent and which specialized agent should handle them. This analysis is guided by the system message, which provides detailed instructions on how to identify different types of queries.

For example, the system message instructs the supervisor to route scene-related queries to the `ChatUSD_SceneInfo` agent first, to gather information about the scene before generating code:

```markdown
# Scene Operation:

1. ALWAYS query ChatUSD_SceneInfo first when:
   - User doesn't provide complete prim information
   - Task involves existing scene elements
   - Operation requires current state knowledge
   - Manipulation of relative values is needed
   - Working with hierarchical relationships
   - Checking for validity of operations

2. Information Flow:
   ChatUSD_SceneInfo -> ChatUSD_USDCodeInteractive
   - ChatUSD_SceneInfo must provide context before code generation
   - All scene-dependent values must be validated
```

This ensures that the supervisor has all the necessary information before generating code, leading to more accurate and effective responses.

## Agent Selection

Based on its analysis, the `ChatUSDSupervisorNode` selects the appropriate specialized agent to handle the user's query. The selection is based on the intent of the query and the capabilities of each agent, as described in the system message.

For example:

- For USD code generation queries, the supervisor selects the `ChatUSD_USDCodeInteractive` agent
- For USD asset search queries, the supervisor selects the `ChatUSD_USDSearch` agent
- For scene information queries, the supervisor selects the `ChatUSD_SceneInfo` agent

This selection process ensures that each query is handled by the most appropriate agent, leading to more accurate and effective responses.

## Query Reformulation

After selecting the appropriate agent, the `ChatUSDSupervisorNode` reformulates the user's query to match the expectations of the selected agent. This reformulation is guided by the system message, which provides detailed instructions on how to format queries for each agent.

For example, the system message provides guidelines for formatting queries for the `ChatUSD_SceneInfo` agent:

```markdown
3. ChatUSD_SceneInfo MUST ALWAYS print the prim name related to the information it collects
    Wrong ChatUSD_SceneInfo prompt:
    - Get the sphere position in the current USD stage.
    Good ChatUSD_SceneInfo prompt:
    - Get the sphere prim path and its position in the current USD stage.
```

This ensures that the agent receives a well-formatted query that it can process effectively.

## Response Integration

After the specialized agent processes the query and returns a response, the `ChatUSDSupervisorNode` integrates the response into a coherent reply. This integration may involve combining information from multiple agents, resolving conflicts, and ensuring that the response is coherent and comprehensive.

The system message provides guidelines for this integration process:

```markdown
# Code Integration:

1. Separate Concerns:
   - ChatUSD_USDCodeInteractive should provide pure USD manipulation code
   - ChatUSD_SceneInfo provides correct prim paths and validation

2. Integration Example:
   For "Move the sphere up":

   a) ChatUSD_SceneInfo Query:
      "Get the sphere prim path and its position in the current USD stage"

      Result:
      Prim: /World/Sphere, Position: (0.0, 0.0, 0.0)

   b) ChatUSD_USDCodeInteractive Query:
      "Sets the vertical position of the prim /World/Sphere"
```

This ensures that the final response is coherent, comprehensive, and effective.

## Variant: ChatUSDWithOmniUISupervisorNode

The `ChatUSDSupervisorNode` has a variant called `ChatUSDWithOmniUISupervisorNode`, which is used in the omni.ui variant of Chat USD:

```python
class ChatUSDWithOmniUISupervisorNode(RunnableNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs.append(RunnableSystemAppend(system_message=identity_omniui))

    def _sanitize_messages_for_chat_model(self, messages, chat_model_name, chat_model):
        """Sanitizes messages and adds metafunction expert type for USD operations."""
        messages = super()._sanitize_messages_for_chat_model(messages, chat_model_name, chat_model)
        return sanitize_messages_with_expert_type(messages, "knowledge", rag_max_tokens=0, rag_top_k=0)
```

This variant uses a different system message (`identity_omniui`) that includes instructions for the `OmniUI_Code` agent, which handles UI generation.
