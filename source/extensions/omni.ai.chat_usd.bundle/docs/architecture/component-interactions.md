# Component Interactions

This document explains how the different components of Chat USD interact with each other to provide a comprehensive USD development assistant. Understanding these interactions is crucial for extending or customizing the system.

## Overview

Chat USD consists of several key components that interact through well-defined interfaces:

1. **ChatUSDNetworkNode**: The main entry point for user interactions
2. **ChatUSDSupervisorNode**: The orchestrator of the multi-agent system
3. **Specialized Agents**:
   - **USDCodeInteractiveNetworkNode**: For USD code generation and execution
   - **USDSearchNetworkNode**: For USD asset search
   - **SceneInfoNetworkNode**: For scene information retrieval

These components interact through message passing, with each component processing messages and passing them to the next component in the chain.

## Interaction Flow

The interaction flow in Chat USD follows these steps:

1. **User Input**: The user provides a query through the chat interface
2. **ChatUSDNetworkNode Processing**: The query is processed by the ChatUSDNetworkNode
3. **Supervisor Analysis**: The ChatUSDSupervisorNode analyzes the query to determine its intent
4. **Agent Selection**: The supervisor selects the appropriate specialized agent
5. **Agent Processing**: The selected agent processes the query
6. **Response Integration**: The supervisor integrates the agent's response
7. **Final Response**: The integrated response is returned to the user

This flow ensures that each query is handled by the most appropriate agent and that the response is coherent and comprehensive.

## Message Passing

Components in Chat USD interact through message passing, using the message types defined in the LC Agent framework:

1. **HumanMessage**: Represents user input
2. **AIMessage**: Represents AI-generated responses
3. **SystemMessage**: Provides instructions to the AI
4. **ToolMessage**: Represents the output of tool calls

Messages flow through the system as follows:

```text
User Input (HumanMessage)
       ↓
ChatUSDNetworkNode
       ↓
ChatUSDSupervisorNode (SystemMessage + HumanMessage)
       ↓
Specialized Agent (SystemMessage + HumanMessage)
       ↓
Agent Response (AIMessage)
       ↓
ChatUSDSupervisorNode (AIMessage)
       ↓
Final Response (AIMessage)
```

This message flow ensures that each component has the information it needs to process the query effectively.

## Component Registration

Components in Chat USD are registered with the node factory, which makes them available for use in the system:

```python
get_node_factory().register(
    USDCodeInteractiveNetworkNode,
    name="ChatUSD_USDCodeInteractive",
    scene_info=False,
    enable_code_interpreter=enable_code_interpreter,
    code_interpreter_hide_items=code_interpreter_hide_items,
    enable_code_atlas=need_rags,
    enable_metafunctions=need_rags,
    enable_interpreter_undo_stack=True,
    max_retries=1,
    enable_code_promoting=True,
    hidden=True,
)
```

This registration process configures each component with the appropriate parameters and makes it available for use by other components.

## Supervisor-Agent Interactions

The interaction between the supervisor and specialized agents is a key aspect of Chat USD. The supervisor:

1. **Analyzes Queries**: Determines the intent of user queries
2. **Selects Agents**: Chooses the appropriate agent for each query
3. **Reformulates Queries**: Adapts queries for the selected agent
4. **Integrates Responses**: Combines responses from multiple agents

These interactions are guided by the system message provided to the supervisor, which contains detailed instructions on how to interact with each agent.

## Agent-Agent Interactions

In some cases, agents may need to interact with each other to provide a comprehensive response. For example:

1. **SceneInfo → USDCodeInteractive**: The SceneInfo agent provides scene information that the USDCodeInteractive agent uses to generate code
2. **USDSearch → USDCodeInteractive**: The USDSearch agent provides asset information that the USDCodeInteractive agent uses to import assets

These interactions are coordinated by the supervisor, which routes queries and responses between agents as needed.

## Modifier Interactions

Modifiers play a crucial role in extending the functionality of Chat USD components. They interact with components by:

1. **Intercepting Messages**: Modifiers can intercept messages before they are processed by a component
2. **Modifying Behavior**: Modifiers can change how a component processes messages
3. **Enhancing Responses**: Modifiers can add information to component responses

For example, the `USDSearchModifier` enhances the USDSearchNetworkNode by intercepting search queries and calling the USD Search API.

Pseudocode:

```python
def on_post_invoke(self, network: "RunnableNetwork", node: RunnableNode):
    output = node.outputs.content if node.outputs else ""
    matches = re.findall(r'@USDSearch\("(.*?)", (.*?), (\d+)\)@', output)

    search_results = {}
    for query, metadata, limit in matches:
        # Cast to proper Python types
        metadata = metadata.lower() == "true"
        limit = int(limit)

        # Call the actual USD Search API
        api_response = self.usd_search_post(query, metadata, limit)
        search_results[query] = api_response
```

## Extension Integration

Chat USD integrates with Omniverse Kit through the extension system. The `ChatUSDBundleExtension` class:

1. **Registers Components**: Registers Chat USD components with the node factory
2. **Configures Settings**: Sets up configuration settings for Chat USD
3. **Handles Lifecycle Events**: Manages startup and shutdown events

This integration ensures that Chat USD is properly initialized and configured when the extension is loaded.

## UI Interactions

Chat USD can interact with the UI through specialized components like `ChatUSDWithOmniUINetworkNode`, which extends the base Chat USD functionality to include UI generation:

```python
class ChatUSDWithOmniUINetworkNode(ChatUSDNetworkNode):
    """
    ChatUSDNetworkNode is a specialized network node designed to handle conversations related to USD (Universal Scene Description).
    It utilizes the ChatUSDNodeModifier to dynamically modify the scene or search for USD assets based on the conversation's context.

    This class is an example of how to implement a multi-agent system where different tasks are handled by specialized agents (nodes)
    based on the user's input.
    """

    default_node: str = "ChatUSDWithOmniUISupervisorNode"
    route_nodes: List[str] = [
        "ChatUSD_USDCode",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
        "OmniUI_Code",
    ]
```

This component adds the `OmniUI_Code` agent to the route nodes, allowing Chat USD to generate UI code in addition to USD code.

## Error Handling

Error handling in Chat USD is managed through a combination of component interactions and modifiers. When an error occurs:

1. **Error Detection**: The component where the error occurred detects the error
2. **Error Reporting**: The error is reported through the component's metadata
3. **Error Handling**: Modifiers or other components handle the error

For example, the `DoubleRunUSDCodeGenInterpreterModifier` handles errors in code execution by attempting to fix the code and run it again.
