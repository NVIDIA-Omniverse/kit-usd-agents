# ChatUSDNetworkNode

The `ChatUSDNetworkNode` is the main entry point for user interactions in the Chat USD system. It extends the `MultiAgentNetworkNode` class from the LC Agent framework and is responsible for routing user queries to specialized agents based on their intent.

## Overview

The `ChatUSDNetworkNode` serves as the central coordinator for the Chat USD system, managing the flow of messages between the user and specialized agents. It uses a supervisor node to analyze user queries and route them to the appropriate specialized agent, then integrates the responses into a coherent reply.

## Implementation

The `ChatUSDNetworkNode` is implemented as a Python class that extends `MultiAgentNetworkNode`:

```python
class ChatUSDNetworkNode(MultiAgentNetworkNode):
    """
    ChatUSDNetworkNode is a specialized network node designed to handle conversations related to USD (Universal Scene Description).
    It utilizes the ChatUSDNodeModifier to dynamically modify the scene or search for USD assets based on the conversation's context.

    This class is an example of how to implement a multi-agent system where different tasks are handled by specialized agents (nodes)
    based on the user's input.
    """

    default_node: str = "ChatUSDSupervisorNode"
    route_nodes: List[str] = [
        "ChatUSD_USDCodeInteractive",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
    ]
    function_calling = False
    generate_prompt_per_agent = True
    multishot = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metadata["description"] = "Multi-agents to modify the scene and search USD assets."
        self.metadata["examples"] = [
            "Find me a traffic cone",
            "Create a sphere with a red material",
            "Find an orange and import it to the scene",
        ]
```

Key features of this class include:

1. **default_node**: Specifies the supervisor node that will analyze user queries
2. **route_nodes**: Lists the specialized agents that can handle specific tasks
3. **function_calling**: Controls whether function calling is enabled
4. **generate_prompt_per_agent**: Determines if each agent gets a custom prompt
5. **multishot**: Enables multi-turn conversations with agents

The class also sets metadata for the node, including a description and examples of queries it can handle.

## Configuration

The `ChatUSDNetworkNode` can be configured with various parameters to customize its behavior:

- **chat_model_name**: The name of the chat model to use
- **multishot**: Whether to enable multi-turn conversations
- **function_calling**: Whether to enable function calling
- **generate_prompt_per_agent**: Whether to generate a custom prompt for each agent

These parameters can be set when registering the node with the node factory:

```python
get_node_factory().register(ChatUSDNetworkNode, name="Chat USD", multishot=chat_usd_multishot)
```

## Routing Process

The routing process in `ChatUSDNetworkNode` follows these steps:

1. **User Query Analysis**: The `ChatUSDSupervisorNode` analyzes the user's query to determine its intent
2. **Agent Selection**: Based on the intent, the supervisor selects the appropriate specialized agent
3. **Query Reformulation**: The supervisor reformulates the query for the selected agent
4. **Agent Execution**: The selected agent processes the reformulated query
5. **Response Integration**: The supervisor integrates the agent's response into a coherent reply
6. **Final Response**: The integrated response is returned to the user

This process ensures that each query is handled by the most appropriate agent and that the response is coherent and comprehensive.

## Specialized Agents

The `ChatUSDNetworkNode` routes queries to the following specialized agents:

1. **ChatUSD_USDCodeInteractive**: Handles USD code generation and execution
2. **ChatUSD_USDSearch**: Handles USD asset search
3. **ChatUSD_SceneInfo**: Handles scene information retrieval

Each agent is registered with the node factory and configured with specific parameters:

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

## UI Integration

The `ChatUSDNetworkNode` integrates with the Omniverse Kit UI through the `ChatView` class, which provides a chat interface for interacting with Chat USD:

```python
try:
    from omni.ai.langchain.widget.core import ChatView

    from .chat.chat_usd_network_node_delegate import ChatUSDNetworkNodeDelegate
    from .chat.multi_agent_delegate import SupervisorNodeDelegate, ToolNodeDelegate

    ChatView.add_delegate("ChatUSDNetworkNode", ChatUSDNetworkNodeDelegate())
    ChatView.add_delegate("RunnableSupervisorNode", SupervisorNodeDelegate())
    ChatView.add_delegate("RunnableToolNode", ToolNodeDelegate())

except ImportError:
    # this extension is not available in the current environment
    pass
```

This integration allows users to interact with Chat USD through a user-friendly chat interface.

## Extension: ChatUSDWithOmniUINetworkNode

The `ChatUSDNetworkNode` can be extended to add new functionality. For example, the `ChatUSDWithOmniUINetworkNode` extends `ChatUSDNetworkNode` to add UI generation capabilities:

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

    first_routing_instruction = (
        '(Reminder to respond in one line only. Format: "<tool_name> <question>". '
        "The only available options are: {options})"
    )

    subsequent_routing_instruction = (
        '(Reminder to respond in one line only. Either "<tool_name> <question>" or "FINAL <answer>". '
        'If there is answer, respond with "FINAL <code>". The final answer should contain the final code no matter what. '
        "Remember to use UI code from OmniUI_Code and USD code from ChatUSD_USDCode. "
        "Never use UI code from ChatUSD_USDCode and ChatUSD_SceneInfo. "
        "Never use USD code from OmniUI_Code. If OmniUI_Code provides USD code, ask ChatUSD_USDCode to redo the USD code. )"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_modifier(RunScriptModifier())

        self.metadata["description"] = "Chat USD Agent can modify the scene and create UI elements using omni.ui"
        self.metadata["examples"] = [
            "Create a window with a slider that moves the sphere\nin the current USD stage up and down",
            "Create a window with a button that creates a red sphere",
        ]
```

This extension adds the `OmniUI_Code` agent to the route nodes, allowing Chat USD to generate UI code in addition to USD code. It also adds a `RunScriptModifier` to automatically run the generated code.
