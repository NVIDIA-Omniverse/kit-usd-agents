# How LC-Agent Works Inside AIQ - Multi-Agent Guide

This guide explains how LC-Agent networks integrate with AIQ to create multi-agent systems in Omniverse. The integration happens through the `lc_agent_aiq` bridge module that enables bidirectional usage between AIQ and LC-Agent.

## The Two-Way Bridge Between AIQ and LC-Agent

The bridge allows developers to use AIQ workflows inside LC-Agent networks and expose LC-Agent networks as AIQ functions. This bidirectional integration is achieved through two main components:

### Integration Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│           RunnableAIQNode Execution Flow in Stage Builder          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  RunnableAIQNode(aiq_config)                                       │
│       │                                                            │
│       └─► AIQWrapper                                               │
│               │                                                    │
│               ├─► convert_langchain_to_aiq_messages()              │
│               │                                                    │
│               └─► WorkflowBuilder.from_config(config)              │
│                       │                                            │
│                       └─► Build workflow (_type: MultiAgent)       │
│                               │                                    │
│                               └─► MultiAgentNetworkFunction        │
│                                       │                            │
│       ┌───────────────────────────────┴─────────────────┐          │
│       │                                                 │          │
│       └─► pre_invoke() registers all functions:         │          │
│               │                                         │          │
│               ├─► planning (LCAgentFunction)            │          │
│               │    └─► PlanningNetworkNode              │          │
│               │                                         │          │
│               ├─► ChatUSD_USDCodeInteractive            │          │
│               │    └─► USDCodeInteractiveNetworkNode    │          │
│               │                                         │          │
│               ├─► ChatUSD_SceneInfo                     │          │
│               │    └─► SceneInfoNetworkNode             │          │
│               │                                         │          │
│               └─► ChatUSD_USDSearch                     │          │
│                    └─► USDSearchNetworkNode             │          │
│                                                         │          │
│       Then creates supervisor:                          │          │
│       └─► MultiAgentNetworkNode                         │          │
│               │                                         │          │
│               └─► Routes to registered functions        │          │
│                                                         │          │
└─────────────────────────────────────────────────────────┘──────────┘
```

The diagram shows the actual execution flow when using the Stage Builder extension. The flow begins when a user interacts with the Chat USD panel in Omniverse Kit, which triggers the RunnableAIQNode registered by the extension.

**Key Points in the Execution Flow:**

- **Extension Registration**: The Stage Builder extension (`extension.py`) loads the workflow configuration from `workflow.yaml` and registers a `RunnableAIQNode` with the LC-Agent node factory. This happens once during extension startup.

- **AIQWrapper as Bridge**: When the node is invoked, it creates an `AIQWrapper` that acts as a LangChain-compatible chat model. This wrapper is the critical bridge that allows AIQ workflows to run inside LC-Agent contexts.

- **Message Conversion**: The wrapper converts LC-Agent messages (HumanMessage, AIMessage, etc.) to AIQ format using `convert_langchain_to_aiq_messages()`. This ensures compatibility between the two message formats.

- **Workflow Building**: The `WorkflowBuilder.from_config()` creates the AIQ workflow. Since the configuration specifies `_type: MultiAgent`, it instantiates a `MultiAgentNetworkFunction`.

- **Dynamic Function Registration**: The `MultiAgentNetworkFunction.pre_invoke()` method (from `multi_agent_network_function.py`) loops through all configured tool names:
  - For LC-Agent functions like `planning`, it calls their `pre_invoke()` to self-register
  - For native AIQ functions, it wraps them in `FunctionRunnableNode`
  - All functions become available as route nodes for the supervisor

- **Supervisor Creation**: Finally, a `MultiAgentNetworkNode` is created with the registered functions as `route_nodes`. This supervisor uses either classification or function calling (configured in workflow.yaml) to route user requests to the appropriate specialized agent.

### 1. RunnableAIQNode - Running AIQ Inside LC-Agent

`RunnableAIQNode` lets you execute entire AIQ workflows within LC-Agent networks. This is useful when you want to leverage AIQ's workflow capabilities inside an LC-Agent conversation flow.

The node works by creating an `AIQWrapper` that acts as a chat model:
- Converts LC-Agent messages to AIQ format
- Runs the AIQ workflow through `WorkflowBuilder`
- Streams or returns results back to LC-Agent
- Captures any child networks created during execution

Example usage from `runnable-example.py`:
```python
aiq_config = {
    "workflow": {
        "_type": "lc_agent_simple",
        "system_message": "You are a dog. Answer like a dog. Woof woof!",
    }
}

with RunnableNetwork(chat_model_name="P-GPT4o") as network:
    RunnableHumanNode("Who are you?")
    RunnableAIQNode(aiq_config=aiq_config)
```

### 2. LCAgentFunction - Exposing LC-Agent as AIQ Functions

`LCAgentFunction` wraps LC-Agent networks to make them available as AIQ functions. This is the more common integration pattern, allowing you to build complex stateful agents with LC-Agent and expose them through AIQ's workflow system.

The function returns an AIQ-compatible function that:
- Registers LC-Agent nodes with the node factory when invoked
- Creates and manages `RunnableNetwork` instances
- Handles message conversion between formats
- Cleans up registrations after execution

## Registering LC-Agent Networks as AIQ Functions

When LC-Agent networks need to be exposed as AIQ functions, the flow works in the opposite direction. This pattern is more common because it allows developers to build complex, stateful agents using LC-Agent's powerful network architecture while exposing them through AIQ's workflow system.

```
┌─────────────────────────────────────────────────────────────────────┐
│            LC-Agent Network Registration as AIQ Function            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  @register_function(config_type=PlanningConfig)                     │
│  async def planning_function(config, builder):                      │
│       │                                                             │
│       └─► yield LCAgentFunction(                                    │
│               config=config,                                        │
│               builder=builder,                                      │
│               lc_agent_node_type=PlanningNetworkNode,               │
│               lc_agent_node_gen_type=PlanningGenNode                │
│           )                                                         │
│               │                                                     │
│               └─► When AIQ calls this function:                     │
│                       │                                             │
│                       ├─► pre_invoke()                              │
│                       │    ├─► Register PlanningNetworkNode         │
│                       │    ├─► Register PlanningGenNode             │
│                       │    └─► Register chat models                 │
│                       │                                             │
│                       ├─► _ainvoke() or _astream()                  │
│                       │    ├─► Create RunnableNetwork               │
│                       │    ├─► Add nodes from AIQ messages          │
│                       │    ├─► Execute network                      │
│                       │    └─► Convert results to AIQ format        │
│                       │                                             │
│                       └─► post_invoke()                             │
│                            └─► Unregister nodes and models          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Understanding the Registration Process:**

The `@register_function` decorator is AIQ's discovery mechanism. When AIQ scans for available functions, it finds these decorated functions and makes them available in workflows. The critical part is that the function doesn't directly implement the logic - instead, it yields an `LCAgentFunction` instance that acts as a bridge.

**The LCAgentFunction Bridge:**

`LCAgentFunction` (from `lc_agent_function.py`) is a sophisticated bridge that:
- Inherits from AIQ's `Function` class to be AIQ-compatible
- Manages the lifecycle of LC-Agent nodes through dynamic registration
- Handles message format conversion between AIQ and LC-Agent
- Supports both streaming and non-streaming execution modes

**Dynamic Registration Pattern:**

The key insight is that LC-Agent nodes are not permanently registered. Instead:
- `pre_invoke()` registers nodes only when the function is about to be used
- `post_invoke()` unregisters them after execution completes
- This prevents namespace pollution and allows concurrent executions without conflicts

To make an LC-Agent network available in AIQ, you use the `@register_function` decorator with `LCAgentFunction`. Here's how the planning agent is registered in `omni_aiq_planning/register.py`:

```python
@register_function(config_type=PlanningConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def planning_function(config: PlanningConfig, builder: Builder):
    from omni_aiq_planning.nodes.planning_node import PlanningGenNode, PlanningNetworkNode

    # Create extended node with system message
    class ExtraPlanningGenNode(PlanningGenNode):
        def __init__(self, **kwargs):
            kwargs["short_plan"] = config.short_plan
            super().__init__(**kwargs)
            if system_message:
                self.inputs.append(RunnableSystemAppend(system_message=system_message))

    # Return LCAgentFunction that wraps the LC-Agent network
    yield LCAgentFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=PlanningNetworkNode,      # NetworkNode class
        lc_agent_node_gen_type=ExtraPlanningGenNode, # Optional generator node
        hidden=True,
    )
```

The registration process involves several steps:
1. AIQ discovers the function through the `@register_function` decorator
2. When the workflow needs this function, AIQ calls it with config and builder
3. The function yields an `LCAgentFunction` instance that AIQ can invoke
4. During execution, `LCAgentFunction` dynamically registers the LC-Agent nodes

## Registering Native AIQ Functions

Not all functions need LC-Agent's stateful networks. Simple operations can be registered as native AIQ functions. From `register.py`, here's a simple addition function:

```python
@register_function(config_type=AddFunctionConfig)
async def register_my_add(config: AddFunctionConfig, builder: Builder):
    async def my_add_direct(value: str) -> int:
        return str(eval(value))

    yield FunctionInfo.create(
        single_fn=my_add_direct,
        description="A simple function that adds two integers together",
    )
```

Native AIQ functions are simpler - they just process input and return output without maintaining conversation state or using language models.

## How MultiAgent Orchestration Works

`MultiAgentNetworkFunction` is a specialized version of `LCAgentFunction` that coordinates multiple sub-agents. It dynamically registers all configured functions before creating the supervisor network.

### MultiAgent Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│              MultiAgent Orchestration (from source code)                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  workflow.yaml:                                                          │
│    workflow:                                                             │
│      _type: MultiAgent                                                   │
│      tool_names: [planning, ChatUSD_USDCodeInteractive, ...]             │
│       │                                                                  │
│       └─► AIQ calls multi_agent_router_function()                        │
│               │                                                          │
│               └─► yield MultiAgentNetworkFunction()                      │
│                       │                                                  │
│                       └─► pre_invoke() loops through tool_names:         │
│                               │                                          │
│                               for name in function_names:                │
│                                   function = builder.get_function(name)  │
│                                   if isinstance(function, LCAgentFunction):
│                                       await function.pre_invoke()        │
│                                   else:                                  │
│                                       register FunctionRunnableNode      │
│                               │                                          │
│                               └─► Creates MultiAgentNetworkNode with:    │
│                                   - route_nodes = filtered_function_names│
│                                   - Supervisor system message            │
│                                           │                              │
│                ┌──────────────────────────┴─────────────────────┐        │
│                │                                                │        │
│  MultiAgentNetworkNode routes based on:                         │        │
│    - classification_node = true (default)                       │        │
│    - function_calling = false (can be configured)               │        │
│    - multishot = true (allows follow-ups)                       │        │
│                │                                                │        │
│                └─► Routes to appropriate registered function    │        │
│                                                                 │        │
└─────────────────────────────────────────────────────────────────┘────────┘
```

The multi-agent system demonstrates the power of combining AIQ's workflow orchestration with LC-Agent's network execution. The `MultiAgentNetworkFunction` serves as a specialized orchestrator that understands how to register different types of functions.

**How the Orchestration Works:**

The `MultiAgentNetworkFunction` extends `LCAgentFunction` with multi-agent capabilities. During its `pre_invoke()` phase, it:
- Retrieves all configured tool names from the workflow configuration
- Iterates through each function name and gets the function instance from AIQ's builder
- Determines the function type and handles registration accordingly:
  - LC-Agent functions (`LCAgentFunction` instances) are allowed to self-register via their own `pre_invoke()`
  - Native AIQ functions are wrapped in `FunctionRunnableNode` for LC-Agent compatibility
- Sets the filtered function names as `route_nodes` for the supervisor

**The Supervisor's Role:**

The `MultiAgentNetworkNode` supervisor (from `multi_agent_network_node.py`) is a sophisticated routing system that:
- Maintains a list of available route nodes (the registered functions)
- Uses either classification or function calling to determine which agent should handle a request
- Supports multishot conversations where multiple agents can be called in sequence
- Implements loop detection to prevent infinite routing cycles
- Provides customizable routing instructions for first and subsequent tool selections

**Configuration Options in workflow.yaml:**

- `classification_node`: When true, uses a classification approach for routing
- `function_calling`: When false, uses natural language classification instead of structured function calls
- `multishot`: Allows multiple agent interactions in a single conversation
- `generate_prompt_per_agent`: Creates agent-specific prompts for better routing decisions

From `multi_agent_register.py`:
```python
class MultiAgentNetworkFunction(LCAgentFunction):
    async def pre_invoke(self, value: AIQChatRequest) -> None:
        function_names = self.config.tool_names

        for name in function_names:
            function = self.builder.get_function(name)
            if isinstance(function, LCAgentFunction):
                # LC-Agent function registers itself
                await function.pre_invoke(value)
            elif isinstance(function, Function):
                # Native AIQ function wrapped in FunctionRunnableNode
                get_node_factory().register(
                    FunctionRunnableNode,
                    name=name,
                    function=function,
                )

        # Configure supervisor with available functions
        self.lc_agent_node_kwargs["route_nodes"] = function_names
        await super().pre_invoke(value)
```

The orchestration flow works as follows:
- User input arrives at the supervisor (MultiAgentNetworkNode)
- Supervisor analyzes the request and routes to appropriate sub-agent
- Sub-agent executes (either LC-Agent network or native AIQ function)
- Results return to supervisor for synthesis or further routing
- Supervisor provides final response or continues conversation

## Workflow Configuration in YAML

AIQ workflows are configured in YAML files that define available functions and the workflow structure. From the Stage Builder's `workflow.yaml`:

```yaml
functions:
  planning:
    _type: planning
    system_message: "{data/planning_gen_system.md}"
    short_plan: true
    add_details: false

  ChatUSD_USDCodeInteractive:
    _type: ChatUSD_USDCodeInteractive

  ChatUSD_SceneInfo:
    _type: ChatUSD_SceneInfo

  ChatUSD_USDSearch:
    _type: ChatUSD_USDSearch
    search_path: "*simready_content*"

workflow:
  _type: MultiAgent
  system_message: "{data/chat_usd_supervisor_identity.md}"
  tool_names:
    - planning
    - ChatUSD_USDCodeInteractive
    - ChatUSD_SceneInfo
    - ChatUSD_USDSearch
```

The configuration defines:
- **functions**: Available agents/tools with their configurations
- **workflow**: The orchestration strategy (MultiAgent in this case)
- **tool_names**: Functions available to the supervisor
- **system_message**: Instructions for the supervisor using file references

## Loading Workflows in Extensions

Extensions can load workflows in two ways, as shown in the example extensions:

### Method 1: Loading from YAML File (Stage Builder)

From `stage_builder/extension.py`:
```python
# Load workflow from file
workflow_path = extension_path / "data" / "workflow.yaml"
aiq_config = load_and_override_config(f"{workflow_path}", None)

# Process markdown file references
aiq_config = replace_md_file_references(aiq_config, extension_path)

# Register with node factory
get_node_factory().register(RunnableAIQNode, name="Stage Builder AIQ", aiq_config=aiq_config)
```

### Method 2: Inline Configuration (Chat USD)

From `chat_usd/extension.py`:
```python
# Define configuration directly
aiq_config = {
    "functions": {
        "ChatUSD_USDCodeInteractive": {"_type": "ChatUSD_USDCodeInteractive"},
        "ChatUSD_SceneInfo": {"_type": "ChatUSD_SceneInfo"},
    },
    "workflow": {
        "_type": "ChatUSD",
        "tool_names": ["ChatUSD_USDCodeInteractive", "ChatUSD_SceneInfo"],
    },
}

get_node_factory().register(RunnableAIQNode, name="ChatUSD AIQ", aiq_config=aiq_config)
```

## Message Flow Through the Bridge

Understanding how messages flow between AIQ and LC-Agent helps debug and extend the system:

### AIQ to LC-Agent Flow
When an AIQ workflow calls an LC-Agent function:
1. AIQ sends `AIQChatRequest` to `LCAgentFunction`
2. Messages are converted to LC-Agent format (HumanMessage, AIMessage, etc.)
3. `RunnableNetwork` is created and messages are added as nodes
4. Network executes and produces results
5. Results are converted back to AIQ format (`AIQChatResponseChunk`)

### LC-Agent to AIQ Flow
When LC-Agent uses an AIQ function through `FunctionRunnableNode`:
1. LC-Agent node extracts input from messages
2. Input is converted to match AIQ function's schema
3. AIQ function executes with converted input
4. Result is wrapped as `AIMessage` and continues in LC-Agent network

## Real-World Example: Stage Builder's Multi-Agent System

The Stage Builder extension demonstrates the full power of this integration by combining multiple specialized agents to build complex USD scenes. Each agent has a specific role defined by its implementation and system messages.

**The Agent Ecosystem:**

In the Stage Builder workflow, four specialized agents work together:

1. **Planning Agent** (`planning`):
   - Implemented as `PlanningNetworkNode` with `PlanningGenNode` for generation
   - Uses system messages from `planning_gen_system.md` to create detailed execution plans
   - Configured with `short_plan: true` for concise planning output
   - First agent typically called to architect the overall approach

2. **USD Code Interactive Agent** (`ChatUSD_USDCodeInteractive`):
   - Implemented as `USDCodeInteractiveNetworkNode`
   - Executes USD Python code to create and modify scene elements
   - Has access to metafunctions for common USD operations
   - Handles the actual scene modifications based on the plan

3. **Scene Info Agent** (`ChatUSD_SceneInfo`):
   - Implemented as `SceneInfoNetworkNode`
   - Analyzes current scene state and provides information about existing prims
   - Critical for operations that depend on current scene context
   - Must be called before modifying existing scene elements

4. **USD Search Agent** (`ChatUSD_USDSearch`):
   - Implemented as `USDSearchNetworkNode`
   - Searches for USD assets in configured repositories
   - Uses `search_path: "*simready_content*"` to filter for SimReady assets
   - Can apply URL replacements for asset path mapping

**How They Work Together:**

The supervisor's system message (combining `chat_usd_supervisor_identity.md` and `chat_usd_planning_supervisor_identity.md`) defines the orchestration logic:
- It first analyzes the user request to determine which agents are needed
- For scene creation tasks, it typically routes to the planning agent first
- The plan is then executed by calling USD Code Interactive with specific instructions
- Scene Info provides context when working with existing elements
- USD Search finds required assets when needed

This architecture allows each agent to focus on its specialty while the supervisor ensures they work together coherently.
