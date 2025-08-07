# USDCodeInteractiveNetworkNode

The `USDCodeInteractiveNetworkNode` is a specialized agent in the Chat USD system that handles USD code generation and execution. It extends the `USDCodeInteractiveNetworkNodeBase` class from the LC Agent USD module and is responsible for generating and executing USD code based on user queries.

## Overview

The `USDCodeInteractiveNetworkNode` serves as the code generation and execution engine of the Chat USD system. It can generate USD code based on natural language descriptions, execute the code to modify the USD stage, and provide feedback on the execution results. This enables users to create and modify USD scenes through natural language interaction.

## Implementation

The `USDCodeInteractiveNetworkNode` is implemented as a Python class that extends `USDCodeInteractiveNetworkNodeBase`:

```python
class USDCodeInteractiveNetworkNode(USDCodeInteractiveNetworkNodeBase):
    """
    "USD Code Interactive" node. Use it to modify USD stage in real-time and import assets that was found with another tools.

    Important:
    - Never use `stage = omni.usd.get_context().get_stage()` in the code. The global variable `stage` is already defined.
    """

    def __init__(
        self,
        snippet_verification=False,
        scene_info=True,
        max_retries: Optional[int] = None,
        enable_code_interpreter=True,
        enable_code_atlas=True,
        enable_metafunctions=True,
        enable_interpreter_undo_stack=True,
        enable_code_promoting=False,
        double_run_first=True,
        double_run_second=True,
        **kwargs,
    ):
        super().__init__(enable_code_atlas=enable_code_atlas, enable_metafunctions=enable_metafunctions, **kwargs)

        if max_retries is None:
            max_retries = carb.settings.get_settings().get(MAX_RETRIES_SETTINGS) or MAX_RETRIES_DEFAULT

        if scene_info and enable_code_interpreter:
            self.add_modifier(
                SceneInfoModifier(
                    code_interpreter_hide_items=self.code_interpreter_hide_items,
                    enable_interpreter_undo_stack=enable_interpreter_undo_stack,
                    enable_rag=enable_code_atlas,
                    max_retries=max_retries,
                ),
                priority=-100,
            )

        if enable_code_promoting:
            # It will find the code and error message in the subnetwork and copy it to this network output
            self.add_modifier(SceneInfoPromoteLastNodeModifier())
        self.add_modifier(NetworkLenghtModifier(max_length=max_retries))
        self.add_modifier(CodeExtractorModifier(snippet_verification=snippet_verification))
        if enable_code_interpreter:
            self.add_modifier(
                DoubleRunUSDCodeGenInterpreterModifier(
                    hide_items=self.code_interpreter_hide_items,
                    undo_stack=enable_interpreter_undo_stack,
                    first_run=double_run_first,
                    second_run=double_run_second,
                )
            )

        self.metadata["description"] = "Agent to interact and modify the USD stage."
        self.metadata["examples"] = [
            "Create a red sphere",
            "Rotate the sphere 90 degrees around Y axis",
            "Move the selected object up 100 units",
        ]
```

The class initializes itself with various parameters that control its behavior, adds several modifiers to extend its functionality, and sets metadata for the node.

## Configuration Parameters

The `USDCodeInteractiveNetworkNode` can be configured with various parameters to customize its behavior:

- **snippet_verification**: Whether to verify code snippets before execution
- **scene_info**: Whether to enable scene information retrieval
- **max_retries**: Maximum number of retries for code execution
- **enable_code_interpreter**: Whether to enable code execution
- **enable_code_atlas**: Whether to enable code atlas for RAG
- **enable_metafunctions**: Whether to enable metafunctions
- **enable_interpreter_undo_stack**: Whether to enable the interpreter undo stack
- **enable_code_promoting**: Whether to enable code promoting
- **double_run_first**: Whether to enable the first run of the double run modifier
- **double_run_second**: Whether to enable the second run of the double run modifier

These parameters allow the node to be customized for different use cases and requirements.

## Modifiers

The `USDCodeInteractiveNetworkNode` uses several modifiers to extend its functionality:

1. **SceneInfoModifier**: Provides scene information for code generation
2. **SceneInfoPromoteLastNodeModifier**: Promotes code and error messages from subnetworks
3. **NetworkLenghtModifier**: Controls the maximum length of the network
4. **CodeExtractorModifier**: Extracts code snippets from responses
5. **DoubleRunUSDCodeGenInterpreterModifier**: Executes code snippets and handles errors

These modifiers work together to provide a comprehensive code generation and execution experience.

### SceneInfoModifier

The `SceneInfoModifier` provides scene information for code generation. It retrieves information about the current USD stage, such as the available prims, their properties, and their relationships. This information is used to generate more accurate and effective code.

### SceneInfoPromoteLastNodeModifier

The `SceneInfoPromoteLastNodeModifier` promotes code and error messages from subnetworks to the main network. This ensures that the user sees the final code and any error messages, even if they were generated by a subnetwork.

### NetworkLenghtModifier

The `NetworkLenghtModifier` controls the maximum length of the network. This prevents the network from growing too large and consuming too many resources.

### CodeExtractorModifier

The `CodeExtractorModifier` extracts code snippets from responses. It identifies code blocks in the response and extracts them for execution. It can also verify the code snippets to ensure they are valid.

### DoubleRunUSDCodeGenInterpreterModifier

The `DoubleRunUSDCodeGenInterpreterModifier` executes code snippets and handles errors. It runs the code twice: once to check for errors and once to execute it. If errors are detected in the first run, it attempts to fix them before the second run.

## Code Generation Process

The code generation process in `USDCodeInteractiveNetworkNode` follows these steps:

1. **Query Analysis**: The user's query is analyzed to understand the requirements
2. **Scene Information Retrieval**: If enabled, scene information is retrieved to provide context
3. **Code Generation**: Code is generated based on the query and scene information
4. **Code Extraction**: Code snippets are extracted from the generated response
5. **Code Execution**: If enabled, the code is executed to modify the USD stage
6. **Error Handling**: If errors occur, they are handled and the code is fixed if possible
7. **Response Generation**: A response is generated with the code and execution results

This process ensures that the generated code is accurate, effective, and executable.

## Chat Model Selection

The `USDCodeInteractiveNetworkNode` selects the appropriate chat model for code generation:

```python
def _pre_invoke_network(self):
    """Called before invoking the network."""
    super()._pre_invoke_network()

    if self.chat_model_name == "nvidia/usdcode-llama3-70b-instruct":
        # In the case of Chat USD, we want to use the interactive version of
        # ChatUSD if available
        chat_model_name = "nvidia/usdcode-llama3-70b-instruct-interactive"
        model = get_chat_model_registry().get_model(chat_model_name)
        if model:
            self.chat_model_name = chat_model_name
```

This ensures that the most appropriate model is used for code generation, based on the requirements of the task.
