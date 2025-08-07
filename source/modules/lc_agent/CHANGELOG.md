## 0.2.9 - 2025-09-04
- Fixed package URL

## 0.2.8 - 2025-08-22
- Added async in code interpreter

## 0.2.7 - 2005-08-14
- Support for nested multiagent functionality: MultiAgentNetworkNode can now be used as a tool by another MultiAgentNetworkNode
- This enables building hierarchical multiagent systems where specialized multiagent subnetworks can be composed together

## 0.2.6 - 2025-07-31
- Added comprehensive profiling system for performance analysis
  - Hierarchical profiling with ProfilingFrame and ProfilingData structures
  - Per-network profiling stacks using contextvars for thread-local storage
  - Profiler context manager with auto-stop functionality
  - Interactive HTML visualization with zoom/pan capabilities
  - Support for profiling network streaming, node processing, modifiers, and retrievers
  - Environment variable LC_AGENT_PROFILING for global enable/disable
- Added lazy initialization support to ChatModelRegistry
- Registry can now accept callable factory functions in addition to model instances
- Chat models are instantiated only when first accessed, improving startup performance

## 0.2.5 - 2025-07-10
- Exposed `get_routing_tools_info`: the function to get all the registered tools from multi-agent node

## 0.2.4 - 2025-07-10
- Added NVIDIA license headers to all Python files

## 0.2.3 - 2025-06-17
- Added multi-agent loop detection to prevent infinite loops
- Added loop_detection_message parameter to MultiAgentNetworkNode
- Enhanced multi_agent_utils with loop detection logic

## 0.2.2 - 2025-05-05
- MultiAgent: Ability to use functions with complex args

## 0.2.1 - 2025-03-18
- Removed debug print statement from network_node.py

## 0.2.0 - 2025-03-17
- Switched to use Pydantic v2 and langchain 0.3.x

## 0.1.70 - 2025-03-17
- Added Pydantic compatibility layer to support both v1 and v2
- Improved serialization and deserialization of RunnableNode and RunnableNetwork
- Fixed model serialization with model_serializer for Pydantic v2
- Enhanced NetworkNode with proper deserialization support
- Removed PIL dependency from tests
- Fixed CodeInterpreterTool for single line code execution
- Updated requirements to be more flexible with dependency versions

## 0.1.69 - 2025-03-10
- Added support for Claude 3.7 Sonnet's list-based content format

## 0.1.68 - 2025-02-27
- Ability to add route nodes to MultiAgentNetworkNode runtime

## 0.1.67 - 2025-02-07
- Mutiagent classification is a separate node, it fixes streaming of thinking

## 0.1.66 - 2025-02-04
- MultiAgentNetworkNode fixes for USD Tutor

## 0.1.65 - 2024-12-04
- Instruction customization for MultiAgentNetworkNode

## 0.1.64 - 2024-12-04
- More strict prompt for routing

## 0.1.63 - 2024-12-04
- RunnableNetwork speed up

## 0.1.62 - 2024-11-29
- Don't call retriever if top-k is 0

## 0.1.61 - 2024-11-28
- Added find_metadata method to RunnableNode
- Added get_active_networks to RunnableNetwork
- Enhanced NodeFactory with deep kwargs merging
- Improved MultiAgentNetworkNode metadata handling

## 0.1.60 - 2024-11-26
- Added the supervisor system message when MultiAgentNetworkNode is "no function calling mode"

## 0.1.59 - 2024-11-26
- Ability to create "system message" nodes: `RunnableNode(outputs=SystemMessage("You are an intelligent helper"))`

## 0.1.58 - 2024-11-25
- Ability to not use function calling for multi-agent communication

## 0.1.57 - 2024-11-22
- Replace MD5 with Python's built-in hash() function for improved security and performance

## 0.1.56 - 2024-11-21
- Fix for error when chat model without tools have input messages containing tool_calls

## 0.1.55 - 2024-11-15
- Add kit extension name of the module if it is root

## 0.1.54 - 2024-11-15
- MultiAgentNetworkNode has optional prompt

## 0.1.53 - 2024-11-14
- Fixed issue in CodeInterpreterTool that variables are freed before execution context is closed

## 0.1.52 - 2024-11-13
- MultiAgentNetworkNode supports the prompt that is passed to the sub-network

## 0.1.51 - 2024-11-05
- Convert ToolMessage to HumanMessage when the chat model doesn't support tools

## 0.1.50 - 2024-10-30
- Ensure code atlas dictionaries and lists are in alphabetic order
- Only include file path starting from the root module directory
- Fix for equivalent modules not being done correctly for multiple ones
- Fix for public submodules not being added
- Fix module class names can contain duplicates
- Fix for directories with multiple root modules not being scanned properly
- Added option to not expand equivalent modules when loading code atlas
- Added option to not overwrite existing modules when scanning code atlas

## 0.1.49 - 2024-11-01
- add function_names to CodeAtlasModuleInfo and process imports in root modules
- add function lookup in CodeAtlasLookup lookup_module

## 0.1.48 - 2024-10-31
- Fixed crash in RunnableAppend when invoke is called with custom kwargs

## 0.1.47 - 2024-10-24
- Fixed a problem with sorting messages when using Function Calling

## 0.1.46 - 2024-10-22
- Reduced requirements

## 0.1.45 - 2024-10-17
- fix missing type information for callbacks

## 0.1.44 - 2024-10-18
- Fixed Redis Network List

## 0.1.43 - 2024-10-18
- fix is_root flag to only be in top-level modules

## 0.1.42 - 2024-10-16
- Added MultiAgentNetworkNode

## 0.1.41 - 2024-10-16
- add async version of code execution to wait for extra time for completion
- fix incorrect traceback line number when error happens at the last line
- fix tests don't run on Windows due to outdated pytest

## 0.1.40 - 2024-10-15
- add is_root flag in CodeAtlasModule

## 0.1.39 - 2024-10-16
- add CodeAtlas processing of __all__ variables

## 0.1.38 - 2024-10-11
- set images field for Image nodes in the constructor

## 0.1.37 - 2024-10-01
- add tests for image nodes
- add system image node
- support for passing embedded image data to image nodes

## 0.1.36 - 2024-09-27
- fix for async, property and overload methods missing
- fix for vararg and kwargs missing
- fix for argument default values missing
- fix for method source code not set correctly for single line methods

## 0.1.35 - 2024-09-26
- add human and AI image node.

## 0.1.34 - 2024-09-24
- add equivelent modules for code atalas, so that we can deal with case like omni.ui.scene and omni.ui_scene.scene

## 0.1.33 - 2024-09-18
- minor bug in converting networks to chain
- fixed version of langchain-core

## 0.1.32 - 2024-07-25
- added 6 to tokenizer when doing culling

## 0.1.31 - 2024-07-23
- ability to secure code interpreter with a list of refused modules and methods

## 0.1.30 - 2024-07-23
- Hotfix

## 0.1.29 - 2024-07-22
- Ability to set the max tokens for the messages
- Cull the messages based on max tokens

## 0.1.28 - 2024-07-18
- Performance fix (speedup up to 2ms per node per token when streaming)

## 0.1.27 - 2024-07-15
- If the node is removed in on_pre_invoke, it will not be processed
- The exception is saved to metadata of the node

## 0.1.26 - 2024-07-05
- Support for priority for modifiers
- Save the modifier time to the node metadata
- Save token per second without time to first token
- Added async `_aprocess_parents` to make sure UI is never blocked

## 0.1.25 - 2024-07-02
- Performance optimization

## 0.1.24 - 2024-06-28
- Fixed CodeInterpreter error when executing one line code "from pxr import Usd"
