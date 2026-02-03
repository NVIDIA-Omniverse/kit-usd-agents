# Changelog

All notable changes to the lc_agent_nat module will be documented in this file.

## [0.2.2] - 2026-01-06
### Fixed
- Fixed streaming response chunk type mismatch with NAT API
  - `AIQChatResponseChunk.choices` expects `ChatResponseChunkChoice` (with `delta` field), not `ChatResponseChoice` (with `message` field)
  - Now correctly uses `ChatResponseChunkChoice` with `ChoiceDelta` for streaming responses
  - Maintains backward compatibility with older NAT versions that lack streaming chunk types

### Added
- Added `output_mode` option to `MultiAgentConfig` for controlling response processing
  - `"default"`: strips "FINAL " prefix from responses (current behavior)
  - `"raw"`: returns responses unprocessed
- Added unit tests for config_utils, multi_agent_config, and message conversion

## [0.2.0] - 2025-11-03
### Changed
- **BREAKING**: Migrated from aiqtoolkit to nvidia-nat 1.3.0
- **BREAKING**: Renamed package from lc_agent_aiq to lc_agent_nat
- **BREAKING**: Updated all imports from aiq.* to nat.*
- **BREAKING**: Updated entry points from aiq.components/aiq.front_ends to nat.components/nat.front_ends
- Updated package description to reference NAT (NVIDIA AgentIQ Toolkit)

### Fixed
- Fixed `MultiAgentNetworkFunction` to support NAT 1.3.0 function groups
  - Function groups are now properly detected and expanded to individual functions
  - Both `get_function_group()` and `get_function()` are tried to handle all tool types
- Fixed missing `await` for async `builder.get_function()` and `builder.get_function_group()` calls
- Fixed tool validation errors to be returned to LLM instead of failing workflow
  - Added `ValidationErrorHandlingFunction` wrapper class to catch Pydantic validation errors
  - All NAT functions from function groups are wrapped to handle validation gracefully
  - Added ValidationError catching in `LCAgentFunction._ainvoke()` and `_astream()`
  - Validation errors are formatted as helpful error messages showing missing/invalid fields and required schema
  - LLM can now see validation errors and retry with correct parameters instead of the workflow failing
- Fixed single-field parameter extraction for MCP and NAT tools
  - When LLM outputs `{"param": "value"}` for single-parameter functions, the value is now correctly extracted
  - JSON parsing in `_convert_value_to_schema()` extracts nested parameter values
  - Dict value extraction in `ValidationErrorHandlingFunction.ainvoke/astream()` handles kwargs-style calls
  - Prevents passing wrapped dicts to MCP servers that expect unwrapped values

## [0.1.12] - 2025-12-16
### Fixed
- Fixed security issue: Changed HTTP placeholder URLs to HTTPS in conversion.py
  - Default URL changed from `http://default.com` to `https://placeholder.invalid`
  - Placeholder URL for base64 images changed from `http://image_data` to `https://placeholder.invalid`
  - Fixes SonarQube security warnings about insecure HTTP protocol usage

## [0.1.11] - 2025-10-10
- fix aiq nodes include previous network's nodes

## [0.1.10] - 2025-09-22
- support data:image url in message

## [0.1.9] - 2025-09-04
- Fixed package URL

## [0.1.8] - 2005-08-14
- Support for nested multiagent functionality: MultiAgent nodes can now be used as tools by other MultiAgent nodes
- Proper description propagation through kwargs for nested multiagent tool registration

## [0.1.7] - 2025-07-15
- Improved streaming: delta is optional, it makes lc_agent_nat compatible with older NAT

## [0.1.6] - 2025-07-10
- Added NVIDIA license headers to all Python files

## [0.1.5] - 2025-07-09
- Streaming response contains the field "delta"

## [0.1.4] - 2025-07-03
- Connecting the sub-networks created in AIQ to the parent network

## [0.1.3] - 2025-07-03
### Added
- Added subnetwork tracking capability to RunnableAIQNode to enable better network hierarchy management
- Added parent node reference passing to AIQWrapper for accessing parent-child network relationships

### Changed
- Enhanced AIQWrapper to automatically detect and set subnetwork references during execution
- Modified streaming and result generation to capture child networks created by lc_agent_function

## [0.1.2] - 2025-06-17
### Changed
- Updated to stable aiqtoolkit 1.1.0 (from 1.1.0rc3)
- Made lc_agent_chat_models import optional to reduce dependencies
- Enhanced AIQ streaming with node separation markers

### Removed
- Removed MCP AIQ plugin (it exists in AIQ)

## [0.1.1] - 2025-05-27
- Fixed functions with no args
- When streaming, yield the final result

## [0.1.0] - 2025-05-05

### Added
- Initial release of LC Agent plugin for NAT
- Integration utilities between LC Agent and NAT
- Node implementations:
  - FunctionRunnableNode
  - RunnableAIQNode
- Multi-agent configuration and network functionality
- Utility functions:
  - AIQWrapper for NAT integration
  - Message conversion between Langchain and NAT
  - LCAgentFunction for function registration
- Configuration examples:
  - Chat workflow
  - Multi-agent workflow