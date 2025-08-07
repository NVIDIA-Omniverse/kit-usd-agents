# Changelog

All notable changes to the lc_agent_aiq module will be documented in this file.

## [0.1.9] - 2025-09-04
- Fixed package URL

## [0.1.8] - 2005-08-14
- Support for nested multiagent functionality: MultiAgent nodes can now be used as tools by other MultiAgent nodes
- Proper description propagation through kwargs for nested multiagent tool registration

## [0.1.7] - 2025-07-15
- Improved streaming: delta is optional, it makes lc_agent_aiq compatible with older AIQ

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
- Initial release of LC Agent plugin for AgentIQ
- Integration utilities between LC Agent and AgentIQ
- Node implementations:
  - FunctionRunnableNode
  - RunnableAIQNode
- Multi-agent configuration and network functionality
- Utility functions:
  - AIQWrapper for AgentIQ integration
  - Message conversion between Langchain and AgentIQ
  - LCAgentFunction for function registration
- Configuration examples:
  - Chat workflow
  - Multi-agent workflow