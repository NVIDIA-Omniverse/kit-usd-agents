# Changelog

All notable changes to the AIQ Planning Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-09-04

### Changed
- Fixed package URL

## [0.1.1] - 2025-01-15

### Added
- **Tool-Aware Planning**: The planning agent now receives information about available tools from the multi-agent network, enabling it to create plans that specifically reference and utilize the tools available in the system.
  
- **Plan Execution Guidance**: Planning modifier now injects itself into the MultiAgent network to ensure the supervisor follows the generated plan step-by-step. This creates a guided execution flow where each step is presented to the supervisor at the appropriate time.

- **Short Plan Format**: Added `short_plan` configuration option that generates concise plans containing only step titles without detailed implementation instructions. This is useful for simple tasks or when details aren't needed upfront.

- **Dynamic Detail Generation**: Added `add_details` configuration option that enables on-demand detail generation during plan execution. When enabled, implementation details for each step are generated just before the supervisor needs to execute that step, providing contextually relevant instructions based on the current state of execution.

### Changed
- Improved code organization with helper methods for better maintainability
- Enhanced documentation and code comments for clarity
- Better separation of concerns between plan generation and execution phases

### Fixed
- Planning details now include full plan context when generating step-specific instructions
- Fixed typo in comments ("planing" → "planning")

## [0.1.0] - Initial Release

### Added
- Initial implementation of the Planning Agent
- Support for creating detailed, step-by-step plans for complex tasks
- Integration with AgentIQ framework
- System prompts for comprehensive plan generation 