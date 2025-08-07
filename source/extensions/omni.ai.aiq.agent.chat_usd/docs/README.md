# AIQ Chat USD Agent

This extension builds a bridge between LC Agent's Chat USD framework and NVIDIA's AgenticIQ platform, enabling advanced conversational AI capabilities for USD development within the AgenticIQ ecosystem.

## Key Integrations

### LC Agent to AgenticIQ
- **Chat USD Network Integration**: Exposes Chat USD as an AIQ workflow and function, allowing it to be used within the AIQ platform.
- **Specialized USD Functions**: Provides USDCodeInteractive and SceneInfo as standalone AIQ functions, usable within Chat USD or other AIQ workflows.
- **Chat Model Registration**: Automatically registers all LC Agent chat models with AgenticIQ.
- **Flexible Configuration**: Enables adding AIQ functions to the Chat USD framework as specialized tools.

### AgenticIQ to LC Agent
- **RunnableAIQNode**: A specialized node that integrates AIQ workflows and functions into LC Agent networks, supporting both streaming and non-streaming responses.
- **Seamless Interoperability**: Allows AIQ components to operate alongside native LC Agent nodes within the same network.

## Example Use Cases
- Using Chat USD's sub-agents within AIQ-powered applications
- Leveraging AIQ's specialized tools (like RAG retrieval) within Chat USD
- Creating hybrid workflows that combine the best capabilities of both platforms

## Technical Integration
The extension handles all the necessary conversions between LC Agent formats and AIQ data structures, making the integration transparent to end users.

## Limitations and Future Work
- AIQ requires Python 3.12, while Kit runs on Python 3.11 (though compatible)
- OpenTelemetry compatibility issues with Kit's pip implementation (can be disabled)
- Nested LC Agent networks in AIQ workflows are not visible to the parent network (potential UI enhancements could address this)
