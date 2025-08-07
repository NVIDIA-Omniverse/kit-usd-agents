# Overview

The `omni.ai.langchain.agent.usd_code` extension is a powerful tool designed to facilitate AI-assisted USD (Universal Scene Description) code generation and manipulation within the Omniverse ecosystem. This extension leverages the capabilities of large language models to interpret user requests, generate appropriate USD code, and execute it to modify the current stage in real-time.

## Important API List

- **USDCodeInteractiveNetworkNode**: A network node that generates and executes compact, efficient USD code to instantly alter the current stage. It utilizes meta-functions for optimized performance, allowing immediate visualization of changes in the scene.
- **USDCodeNetworkNode**: A network node for USD code generation and execution without the interactive features.
- **USDCodeGenNetworkNode**: A network node specifically designed for USD code generation.
- **USDCodeGenNode**: A base node for USD code generation.
- **USDKnowledgeNode**: A node that provides knowledge about USD operations and structures.
- **USDCodeInteractiveNode**: A node that handles interactive USD code generation and execution.
- **SceneInfoGenNode**: A node that generates information about the current scene.

## General Use Case

The `omni.ai.langchain.agent.usd_code` extension is primarily used for:

1. **AI-Assisted USD Code Generation**: Users can describe desired changes or additions to a USD scene in natural language, and the extension will generate the appropriate USD code to achieve those results.

2. **Real-time Scene Modification**: The generated code can be immediately executed to modify the current stage, allowing for rapid prototyping and iteration of USD scenes.

3. **Interactive USD Manipulation**: The USDCodeInteractiveNetworkNode provides an interactive interface for users to make changes to the scene through natural language commands.

4. **Scene Analysis**: The SceneInfoGenNode can be used to gather information about the current scene, which can be useful for context-aware code generation.

5. **USD Knowledge Base**: The USDKnowledgeNode serves as a repository of USD-related information, assisting in code generation and providing explanations about USD concepts.

This extension bridges the gap between natural language understanding and USD code execution, making it easier for both novice and experienced users to manipulate USD scenes efficiently.
