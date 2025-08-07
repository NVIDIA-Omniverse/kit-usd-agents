# Overview

The `omni.ai.langchain.widget.core` extension provides a comprehensive set of UI components and tools for building AI-powered interfaces within the Omniverse environment. It serves as the foundation for creating interactive chat-based interfaces that leverage language models and AI agents.

## Important API List

- **ChatView**: A customizable chat interface for displaying and interacting with AI-generated content.
- **ChatWindow**: A window component that encapsulates the ChatView and provides additional functionality.
- **ChatWidget**: A widget that combines ChatHistoryWidget and ChatView, managing their interactions.
- **ChatHistoryWidget**: A widget for displaying and navigating between different conversations.
- **DefaultDelegate**: A delegate class for processing Markdown output and generating UI blocks.
- **AgentDelegate**: An abstract base class for building UI for agents and networks.
- **BrowserPropertyDelegate**: A base class for item property delegates and registry management.

## General Use Case

The `omni.ai.langchain.widget.core` extension is designed for developers who want to integrate AI-powered chat interfaces into their Omniverse applications. It provides a set of flexible and customizable UI components that can be used to:

1. Create interactive chat interfaces for communicating with AI agents.
2. Display and manage multiple conversations or chat sessions.
3. Render AI-generated content, including text, code snippets, and other formatted outputs.
4. Build custom property panels for displaying and editing AI agent properties.
5. Implement drag-and-drop functionality for AI-generated assets or commands.
6. Develop context menus and toolbars for AI-related actions.

By utilizing the components provided by this extension, developers can rapidly prototype and build sophisticated AI-powered user interfaces within the Omniverse ecosystem, enabling users to interact with AI agents, generate content, and manipulate scenes through natural language interactions.

This extension serves as a crucial building block for creating AI-assisted tools and workflows in Omniverse, bridging the gap between powerful language models and the 3D virtual environment.