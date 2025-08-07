# Message Flow

This document explains how messages flow through the Chat USD system, from user input to final response. Understanding this flow is essential for extending or customizing the system.

## Overview

Messages in Chat USD flow through a series of components, each processing the message and passing it to the next component in the chain. The flow is orchestrated by the `ChatUSDNetworkNode` and `ChatUSDSupervisorNode`, which route messages to specialized agents based on the message content.

## Message Types

Chat USD uses several types of messages, defined in the LC Agent framework:

1. **HumanMessage**: Represents user input
2. **AIMessage**: Represents AI-generated responses
3. **SystemMessage**: Provides instructions to the AI
4. **ToolMessage**: Represents the output of tool calls

These message types are used throughout the system to represent different types of information.

## Basic Message Flow

The basic message flow in Chat USD follows these steps:

1. **User Input**: The user provides a query through the chat interface, which is converted to a `HumanMessage`
2. **ChatUSDNetworkNode**: The message is received by the `ChatUSDNetworkNode`, which is the main entry point for the system
3. **ChatUSDSupervisorNode**: The message is passed to the `ChatUSDSupervisorNode`, which analyzes it to determine its intent
4. **Specialized Agent**: The message is routed to the appropriate specialized agent based on its intent
5. **Agent Processing**: The agent processes the message and generates a response as an `AIMessage`
6. **Response Integration**: The response is passed back to the `ChatUSDSupervisorNode`, which integrates it into a coherent reply
7. **Final Response**: The integrated response is returned to the user

This flow ensures that each message is handled by the most appropriate agent and that the response is coherent and comprehensive.

## Error Handling in Message Flow

When errors occur in the message flow, they are handled through a combination of error detection, reporting, and handling:

1. **Error Detection**: Errors are detected at the component where they occur
2. **Error Reporting**: Errors are reported through the component's metadata
3. **Error Handling**: Errors are handled by modifiers or other components

For example, if a code execution error occurs in the `USDCodeInteractiveNetworkNode`, it is detected, reported, and handled by the `DoubleRunUSDCodeGenInterpreterModifier`.
