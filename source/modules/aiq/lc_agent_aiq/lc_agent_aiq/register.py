## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Register AgentIQ functions.

This module registers LC Agent functions with the AgentIQ plugin system.
It provides a simple function that integrates LC Agent's RunnableNetwork
with AgentIQ's function registry.
"""

from collections.abc import AsyncGenerator
import datetime
import logging
import uuid

from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.messages.utils import _get_message_openai_role
from langchain_core.runnables.base import RunnableLambda
from pydantic import Field

# Import core components from LC Agent
from lc_agent import RunnableAINode
from lc_agent import RunnableHumanNode
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode
from lc_agent import get_chat_model_registry
from lc_agent import get_node_factory
from lc_agent.runnable_node import AINodeMessageChunk

try:
    from lc_agent_chat_models import register_all as register_chat_models
except ImportError:
    register_chat_models = None

# Import AgentIQ components for plugin registration
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import AIQChoice
from aiq.data_models.api_server import AIQChoiceMessage
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

# Try to import AIQChoiceDelta, but make it optional
try:
    from aiq.data_models.api_server import AIQChoiceDelta

    HAS_AIQ_CHOICE_DELTA = True
except ImportError:
    HAS_AIQ_CHOICE_DELTA = False

# LLM Provider
from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt

from aiq.builder.builder import Builder
from aiq.builder.llm import LLMProviderInfo
from aiq.cli.register_workflow import register_llm_provider
from aiq.data_models.llm import LLMBaseConfig

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_llm_client

logger = logging.getLogger(__name__)

if register_chat_models:
    register_chat_models()


class SimpleChatModelConfig(LLMBaseConfig, name="lc_agent_chat_model"):
    """Configuration for the lc_agent chat model.

    Attributes:
        model_name: The name of the chat model to use from the lc_agent registry.
    """

    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The lc_agent chat_model.",
    )


@register_llm_provider(config_type=SimpleChatModelConfig)
async def chat_model_provider(llm_config: SimpleChatModelConfig, builder: Builder):
    """Register a lc_agent chat model provider with AgentIQ."""
    yield LLMProviderInfo(config=llm_config, description="A lc_agent chat_model for use with an LLM client.")


@register_llm_client(config_type=SimpleChatModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def chat_model(llm_config: SimpleChatModelConfig, builder: Builder):
    """Create a lc_agent chat model client for use with AgentIQ.

    This function retrieves a chat model from the lc_agent model registry
    based on the provided configuration and yields it for use with AgentIQ.
    """
    model_registry = get_chat_model_registry()
    chat_model = model_registry.get_model(llm_config.model_name)

    if chat_model is None:
        raise ValueError(f"Chat model {llm_config.model_name} not found")

    yield chat_model


class SimpleFunctionConfig(FunctionBaseConfig, name="lc_agent_simple"):
    """Configuration for the simple function.

    This Pydantic model defines the configuration needed for the simple_function.
    The 'name' parameter specifies the type identifier used in workflow YAML configs.

    Attributes:
        system_message: The system message to send to the LLM.
        llm_name: Reference to the LLM model to use with the agent.
        verbose: Whether to log detailed information.
    """

    system_message: str = ""
    llm_name: LLMRef = Field(description="The LLM model to use with the tool calling agent.")
    verbose: bool = Field(default=False, description="Whether to log detailed information.")


@register_function(config_type=SimpleFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def simple_function(config: SimpleFunctionConfig, builder: Builder):
    """Register a simple function that uses LC Agent's RunnableNetwork.

    This is an AgentIQ plugin function that demonstrates integration with LC Agent.
    It follows the AgentIQ plugin pattern using the @register_function decorator
    and yields a FunctionInfo object.

    Args:
        config: Configuration instance of SimpleFunctionConfig.
        builder: AgentIQ Builder instance used to obtain resources.

    Yields:
        FunctionInfo: Information about the registered function.
    """
    logger.info("Registering simple_function with LLM: %s", config.llm_name)

    # Get the chat model name and instance from the config
    chat_model_name = config.llm_name
    chat_model = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    system_message = config.system_message

    # Register the chat model with LC Agent's model registry
    model_registry = get_chat_model_registry()
    model_registry.register(chat_model_name, chat_model)

    # Register LC Agent node types with the node factory
    get_node_factory().register(RunnableHumanNode)
    get_node_factory().register(RunnableNode)

    async def setup_network(input_message: AIQChatRequest) -> RunnableNetwork:
        """Set up a RunnableNetwork with messages from AIQChatRequest.

        Args:
            input_message: The AIQChatRequest containing the messages.

        Returns:
            RunnableNetwork: The configured network ready for invocation.
        """
        if config.verbose:
            logger.info(
                "Setting up network with messages: %s",
                (
                    str(input_message.messages)[:100] + "..."
                    if len(str(input_message.messages)) > 100
                    else str(input_message.messages)
                ),
            )

        system_messages = []
        if system_message:
            system_messages.append(SystemMessage(content=system_message))

        # Create a RunnableNetwork with the specified chat model
        with RunnableNetwork(default_node="RunnableNode", chat_model_name=chat_model_name) as network:
            # Convert AIQChatRequest messages to LangChain messages
            for msg in input_message.messages:
                if msg.role == "user":
                    RunnableHumanNode(msg.content)
                elif msg.role == "assistant":
                    RunnableAINode(msg.content)
                elif msg.role == "system":
                    system_messages.append(SystemMessage(content=msg.content))

            # Add a node for the system message and LLM processing
            RunnableNode(inputs=[RunnableLambda(lambda x, system_messages=system_messages: system_messages)])

        return network

    async def generate_response(input_message: AIQChatRequest) -> str:
        """Process a list of input messages and generate a complete response.

        Args:
            input_message: The AIQChatRequest containing the messages.

        Returns:
            str: The response content from the LLM.
        """
        with await setup_network(input_message) as network:
            # Invoke the network asynchronously to get the result
            result = await network.ainvoke()

        if config.verbose:
            response = result.content
            logger.info("Generated response (length: %d characters)", len(response))

        return result

    async def stream_response(input_message: AIQChatRequest) -> AsyncGenerator[str, None]:
        """Process messages and stream the response chunks.

        Args:
            input_message: The AIQChatRequest containing the messages.

        Yields:
            str: Chunks of the response from the LLM.
        """
        with await setup_network(input_message) as network:
            # Stream the response
            async for chunk in network.astream():
                if chunk.content:
                    if config.verbose:
                        logger.info("Streaming chunk (length: %d characters)", len(chunk.content))
                    yield chunk

    def convert_base_message(value: BaseMessage) -> str:
        return str(value.content)

    def convert_chunk(value: AINodeMessageChunk) -> AIQChatResponseChunk:
        """Convert a AINodeMessageChunk to AIQChatResponseChunk."""
        role = _get_message_openai_role(value)

        # Build choice arguments based on available imports
        choice_kwargs = {
            "message": AIQChoiceMessage(content=value.content, role=role),
            "index": 0,
        }

        # Only add delta if AIQChoiceDelta is available
        if HAS_AIQ_CHOICE_DELTA:
            choice_kwargs["delta"] = AIQChoiceDelta(content=value.content, role=role)

        choice = AIQChoice(**choice_kwargs)
        chunk_id = value.id if value.id is not None else str(uuid.uuid4())
        created_time = datetime.datetime.now(datetime.timezone.utc)
        return AIQChatResponseChunk(id=chunk_id, choices=[choice], created=created_time)

    # Yield the function info to register with AgentIQ
    yield FunctionInfo.create(
        single_fn=generate_response,
        stream_fn=stream_response,
        description="A function from LC Agent that supports both streaming and non-streaming responses",
        converters=[convert_base_message, convert_chunk],
    )


class AddFunctionConfig(FunctionBaseConfig, name="my_add_function"):
    """Configuration for the my_add function.

    Attributes:
        verbose: Whether to log detailed information.
    """

    verbose: bool = Field(default=False, description="Whether to log detailed information.")


@register_function(config_type=AddFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register_my_add(config: AddFunctionConfig, builder: Builder):
    """Register a simple addition function with AgentIQ.

    Args:
        config: Configuration instance of AddFunctionConfig.
        builder: AgentIQ Builder instance used to obtain resources.

    Yields:
        FunctionInfo: Information about the registered function.
    """
    logger.info("Registering my_add function")

    # Define a class for the input parameters with descriptions
    from pydantic import BaseModel, Field

    async def my_add_direct2(a: int, b: int) -> int:
        """Add two integers together directly.

        Args:
            a: First integer to add
            b: Second integer to add

        Returns:
            The sum of a and b
        """
        if config.verbose:
            logger.warning(f"Adding {a} + {b}")
        return a + b

    async def my_add_direct(value: str) -> int:
        return str(eval(value))

    # Yield the function info to register with AgentIQ
    yield FunctionInfo.create(
        single_fn=my_add_direct,
        description="A simple function that adds two integers together directly",
        # input_schema=AddInput,  # Use our schema with descriptions
        # No input_schema needed, parameters will be extracted from the function signature
        # and descriptions from the docstring
    )
