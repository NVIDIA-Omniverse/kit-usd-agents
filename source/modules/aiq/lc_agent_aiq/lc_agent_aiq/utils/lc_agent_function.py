## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
LC Agent Function implementation.

This module provides a Function implementation that integrates with LC Agent's
RunnableNetwork system for processing messages and generating responses.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Generic, TypeVar, Optional, Any
import datetime
import uuid


from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.messages.utils import _get_message_openai_role
from lc_agent import RunnableHumanNode, RunnableAINode, RunnableNetwork
from lc_agent.runnable_node import AINodeMessageChunk

# Import AgentIQ components
from aiq.builder.builder import Builder
from aiq.builder.function import Function
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.api_server import AIQChatRequest
from aiq.data_models.api_server import AIQChatResponseChunk
from aiq.data_models.api_server import AIQChoice
from aiq.data_models.api_server import AIQChoiceMessage
from aiq.data_models.function import FunctionBaseConfig
from lc_agent import get_node_factory, get_chat_model_registry

# Try to import AIQChoiceDelta, but make it optional
try:
    from aiq.data_models.api_server import AIQChoiceDelta

    HAS_AIQ_CHOICE_DELTA = True
except ImportError:
    HAS_AIQ_CHOICE_DELTA = False

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
StreamingOutputT = TypeVar("StreamingOutputT")
SingleOutputT = TypeVar("SingleOutputT")


class LCAgentFunction(Function[AIQChatRequest, str, str]):
    """Network function implementation for MultiAgent that handles message processing and response generation."""

    def __init__(
        self,
        config: FunctionBaseConfig,
        builder: Builder,
        lc_agent_node_type: type,
        lc_agent_node_gen_type: type,
        **kwargs,
    ):
        # Extract description early to ensure it's available for nested multiagent nodes
        # This is critical for nested multiagent functionality where the description
        # needs to be propagated through the kwargs to child nodes
        description = (hasattr(config, "description") and config.description) or lc_agent_node_type.__doc__

        super().__init__(
            config=config,
            description=description,
            input_schema=AIQChatRequest,
            streaming_output_schema=str,
            single_output_schema=str,
            converters=[LCAgentFunction.convert_base_message, LCAgentFunction.convert_chunk],
        )
        self.config = config
        self.builder = builder
        self.lc_agent_node_name = (hasattr(self.config, "name") and self.config.name) or self.config.type
        self.lc_agent_node_type = lc_agent_node_type
        self.lc_agent_node_gen_name = self.lc_agent_node_name + "Gen"
        self.lc_agent_node_gen_type = lc_agent_node_gen_type
        self.lc_agent_node_kwargs = kwargs
        
        # Ensure description is passed through kwargs to enable proper tool registration
        # for nested multiagent nodes. This allows multiagent nodes to be used as tools
        # by parent multiagent nodes with correct descriptions.
        if description:
            self.lc_agent_node_kwargs["description"] = description

        self.chat_model_name = None
        if hasattr(self.config, "llm_name") and self.config.llm_name:
            self.chat_model_name = self.config.llm_name + " " + self.lc_agent_node_name

        if self.lc_agent_node_gen_type and "default_node" not in self.lc_agent_node_kwargs:
            self.lc_agent_node_kwargs["default_node"] = self.lc_agent_node_gen_name

        if "chat_model_name" not in self.lc_agent_node_kwargs:
            self.lc_agent_node_kwargs["chat_model_name"] = self.chat_model_name

    @staticmethod
    def convert_base_message(value: BaseMessage) -> str:
        """Convert a BaseMessage to a string containing its content."""
        return str(value.content)

    @staticmethod
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

    async def pre_invoke(self, value: AIQChatRequest) -> None:
        """Called before invoking or streaming from the network.

        This method can be used to perform setup operations before processing
        the request, such as logging, validation, or initialization.

        Args:
            value: The AIQChatRequest containing the messages to process.
        """
        logger.debug(f"Pre-invoke registering {self.lc_agent_node_name} with {self.lc_agent_node_type}")
        get_node_factory().register(self.lc_agent_node_type, name=self.lc_agent_node_name, **self.lc_agent_node_kwargs)
        if self.lc_agent_node_gen_type:
            get_node_factory().register(self.lc_agent_node_gen_type, name=self.lc_agent_node_gen_name)

        # Get the chat model name and instance from the config
        if self.chat_model_name:
            chat_model = await self.builder.get_llm(self.config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

            # Register the chat model with LC Agent's model registry
            model_registry = get_chat_model_registry()
            model_registry.register(self.chat_model_name, chat_model)

    async def post_invoke(self, value: AIQChatRequest, success: bool = True, error: Optional[Exception] = None) -> None:
        """Called after invoking or streaming from the network, regardless of success.

        This method is useful for cleanup operations, logging, monitoring, etc.
        It's called even if the invoke operation fails.

        Args:
            value: The AIQChatRequest that was processed
            success: Whether the invoke operation succeeded
            error: The exception that occurred, if any
        """
        if success:
            logger.debug(f"Post-invoke called successfully for {self.lc_agent_node_name}")
        else:
            logger.warning(f"Post-invoke called after error for {self.lc_agent_node_name}: {error}")

        get_node_factory().unregister(self.lc_agent_node_name)
        if self.lc_agent_node_gen_type:
            get_node_factory().unregister(self.lc_agent_node_gen_name)

        if self.chat_model_name:
            model_registry = get_chat_model_registry()
            model_registry.unregister(self.chat_model_name)

    def _find_reusable_parent_node(self) -> Optional[Any]:
        """Find a suitable parent node from active networks that can be reused.

        This method searches through active RunnableNetworks to find a parent node
        that can be extended with our new node. It looks for networks with a single
        parent node connected to a leaf node, which indicates a linear chain that
        we can safely extend.

        Returns:
            The parent node if a suitable one is found, None otherwise.
        """
        # Get the first active network (if any)
        active_network = None
        for network in RunnableNetwork.get_active_networks():
            active_network = network
            break

        if not active_network:
            return None

        # Get the leaf node of the active network
        leaf_node = active_network.get_leaf_node()
        if not leaf_node:
            return None

        # Check if the leaf node has exactly one parent
        # This ensures we're dealing with a linear chain, not a complex graph
        parent_nodes = active_network.get_parents(leaf_node)
        if len(parent_nodes) != 1:
            return None

        return parent_nodes[0]

    def _create_network_from_parent_node(self, parent_node: Any) -> RunnableNetwork:
        """Create a new RunnableNetwork that extends from an existing parent node.

        This allows us to reuse the existing network hierarchy instead of creating
        a completely new one, which can be more efficient and maintain context
        from previous operations.

        NOTE: This method intentionally creates a cross-network connection by linking
        a node from an existing network (parent_node) to a new node in a newly created
        RunnableNetwork. This is a supported feature in LC Agent that allows network
        composition and hierarchy reuse.

        Args:
            parent_node: The parent node to connect our new node to.

        Returns:
            A new RunnableNetwork with our node connected to the parent.
        """
        with RunnableNetwork(default_node=self.lc_agent_node_name, chat_model_name=self.chat_model_name) as network:
            # Intentionally connect parent node from existing network to new node in this network
            # This cross-network connection is supported by LC Agent and allows network composition
            parent_node >> get_node_factory().create_node(self.lc_agent_node_name)
            logger.info(
                f"Created network extension: Reusing existing hierarchy by connecting "
                f"{self.lc_agent_node_name} to parent node"
            )

        return network

    async def setup_network(self, config: FunctionBaseConfig, input_message: AIQChatRequest) -> RunnableNetwork:
        """Set up a RunnableNetwork with messages from AIQChatRequest."""
        system_messages = []
        if hasattr(config, "system_message") and config.system_message:
            system_messages.append(SystemMessage(content=config.system_message))

        # Try to reuse an existing network hierarchy if no system messages are present
        # System messages typically indicate a new conversation context, so we only
        # reuse existing networks when continuing an existing conversation
        if not system_messages:
            reusable_parent_node = self._find_reusable_parent_node()
            if reusable_parent_node:
                return self._create_network_from_parent_node(reusable_parent_node)

        # Create a RunnableNetwork with the specified configuration
        with RunnableNetwork(default_node=self.lc_agent_node_name, chat_model_name=self.chat_model_name) as network:
            # Convert AIQChatRequest messages to LangChain messages
            for msg in input_message.messages:
                if msg.role == "user":
                    RunnableHumanNode(msg.content)
                elif msg.role == "assistant":
                    RunnableAINode(msg.content)
                elif msg.role == "system":
                    system_messages.append(SystemMessage(content=msg.content))

            # if system_messages:
            #     # Add a node for the system message and LLM processing
            #     RunnableNode(inputs=[RunnableLambda(lambda x, system_messages=system_messages: system_messages)])

        return network

    async def _ainvoke(self, value: AIQChatRequest) -> str:
        """Process input messages and generate a complete response."""
        await self.pre_invoke(value)
        success = False
        error = None
        try:
            network = await self.setup_network(self.config, value)

            # Invoke the network asynchronously to get the result
            result = await network.ainvoke()

            response = result.content if hasattr(result, "content") else str(result)
            if isinstance(response, str) and response.startswith("FINAL "):
                response = response[6:]

            success = True
            return response
        except Exception as e:
            error = e
            raise
        finally:
            await self.post_invoke(value, success=success, error=error)

    async def _astream(self, value: AIQChatRequest) -> AsyncGenerator[str, None]:
        """Process input messages and stream the response chunks."""
        await self.pre_invoke(value)
        success = False
        error = None
        try:
            network = await self.setup_network(self.config, value)
            # Network instance created by lc_agent_function will be accessible via
            # parent_node.subnetwork after the first chunk is yielded in AIQWrapper

            last_node = None

            # Stream the response
            async for chunk in network.astream():
                if isinstance(chunk, AINodeMessageChunk):
                    current_node = chunk.node
                    if current_node != last_node:
                        if last_node:
                            yield AINodeMessageChunk(content="\n\n", node=current_node)

                        last_node = chunk.node

                    if (
                        isinstance(current_node.outputs, AIMessage)
                        and current_node.outputs.content == chunk.content
                        and chunk.content.startswith("FINAL")
                    ):
                        chunk.content = chunk.content[6:]

                if not chunk.content:
                    continue

                yield chunk

            success = True
        except Exception as e:
            error = e
            raise
        finally:
            # Call post_invoke in finally to ensure it's always called
            await self.post_invoke(value, success=success, error=error)
