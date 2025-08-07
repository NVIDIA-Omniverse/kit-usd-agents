## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from aiq.data_models.config import AIQConfig
from aiq.builder.workflow_builder import WorkflowBuilder
import copy
from lc_agent import RunnableNetwork

from typing import Any, AsyncIterator, List, Optional

from .conversion import convert_langchain_to_aiq_messages


class AIQWrapper(BaseChatModel):
    def __init__(self, aiq_config, model_name=None, parent_node=None):
        super().__init__()
        self._aiq_config = aiq_config
        self._chat_model_name = model_name
        self._parent_node = parent_node

    def _prepare_config(self, chat_model_name=None):
        """Prepare AIQ config, ensuring it has required LLM configuration.

        This method handles several important tasks:
        1. Creates a copy of the AIQ configuration to avoid modifying the original
        2. Determines which model to use (from parameters or defaults)
        3. Sets up LLM configuration if not present, adding a LangChain model reference
        4. Updates the workflow to use the configured LLM

        Args:
            chat_model_name: Optional name of the chat model to use, overrides the instance default

        Returns:
            AIQConfig: Validated configuration object ready for AIQ
        """
        # Create a deep copy of the configuration to avoid modifying the original
        aiq_config = copy.deepcopy(self._aiq_config)

        # Determine which model to use - priority: parameter, instance variable, default
        model_name = chat_model_name or self._chat_model_name or "default_model"

        # Get the workflow configuration, which controls the execution
        workflow = aiq_config.get("workflow", {})

        # Check if the workflow already has a specified LLM
        if "llm_name" not in workflow:
            # Define a standard name for the LangChain model reference
            lc_model_name = "lc_agent_chat_model"

            # Create the LLMs section if it doesn't exist
            if "llms" not in aiq_config:
                aiq_config["llms"] = {}

            # Configure the LangChain model with the specified or default model name
            if lc_model_name not in aiq_config["llms"]:
                aiq_config["llms"][lc_model_name] = {
                    "_type": "lc_agent_chat_model",
                    "model_name": model_name,
                }
            else:
                # Update existing model name if the model already exists
                aiq_config["llms"][lc_model_name]["model_name"] = model_name

            # Update the workflow to use our configured LLM
            workflow["llm_name"] = lc_model_name
            aiq_config["workflow"] = workflow

        # Validate the configuration using AIQ's validation system
        return AIQConfig.model_validate(aiq_config)

    def _get_child_network_of_parent_node(self):
        """Find the child network created by the parent node.

        This method searches through active RunnableNetworks to find the network
        that was created as a child of the parent node. It works by iterating
        through networks in leaf-to-root order (from the current/innermost network
        up through its ancestors). When we find the network containing the parent
        node, we return the network that was visited just before it in the iteration,
        which is the child network of that parent.

        This is used to identify networks created by lc_agent_function.py when
        AIQWrapper is invoked within a parent RunnableNetwork context.

        Returns:
            RunnableNetwork or None: The child network if found, None otherwise.
        """
        if not self._parent_node:
            return None
            
        previous_network = None
        # get_active_networks() returns networks in leaf-to-root order
        for network in RunnableNetwork.get_active_networks():
            if self._parent_node in network.nodes:
                # The previous network in the iteration is the child of this parent
                return previous_network
            previous_network = network
        return None

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        config = self._prepare_config(kwargs.get("chat_model_name"))

        # Convert LangChain messages to AIQ format
        chat_request = convert_langchain_to_aiq_messages(messages)

        first_chunk = True

        # Create workflow and run it
        async with WorkflowBuilder.from_config(config) as builder:
            workflow = builder.build()
            async with workflow.run(chat_request) as runner:
                async for chunk in runner.result_stream():
                    if first_chunk:
                        # Capture the child network on first chunk to ensure it exists
                        # This allows parent nodes to access networks created by lc_agent_function
                        if self._parent_node:
                            self._parent_node.subnetwork = self._get_child_network_of_parent_node()
                        first_chunk = False

                    if isinstance(chunk, BaseMessageChunk):
                        yield ChatGenerationChunk(message=chunk)
                    elif isinstance(chunk, str):
                        yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
                    else:
                        # Convert other types to string
                        converted_chunk = runner.convert(value=chunk, to_type=str)
                        yield ChatGenerationChunk(message=AIMessageChunk(content=converted_chunk))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        config = self._prepare_config(kwargs.get("chat_model_name"))

        # Convert LangChain messages to AIQ format
        chat_request = convert_langchain_to_aiq_messages(messages)

        # Create workflow and run it
        async with WorkflowBuilder.from_config(config) as builder:
            workflow = builder.build()
            async with workflow.run(chat_request) as runner:
                result = await runner.result()

                # Capture the child network after result is available
                # This allows parent nodes to access networks created by lc_agent_function
                if self._parent_node:
                    self._parent_node.subnetwork = self._get_child_network_of_parent_node()

                if isinstance(result, BaseMessage):
                    generation = ChatGeneration(message=result)
                elif isinstance(result, str):
                    generation = ChatGeneration(message=AIMessage(content=result))
                else:
                    # Convert other types to string
                    converted_result = runner.convert(value=result, to_type=str)
                    generation = ChatGeneration(message=AIMessage(content=converted_result))

                return ChatResult(generations=[generation])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("This method is not implemented")

    @property
    def _llm_type(self) -> str:
        return "aiq-chat"
