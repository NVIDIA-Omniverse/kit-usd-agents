## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Register MultiAgentNetworkNode functions.

This module registers LC Agent's MultiAgentNetworkNode with the AgentIQ plugin system.
It provides a flexible function that integrates MultiAgentNetworkNode with AgentIQ's function registry.
"""

from .utils.config_utils import replace_md_file_references
from .utils.multi_agent_network_function import MultiAgentNetworkFunction
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef, FunctionRef
from aiq.data_models.function import FunctionBaseConfig
from lc_agent import RunnableNode
from lc_agent import RunnableSystemAppend
from lc_agent.multi_agent_network_node import MultiAgentNetworkNode
from pydantic import Field
from typing import Optional, TypeVar
import logging

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
StreamingOutputT = TypeVar("StreamingOutputT")
SingleOutputT = TypeVar("SingleOutputT")


class MultiAgentConfigBase(FunctionBaseConfig):
    """Base configuration for MultiAgent functions."""

    name: Optional[str] = None
    description: Optional[str] = "A function that supports multi-agent workflow with routing capabilities"


class MultiAgentConfig(MultiAgentConfigBase, name="MultiAgent"):
    """Configuration for the MultiAgentRouter function."""

    description: str = "Multi-agent router that directs requests to specialized agents"
    llm_name: Optional[LLMRef] = Field(default=None, description="The LLM model to use with the multi-agent router")
    tool_names: list[FunctionRef] = Field(
        default_factory=list, description="The list of tools or agents to provide to the multi-agent router"
    )
    system_message: str = Field(default="", description="Optional system message to guide the router's behavior")
    multishot: bool = Field(default=True, description="Whether to support multi-step/multishot conversations")
    function_calling: bool = Field(
        default=False, description="Whether to use function calling or classification for routing"
    )
    classification_node: bool = Field(
        default=True, description="Whether to use a classification node for routing when function_calling is False"
    )
    generate_prompt_per_agent: bool = Field(default=True, description="Whether to generate a prompt for each agent")
    first_routing_instruction: str = Field(default="", description="Instructions for the AI's first tool selection")
    subsequent_routing_instruction: str = Field(
        default="", description="Instructions for the AI's subsequent tool selections"
    )


@register_function(config_type=MultiAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def multi_agent_router_function(config: MultiAgentConfig, builder: Builder):
    """Register the MultiAgentRouter function with AgentIQ.

    This function sets up a MultiAgentNetworkNode with the specified configuration
    and registers it with the node factory for use in agent workflows.
    """

    system_message = replace_md_file_references(config.system_message)

    class MultiAgentNetworkSupervisorNode(RunnableNode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.inputs.append(RunnableSystemAppend(system_message=system_message))

        # def _sanitize_messages_for_chat_model(self, messages, chat_model_name, chat_model):
        #     """Sanitizes messages and adds metafunction expert type for USD operations."""
        #     messages = super()._sanitize_messages_for_chat_model(messages, chat_model_name, chat_model)
        #     return sanitize_messages_with_expert_type(messages, "knowledge", rag_max_tokens=0, rag_top_k=0)

    # Create and yield the function instance
    yield MultiAgentNetworkFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=MultiAgentNetworkNode,
        lc_agent_node_gen_type=MultiAgentNetworkSupervisorNode,
        multishot=config.multishot,
        function_calling=config.function_calling,
        classification_node=config.classification_node,
        generate_prompt_per_agent=config.generate_prompt_per_agent,
        first_routing_instruction=config.first_routing_instruction,
        subsequent_routing_instruction=config.subsequent_routing_instruction,
    )
