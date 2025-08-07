## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from urllib.parse import urlparse, urlunparse

import carb.settings

# Import AgentIQ components for plugin registration
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef, LLMRef
from aiq.data_models.function import FunctionBaseConfig
from lc_agent import RunnableSystemAppend

try:
    from lc_agent_aiq import LCAgentFunction, MultiAgentNetworkFunction
except ImportError:
    LCAgentFunction = None
    MultiAgentNetworkFunction = None

# LLM Provider
from pydantic import Field, validator

logger = logging.getLogger(__name__)


InputT = TypeVar("InputT")
StreamingOutputT = TypeVar("StreamingOutputT")
SingleOutputT = TypeVar("SingleOutputT")


class ChatUSDConfigBase(FunctionBaseConfig):
    name: Optional[str] = None
    description: Optional[str] = "A function that supports both streaming and non-streaming responses for Chat USD"


class ChatUSDConfig(ChatUSDConfigBase, name="ChatUSD"):
    name: str = "ChatUSD AIQ Config"
    description: str = "Multi-agents to modify the scene and search USD assets."
    llm_name: Optional[LLMRef] = Field(default=None, description="The LLM model to use with the tool calling agent.")
    tool_names: list[FunctionRef] = Field(
        default_factory=list, description="The list of tools to provide to the react agent."
    )


class ChatUSDCodeConfig(ChatUSDConfigBase, name="ChatUSD_USDCodeInteractive"):
    name: str = "ChatUSD_USDCodeInteractive"


class ChatUSDSceneInfoConfig(ChatUSDConfigBase, name="ChatUSD_SceneInfo"):
    name: str = "ChatUSD_SceneInfo"


class ChatUSDSearchConfig(ChatUSDConfigBase, name="ChatUSD_USDSearch"):
    name: str = "ChatUSD_USDSearch"
    host_url: Optional[str] = Field(
        default=None, description="Optional host URL for USD Search API. If provided, overrides the default settings."
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for USD Search API. If provided, overrides the default settings/environment.",
    )
    username: Optional[str] = Field(
        default=None,
        description="Optional username for USD Search API. If provided, uses basic auth with username/api_key instead of bearer token.",
    )
    url_replacements: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional URL replacements to apply to search results. Dictionary mapping old URLs to new URLs.",
    )
    search_path: Optional[str] = Field(
        default=None,
        description="Optional search path to filter USD Search results. Used to search only specific content like SimReady assets with scale alignment.",
    )

    @validator("url_replacements", pre=True)
    def validate_url_replacements(cls, v):
        """Validate that url_replacements is a dictionary and coerce if possible."""
        if v is None:
            return v

        if isinstance(v, dict):
            # Ensure all keys and values are strings
            try:
                return {str(k): str(v) for k, v in v.items()}
            except Exception as e:
                raise ValueError(f"Unable to convert url_replacements to dictionary of strings: {e}")

        # If it's not a dictionary, try to coerce common formats
        if isinstance(v, (list, tuple)):
            # Try to convert list of tuples to dict
            try:
                return dict(v)
            except Exception:
                raise ValueError(f"url_replacements must be a dictionary, got {type(v).__name__}")

        raise ValueError(f"url_replacements must be a dictionary, got {type(v).__name__}")

    @validator("host_url", pre=True)
    def normalize_host_url(cls, v):
        """Normalize host_url by ensuring it has a scheme and removing trailing slashes."""
        if v is None or v == "":
            return v

        # Convert to string and strip whitespace
        url = str(v).strip()

        # Parse the URL
        parsed = urlparse(url)

        # If no scheme is provided, assume https
        if not parsed.scheme:
            url = f"https://{url}"
            parsed = urlparse(url)

        # Remove trailing slashes from the path
        path = parsed.path.rstrip("/")

        # Reconstruct the URL without trailing slashes
        normalized = urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment))

        return normalized


class ChatUSDPlanningConfig(ChatUSDConfigBase, name="ChatUSD_Planning"):
    name: str = "ChatUSD_Planning"
    system_message: Optional[str] = Field(
        default=None, description="The system message to use with the planning agent."
    )


@register_function(config_type=ChatUSDConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def chat_usd_function(config: ChatUSDConfig, builder: Builder):
    if MultiAgentNetworkFunction is None:
        logger.warning("LangChain agent is not installed, skipping USD code function registration")
        return

    from omni.ai.chat_usd.bundle.chat.chat_usd_network_node import ChatUSDNetworkNode, ChatUSDSupervisorNode

    # Create and yield the function instance
    yield MultiAgentNetworkFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=ChatUSDNetworkNode,
        lc_agent_node_gen_type=ChatUSDSupervisorNode,
        multishot=True,
        hidden=True,
    )


@register_function(config_type=ChatUSDCodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def chat_usd_code_function(config: ChatUSDCodeConfig, builder: Builder):
    if LCAgentFunction is None:
        logger.warning("LangChain agent is not installed, skipping USD code function registration")
        return

    from omni.ai.langchain.agent.usd_code import USDCodeInteractiveNetworkNode

    enable_interpreter_undo_stack = carb.settings.get_settings().get(
        "/exts/omni.ai.langchain.agent.usd_code/enable_undo_stack"
    )

    yield LCAgentFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=USDCodeInteractiveNetworkNode,
        lc_agent_node_gen_type=None,
        scene_info=False,
        enable_code_interpreter=True,
        code_interpreter_hide_items=None,
        enable_code_atlas=True,
        enable_metafunctions=True,
        enable_interpreter_undo_stack=enable_interpreter_undo_stack,
        max_retries=1,
        enable_code_promoting=True,
        hidden=True,
    )


@register_function(config_type=ChatUSDSceneInfoConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def chat_usd_scene_info_function(config: ChatUSDSceneInfoConfig, builder: Builder):
    if LCAgentFunction is None:
        logger.warning("LangChain agent is not installed, skipping USD code function registration")
        return

    # TODO: Register it
    from omni.ai.langchain.agent.usd_code import SceneInfoNetworkNode

    yield LCAgentFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=SceneInfoNetworkNode,
        lc_agent_node_gen_type=None,
        enable_interpreter_undo_stack=False,
        max_retries=1,
        enable_rag=True,
        hidden=True,
    )


@register_function(config_type=ChatUSDSearchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def chat_usd_search_function(config: ChatUSDSearchConfig, builder: Builder):
    if LCAgentFunction is None:
        logger.warning("LangChain agent is not installed, skipping USD search function registration")
        return

    from omni.ai.chat_usd.bundle.search.usd_search_network_node import USDSearchNetworkNode

    # Extract optional parameters from config
    kwargs = {}
    if config.host_url is not None:
        kwargs["host_url"] = config.host_url
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    if config.username is not None:
        kwargs["username"] = config.username
    if config.url_replacements is not None:
        kwargs["url_replacements"] = config.url_replacements
    if config.search_path is not None:
        kwargs["search_path"] = config.search_path

    yield LCAgentFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=USDSearchNetworkNode,
        lc_agent_node_gen_type=None,
        hidden=True,
        **kwargs,
    )


@register_function(config_type=ChatUSDPlanningConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def chat_usd_planning_function(config: ChatUSDPlanningConfig, builder: Builder):
    if LCAgentFunction is None:
        logger.warning("LangChain agent is not installed, skipping USD code function registration")
        return

    from omni.ai.langchain.agent.planning import PlanningGenNode, PlanningNetworkNode

    system_message = config.system_message

    class ExtraPlanningGenNode(PlanningGenNode):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            if system_message:
                self.inputs.append(RunnableSystemAppend(system_message=system_message))

    yield LCAgentFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=PlanningNetworkNode,
        lc_agent_node_gen_type=ExtraPlanningGenNode,
        hidden=True,
    )
