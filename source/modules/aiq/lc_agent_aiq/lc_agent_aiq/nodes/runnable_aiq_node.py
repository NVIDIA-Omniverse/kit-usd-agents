## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Module for the RunnableAIQNode, which integrates AIQ workflows with LangChain runnables.
"""

from ..utils.aiq_wrapper import AIQWrapper
from aiq.cli.cli_utils.config_override import load_and_override_config
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode
from typing import Dict, Any, Union, Optional


class RunnableAIQNode(RunnableNode):
    """
    RunnableAIQNode integrates AIQ workflow system with LangChain runnables.

    This node allows AIQ workflows to be used within a LangChain Runnable pipeline.
    It wraps AIQ's workflow builder and runtime into the LangChain runnable interface.
    """

    aiq_config: Union[Dict[str, Any], str]
    subnetwork: Optional[RunnableNetwork] = None  # Child network created during AIQ workflow execution

    def _get_chat_model(self, chat_model_name, chat_model_input, invoke_input, config):
        """
        Get an AIQ-compatible chat model for use in the runnable.

        Args:
            chat_model_name: The name of the chat model to use
            chat_model_input: Input for the chat model
            invoke_input: Input for the invoke method
            config: Configuration for the chat model

        Returns:
            AIQWrapper: A chat model that wraps AIQ workflow execution
        """
        if isinstance(self.aiq_config, str):
            aiq_config = load_and_override_config(self.aiq_config, None)
        else:
            aiq_config = self.aiq_config

        return AIQWrapper(aiq_config, model_name=chat_model_name, parent_node=self)
