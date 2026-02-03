## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Module for the RunnableNATNode, which integrates NAT workflows with LangChain runnables.
"""

from ..utils.nat_wrapper import NATWrapper
from nat.cli.cli_utils.config_override import load_and_override_config
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode
from typing import Dict, Any, Union, Optional


class RunnableNATNode(RunnableNode):
    """
    RunnableNATNode integrates NAT workflow system with LangChain runnables.

    This node allows NAT workflows to be used within a LangChain Runnable pipeline.
    It wraps NAT's workflow builder and runtime into the LangChain runnable interface.
    """

    nat_config: Union[Dict[str, Any], str]
    subnetwork: Optional[RunnableNetwork] = None  # Child network created during NAT workflow execution

    def _get_chat_model(self, chat_model_name, chat_model_input, invoke_input, config):
        """
        Get a NAT-compatible chat model for use in the runnable.

        Args:
            chat_model_name: The name of the chat model to use
            chat_model_input: Input for the chat model
            invoke_input: Input for the invoke method
            config: Configuration for the chat model

        Returns:
            NATWrapper: A chat model that wraps NAT workflow execution
        """
        if isinstance(self.nat_config, str):
            nat_config = load_and_override_config(self.nat_config, None)
        else:
            nat_config = self.nat_config

        return NATWrapper(nat_config, model_name=chat_model_name, parent_node=self)
