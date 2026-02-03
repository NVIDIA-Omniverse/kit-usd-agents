## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
MultiAgentNetworkFunction implementation.

This module provides a specialized LCAgentFunction implementation for multi-agent networks
that handles dynamic registration and cleanup of agent functions.
"""

from typing import Optional

from nat.builder.builder import Builder
from nat.builder.function import Function, FunctionGroup
from nat.data_models.api_server import AIQChatRequest

from lc_agent import get_node_factory
from .lc_agent_function import LCAgentFunction
from ..nodes.function_runnable_node import FunctionRunnableNode, ValidationErrorHandlingFunction


class MultiAgentNetworkFunction(LCAgentFunction):
    async def pre_invoke(self, value: AIQChatRequest) -> None:
        # Get the tool names from the config
        function_names = self.config.tool_names
        self.filtered_function_names = []

        # Register each function or verify it exists
        for name in function_names:
            # Try to get it as a function group first
            try:
                function_group = await self.builder.get_function_group(name)
                # It's a function group - expand it to individual functions
                accessible_functions = await function_group.get_accessible_functions()
                for fn_name, fn_instance in accessible_functions.items():
                    if isinstance(fn_instance, LCAgentFunction):
                        await fn_instance.pre_invoke(value)
                        self.filtered_function_names.append(fn_name)
                    elif isinstance(fn_instance, Function):
                        # Wrap the function to handle validation errors gracefully
                        wrapped_function = ValidationErrorHandlingFunction(fn_instance)
                        get_node_factory().register(
                            FunctionRunnableNode,
                            name=fn_name,
                            function=wrapped_function,
                            description=fn_instance.description,
                        )
                        self.filtered_function_names.append(fn_name)
            except (ValueError, KeyError):
                # Not a function group, try as a regular function
                function = await self.builder.get_function(name)
                if isinstance(function, LCAgentFunction):
                    # It will register the function with the node factory
                    await function.pre_invoke(value)
                    self.filtered_function_names.append(name)
                elif isinstance(function, Function):
                    # Wrap the function to handle validation errors gracefully
                    wrapped_function = ValidationErrorHandlingFunction(function)
                    get_node_factory().register(
                        FunctionRunnableNode,
                        name=name,
                        function=wrapped_function,
                        description=function.description,
                    )
                    self.filtered_function_names.append(name)

        self.lc_agent_node_kwargs["route_nodes"] = self.filtered_function_names

        await super().pre_invoke(value)

    async def post_invoke(self, value: AIQChatRequest, success: bool = True, error: Optional[Exception] = None) -> None:
        # Unregister all the functions we registered (including those from function groups)
        for name in self.filtered_function_names:
            # Try to get it as a regular function
            try:
                function = await self.builder.get_function(name)
                if isinstance(function, LCAgentFunction):
                    # It will unregister the function from the node factory
                    await function.post_invoke(value, success, error)
                elif isinstance(function, Function):
                    get_node_factory().unregister(name)
            except (ValueError, KeyError):
                # Might be a function from a function group, just unregister by name
                get_node_factory().unregister(name)

        await super().post_invoke(value, success, error)
