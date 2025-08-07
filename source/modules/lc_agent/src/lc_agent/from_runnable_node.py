## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .runnable_node import RunnableNode
from typing import Any, Optional, Union, List
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import BaseMessage
from pydantic import Field


class FromRunnableNode(RunnableNode):
    """A RunnableNode that wraps a Runnable and invokes it."""

    runnable: Any = Field(..., description="The runnable to wrap")
    outputs: Optional[Union[List[BaseMessage], BaseMessage]] = None

    def __init__(self, runnable, *args, **kwargs):
        # Pass runnable as a keyword argument to ensure it's properly set during validation
        kwargs["runnable"] = runnable
        super().__init__(*args, **kwargs)

    def _get_chat_model(self, chat_model_name, chat_model_input, invoke_input, config):
        """Get chat model for this node."""
        # No chat model
        return self.runnable

    async def _ainvoke_chat_model(
        self, chat_model, chat_model_input, invoke_input, config, **kwargs
    ):
        """Invoke chat model for this node."""
        # Convert input because every runnable type has its own input type
        if isinstance(self.runnable, BasePromptTemplate):
            input = invoke_input
            exclude = set()
        else:
            input = chat_model_input[:]
            exclude = {id(msg) for msg in chat_model_input}

        result = await self.runnable.ainvoke(input, config, **kwargs)

        # Convert output because every runnable type has its own output type
        if isinstance(result, ChatPromptValue):
            result = result.to_messages()
        
        # Filter out previous messages if result is a list
        if isinstance(result, list) and exclude:
            result = [msg for msg in result if id(msg) not in exclude]

        return result

    def _invoke_chat_model(
        self, chat_model, chat_model_input, invoke_input, config, **kwargs
    ):
        """Invoke chat model for this node."""
        # Convert input because every runnable type has its own input type
        if isinstance(self.runnable, BasePromptTemplate):
            input = invoke_input
            exclude = set()
        else:
            input = chat_model_input[:]
            exclude = {id(msg) for msg in chat_model_input}

        result = self.runnable.invoke(input, config, **kwargs)

        # Convert output because every runnable type has its own output type
        if isinstance(result, ChatPromptValue):
            result = result.to_messages()
        
        # Filter out previous messages if result is a list
        if isinstance(result, list) and exclude:
            result = [msg for msg in result if id(msg) not in exclude]

        return result
