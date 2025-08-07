## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.messages import convert_to_openai_messages
from aiq.data_models.api_server import AIQChatRequest, Message


def convert_langchain_to_aiq_messages(lc_messages: List[BaseMessage]) -> AIQChatRequest:
    """Convert LangChain messages to AIQ chat request format.
    
    This function takes a list of LangChain BaseMessage objects and converts them
    to the AIQ message format required for AIQ chat requests.
    
    Args:
        lc_messages: List of LangChain BaseMessage objects
        
    Returns:
        AIQChatRequest: A properly formatted AIQ chat request object
    """
    # First convert LangChain messages to OpenAI format
    openai_messages = convert_to_openai_messages(lc_messages)
    aiq_messages = []

    # Convert each OpenAI message to AIQ message format
    for i, openai_msg in enumerate(openai_messages):
        aiq_messages.append(Message(content=openai_msg["content"], role=openai_msg["role"]))

    # Return a properly formatted AIQ chat request
    return AIQChatRequest(messages=aiq_messages) 