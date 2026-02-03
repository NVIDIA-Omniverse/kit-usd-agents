## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from lc_agent_nat.utils.conversion import convert_langchain_to_nat_messages


def test_convert_simple_human_message():
    messages = [HumanMessage(content="Hello")]
    result = convert_langchain_to_nat_messages(messages)

    assert len(result.messages) == 1
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "Hello"


def test_convert_simple_ai_message():
    messages = [AIMessage(content="Hi there")]
    result = convert_langchain_to_nat_messages(messages)

    assert len(result.messages) == 1
    assert result.messages[0].role == "assistant"
    assert result.messages[0].content == "Hi there"


def test_convert_simple_system_message():
    messages = [SystemMessage(content="You are helpful")]
    result = convert_langchain_to_nat_messages(messages)

    assert len(result.messages) == 1
    assert result.messages[0].role == "system"
    assert result.messages[0].content == "You are helpful"


def test_convert_conversation():
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="4"),
        HumanMessage(content="Thanks!"),
    ]
    result = convert_langchain_to_nat_messages(messages)

    assert len(result.messages) == 4
    assert result.messages[0].role == "system"
    assert result.messages[1].role == "user"
    assert result.messages[2].role == "assistant"
    assert result.messages[3].role == "user"


def test_convert_empty_messages():
    messages = []
    result = convert_langchain_to_nat_messages(messages)
    assert len(result.messages) == 0

