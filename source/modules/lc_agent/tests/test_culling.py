## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.utils.culling import (
    _get_message_role,
    _get_message_tokens,
    _cull_message,
    _cull_messages
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class MockTokenizer:
    def encode(self, text):
        return [0] * len(text.split())  # Simulate tokenization by word count

    def decode(self, tokens):
        return " ".join(["word"] * len(tokens))

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

def test_get_message_role():
    assert _get_message_role(AIMessage(content="AI")) == "assistant"
    assert _get_message_role(HumanMessage(content="Human")) == "user"
    assert _get_message_role(SystemMessage(content="System")) == "system"
    assert _get_message_role("Not a message") == "none"

def test_get_message_tokens(mock_tokenizer):
    ai_message = AIMessage(content="This is an AI message")
    human_message = HumanMessage(content="This is a human message")
    system_message = SystemMessage(content="This is a system message")

    assert _get_message_tokens(ai_message, mock_tokenizer) == 12  # 5 words + 3 (role tokens) + 4 (additional tokens)
    assert _get_message_tokens(human_message, mock_tokenizer) == 12
    assert _get_message_tokens(system_message, mock_tokenizer) == 12

def test_cull_message(mock_tokenizer):
    long_message = HumanMessage(content="This is a very long message that needs to be culled")
    culled_message = _cull_message(long_message, mock_tokenizer, max_tokens=10)

    assert isinstance(culled_message, HumanMessage)
    assert len(culled_message.content.split()) == 3  # 10 - 3 (role tokens) - 4 (additional tokens) = 3

    # Test culling from the end
    culled_message_end = _cull_message(long_message, mock_tokenizer, max_tokens=10, remove_from_start=False)
    assert len(culled_message_end.content.split()) == 3

def test_cull_messages(mock_tokenizer):
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
        HumanMessage(content="Another human message")
    ]

    culled_messages = _cull_messages(messages, max_tokens=30, tokenizer=mock_tokenizer)

    assert len(culled_messages) <= 4
    assert isinstance(culled_messages[-1], HumanMessage)  # Last message should be preserved
    assert isinstance(culled_messages[0], SystemMessage)  # System message should be preserved

def test_cull_messages_no_culling_needed(mock_tokenizer):
    messages = [
        SystemMessage(content="Short system message"),
        HumanMessage(content="Short human message"),
    ]

    culled_messages = _cull_messages(messages, max_tokens=100, tokenizer=mock_tokenizer)

    assert len(culled_messages) == 2
    assert culled_messages == messages

def test_cull_messages_no_max_tokens(mock_tokenizer):
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]

    culled_messages = _cull_messages(messages, max_tokens=None, tokenizer=mock_tokenizer)

    assert culled_messages == messages

def test_cull_messages_no_tokenizer(mock_tokenizer):
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
    ]

    culled_messages = _cull_messages(messages, max_tokens=10, tokenizer=None)

    assert culled_messages == messages

if __name__ == "__main__":
    pytest.main(["-v", "test_culling.py"])