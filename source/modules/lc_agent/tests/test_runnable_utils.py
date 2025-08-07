## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.runnable_utils import (
    RunnableAppend,
    RunnableSystemAppend,
    RunnableHumanNode,
    RunnableAINode,
    RunnableHumanImageNode,
    RunnableAIImageNode
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from lc_agent.runnable_node import RunnableNode
from tempfile import NamedTemporaryFile
import io
import base64
import os
from unittest.mock import patch, Mock

def test_runnable_append_with_chat_prompt_value():
    append = RunnableAppend(message=HumanMessage(content="Hello"))
    input_value = ChatPromptValue(messages=[SystemMessage(content="System message")])
    result = append._execute(input_value)
    
    assert isinstance(result, ChatPromptValue)
    assert len(result.messages) == 2
    assert isinstance(result.messages[0], SystemMessage)
    assert isinstance(result.messages[1], HumanMessage)
    assert result.messages[1].content == "Hello"

def test_runnable_append_with_list():
    append = RunnableAppend(message=AIMessage(content="AI response"))
    input_value = [HumanMessage(content="Human message")]
    result = append._execute(input_value)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    assert result[1].content == "AI response"

def test_runnable_append_with_dict():
    append = RunnableAppend(message=SystemMessage(content="System instruction"))
    input_value = {"role": "user", "content": "User input"}
    result = append._execute(input_value)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == input_value
    assert isinstance(result[1], SystemMessage)
    assert result[1].content == "System instruction"

def test_runnable_system_append():
    system_append = RunnableSystemAppend(system_message="System instruction")
    input_value = [HumanMessage(content="Human message")]
    result = system_append._execute(input_value)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], SystemMessage)
    assert result[1].content == "System instruction"

def test_runnable_human_node():
    human_node = RunnableHumanNode(human_message="Human question")
    
    assert isinstance(human_node, RunnableNode)
    assert isinstance(human_node.outputs, HumanMessage)
    assert human_node.outputs.content == "Human question"

def test_runnable_ai_node():
    ai_node = RunnableAINode(ai_message="AI answer")
    
    assert isinstance(ai_node, RunnableNode)
    assert isinstance(ai_node.outputs, AIMessage)
    assert ai_node.outputs.content == "AI answer"

def test_runnable_image_node():
    # Hardcoded 4x4 PNG image data from empty4x4.png
    PNG_DATA = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x04\x00\x00\x00\x04\x08\x02\x00\x00\x00&\x93\t)\x00\x00\x01\x85iCCPICC profile\x00\x00(\x91}\x91;H\xc3@\x1c\xc6\xbf\xa6\x8a\x0f\xea\x03\xec \xe2\x10\xa4:Y\x10\x15q\x94*\x16\xc1Bi+\xb4\xea`r\xe9\x0b\x9a4$).\x8e\x82k\xc1\xc1\xc7b\xd5\xc1\xc5YW\x07WA\x10|\x80\xb8\x0bN\x8a.R\xe2\xff\x92B\x8b\x18\x0f\x8e\xfb\xf1\xdd}\x1fw\xdf\x01B\xad\xc4T\xb3m\x02P5\xcbHD#b:\xb3*v\xbc\xc2\x8f^\xf4\xa3\x0b#\x123\xf5Xr1\x05\xcf\xf1u\x0f\x1f_\xef\xc2<\xcb\xfb\xdc\x9f\xa3G\xc9\x9a\x0c\xf0\x89\xc4sL7,\xe2\r\xe2\x99MK\xe7\xbcO\x1cd\x05I!>'\x1e7\xe8\x82\xc4\x8f\\\x97]~\xe3\x9cwX\xe0\x99A#\x95\x98'\x0e\x12\x8b\xf9\x16\x96[\x98\x15\x0c\x95x\x9a8\xa4\xa8\x1a\xe5\x0bi\x97\x15\xce[\x9c\xd5R\x855\xee\xc9_\x18\xc8j+I\xae\xd3\x1cF\x14K\x88!\x0e\x112*(\xa2\x04\x0baZ5RL$h?\xe2\xe1\x1fr\xfcqr\xc9\xe4*\x82\x91c\x01e\xa8\x90\x1c?\xf8\x1f\xfc\xee\xd6\xccMM\xbaI\x81\x08\xd0\xfeb\xdb\x1f\xa3@\xc7.P\xaf\xda\xf6\xf7\xb1m\xd7O\x00\xff3p\xa55\xfd\xe5\x1a0\xfbIz\xb5\xa9\x85\x8e\x80\xbem\xe0\xe2\xba\xa9\xc9{\xc0\xe5\x0e0\xf8\xa4K\x86\xe4H~\x9aB.\x07\xbc\x9f\xd17e\x80\x81[\xa0{\xcd\xed\xad\xb1\x8f\xd3\x07 E]-\xdf\x00\x07\x87\xc0X\x9e\xb2\xd7=\xde\xdd\xd9\xda\xdb\xbfg\x1a\xfd\xfd\x00T\xadr\x9b\x8eT*\xd4\x00\x00\x00\tpHYs\x00\x00.#\x00\x00.#\x01x\xa5?v\x00\x00\x00\x07tIME\x07\xe9\x03\x0e\x10\n\x19\xd8B\x0e3\x00\x00\x00\x19tEXtComment\x00Created with GIMPW\x81\x0e\x17\x00\x00\x00\x0cIDAT\x08\xd7c` \x1d\x00\x00\x004\x00\x01\xa7}\xb1\x9c\x00\x00\x00\x00IEND\xaeB`\x82"
    
    # Write the PNG data to a temporary file
    file = NamedTemporaryFile(mode="w+b", suffix=".png", delete=False)
    file.write(PNG_DATA)
    file.close()

    with patch("httpx.get") as mock_get:
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.content = bytes()
        mock_get.return_value = mock_response
        human_image_node = RunnableHumanImageNode(human_message="Human question", images=[file.name, "https://www.foo.com/bar.png"])
        mock_get.assert_called_once_with("https://www.foo.com/bar.png")
        mock_response.raise_for_status.assert_called_once()
        assert isinstance(human_image_node, RunnableNode)
        assert isinstance(human_image_node.outputs, HumanMessage)
        assert len(human_image_node.outputs.content) == 3
        assert human_image_node.outputs.content[0]["type"] == "text"
        assert human_image_node.outputs.content[1]["type"] == "image_url"
        assert human_image_node.outputs.content[2]["type"] == "image_url"

    with io.BytesIO() as buffer:
        # Use the same PNG data for the base64 encoding
        buffer.write(PNG_DATA)
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        ai_image_node = RunnableAIImageNode(ai_message="AI Message", images=[f"data:image/png;base64,{image_data}"])
        assert isinstance(ai_image_node.outputs, AIMessage)

    os.remove(file.name)

@pytest.mark.asyncio
async def test_runnable_append_invoke():
    append = RunnableAppend(message=HumanMessage(content="Hello"))
    input_value = ChatPromptValue(messages=[SystemMessage(content="System message")])
    result = await append.ainvoke(input_value)
    
    assert isinstance(result, ChatPromptValue)
    assert len(result.messages) == 2
    assert isinstance(result.messages[0], SystemMessage)
    assert isinstance(result.messages[1], HumanMessage)
    assert result.messages[1].content == "Hello"

@pytest.mark.asyncio
async def test_runnable_system_append_invoke():
    system_append = RunnableSystemAppend(system_message="System instruction")
    input_value = [HumanMessage(content="Human message")]
    result = await system_append.ainvoke(input_value)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], SystemMessage)
    assert result[1].content == "System instruction"

if __name__ == "__main__":
    pytest.main(["-v", "test_runnable_utils.py"])