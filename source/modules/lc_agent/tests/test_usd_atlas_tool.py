## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.code_atlas.usd_atlas_tool import USDAtlasTool
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.runnable_utils import RunnableHumanNode
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Any
from lc_agent import get_chat_model_registry
import json

class MockUSDChatModel(BaseChatModel):
    def _generate(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> ChatResult:
        content = json.dumps({
            "action": "USDAtlasTool",
            "action_input": {
                "lookup_type": "CLASS",
                "lookup_name": "UsdStage",
                "methods": True,
                "full_source": False,
                "docs": True
            }
        })
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock_usd"

def setup_module(module):
    model = MockUSDChatModel()
    get_chat_model_registry().register("MockUSDModel", model)

def teardown_module(module):
    get_chat_model_registry().unregister("MockUSDModel")

def test_usd_atlas_tool_initialization():
    tool = USDAtlasTool()
    assert isinstance(tool, USDAtlasTool)
    assert tool.name == "USDAtlasTool"
    assert "USD API" in tool.description

def test_usd_atlas_tool_invoke():
    with RunnableNetwork(chat_model_name="MockUSDModel") as network:
        RunnableHumanNode("Use the USDAtlasTool to look up the UsdStage class")
        USDAtlasTool()
    result = network.invoke()
    assert isinstance(result, HumanMessage)
    assert "Use the USDAtlasTool to look up the UsdStage class" in result.content

@pytest.mark.asyncio
async def test_usd_atlas_tool_ainvoke():
    with RunnableNetwork(chat_model_name="MockUSDModel") as network:
        RunnableHumanNode("Use the USDAtlasTool to find information about UsdPrim")
        USDAtlasTool()
    result = await network.ainvoke()
    assert isinstance(result, HumanMessage)
    assert "Use the USDAtlasTool to find information about UsdPrim" in result.content

def test_usd_atlas_tool_run():
    tool = USDAtlasTool()
    result = tool._run(lookup_type="MODULE", lookup_name="UsdGeom")
    assert isinstance(result, str)
    assert "UsdGeom" in result

@pytest.mark.asyncio
async def test_usd_atlas_tool_arun():
    tool = USDAtlasTool()
    result = await tool._arun(lookup_type="CLASS", lookup_name="UsdStage")
    assert isinstance(result, str)
    assert "UsdStage" in result

def test_usd_atlas_tool_cache_loading():
    tool = USDAtlasTool()
    assert tool.cache is not None
    assert len(tool.cache._modules) > 0

def test_usd_atlas_tool_topic_loading():
    tool = USDAtlasTool()
    assert tool.topic is not None
    assert len(tool.topic.topics) > 0

if __name__ == "__main__":
    pytest.main(["-v", "test_usd_atlas_tool.py"])