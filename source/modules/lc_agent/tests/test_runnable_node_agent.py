## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.runnable_node_agent import RunnableNodeAgent
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.runnable_utils import RunnableHumanNode, RunnableSystemAppend
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, Field
from typing import Type, List, Any, Dict, Optional
from lc_agent import get_chat_model_registry

class DummyToolInput(BaseModel):
    query: str = Field(description="The input query for the dummy tool")

class DummyTool(BaseTool):
    name: str = "DummyTool"
    description: str = "A dummy tool for testing"
    args_schema: Type[BaseModel] = DummyToolInput

    def _run(self, query: str) -> str:
        return f"Dummy result for: {query}"

    async def _arun(self, query: str) -> str:
        return f"Async dummy result for: {query}"

class MyAgent(RunnableNodeAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        system_prompt = "When the user asks for the weather you need to compare this weather with Paris"
        self.inputs.append(RunnableSystemAppend(system_message=system_prompt))
        self.tools = [DummyTool()]

class MockChatModel(BaseChatModel):
    def _generate(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> ChatResult:
        last_message = messages[-1].content if messages else ""
        if "dummy result for" in last_message.lower():
            content = '''Action:
```{
  "action": "Final Answer",
  "action_input": "This is the final answer based on the dummy tool result."
}```'''
        else:
            content = '''Action:
```{
  "action": "DummyTool",
  "action_input": {
    "query": "test input"
  }
}```'''
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock"

def setup_module(module):
    model = MockChatModel()
    model_registry = get_chat_model_registry()
    model_registry.register("MockModel", model)

def teardown_module(module):
    model_registry = get_chat_model_registry()
    model_registry.unregister("MockModel")

def test_runnable_node_agent_initialization():
    agent = MyAgent()
    assert isinstance(agent, RunnableNodeAgent)
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], DummyTool)

def test_runnable_node_agent_invoke():
    with RunnableNetwork(chat_model_name="MockModel") as network:
        RunnableHumanNode("Use the dummy tool to process: test input")
        MyAgent()

    result = network.invoke()
    
    assert isinstance(result, AIMessage)
    assert "This is the final answer" in result.content

@pytest.mark.asyncio
async def test_runnable_node_agent_ainvoke():
    with RunnableNetwork(chat_model_name="MockModel") as network:
        RunnableHumanNode("Use the dummy tool to process: test input")
        MyAgent()

    result = await network.ainvoke()
    
    assert isinstance(result, AIMessage)
    assert "This is the final answer" in result.content

@pytest.mark.asyncio
async def test_runnable_node_agent_astream():
    with RunnableNetwork(chat_model_name="MockModel") as network:
        RunnableHumanNode("Use the dummy tool to process: test input")
        MyAgent()

    result = []
    async for chunk in network.astream():
        result.append(chunk)
    
    assert len(result) > 0
    assert all(isinstance(chunk, AIMessage) for chunk in result)
    assert "This is the final answer" in result[-1].content

@pytest.mark.asyncio
async def test_runnable_node_agent_with_multiple_tools():
    class AnotherDummyTool(BaseTool):
        name: str = "AnotherDummyTool"
        description: str = "Another dummy tool for testing"
        args_schema: Type[BaseModel] = DummyToolInput

        def _run(self, query: str) -> str:
            return f"Another dummy result for: {query}"

    with RunnableNetwork(chat_model_name="MockModel") as network:
        RunnableHumanNode("Use both dummy tools to process: test input")
        agent = MyAgent()
        agent.tools.append(AnotherDummyTool())

    result = await network.ainvoke()
    
    assert isinstance(result, AIMessage)
    assert "This is the final answer" in result.content

if __name__ == "__main__":
    pytest.main(["-v", "test_runnable_node_agent.py"])