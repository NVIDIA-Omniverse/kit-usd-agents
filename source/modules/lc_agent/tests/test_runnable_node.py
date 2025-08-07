## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.runnable_node import RunnableNode, AINodeMessageChunk, _is_message
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.network_modifier import NetworkModifier
from lc_agent.chat_model_registry import get_chat_model_registry
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.runnables import RunnableConfig
from typing import List, Dict, Any, AsyncIterator
import asyncio
from langchain_core.runnables import Runnable
from lc_agent.from_runnable_node import FromRunnableNode

class DummyChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _generate(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Dummy response"))])

    async def _agenerate(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Async dummy response"))])

    async def _astream(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="Streaming ", type="AIMessageChunk"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="dummy ", type="AIMessageChunk"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="response", type="AIMessageChunk"))

class TestRunnableNode(RunnableNode):
    def _get_chat_model(self, chat_model_name, chat_model_input, invoke_input, config):
        return DummyChatModel()

class DummyNode(TestRunnableNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invoked = False

    def invoke(self, input: Dict[str, Any] = {}, config=None, **kwargs):
        if self.invoked:
            return self.outputs
        self.outputs = AIMessage(content="Dummy response")
        self.invoked = True
        return self.outputs

    async def ainvoke(self, input: Dict[str, Any] = {}, config=None, **kwargs):
        if self.invoked:
            return self.outputs
        self.outputs = AIMessage(content="Async dummy response")
        self.invoked = True
        return self.outputs

    async def astream(self, input: Dict[str, Any] = {}, config=None, **kwargs) -> AsyncIterator[AINodeMessageChunk]:
        if not self.invoked:
            self.outputs = AIMessage(content="Streaming dummy response")
            self.invoked = True
            yield AINodeMessageChunk(content="Streaming ", node=self)
            yield AINodeMessageChunk(content="dummy ", node=self)
            yield AINodeMessageChunk(content="response", node=self)
        else:
            yield AINodeMessageChunk(content=self.outputs.content, node=self)

@pytest.fixture
def dummy_chat_model():
    return DummyChatModel()

def test_runnable_node_creation():
    node = TestRunnableNode()
    assert isinstance(node, RunnableNode)
    assert node.parents == []
    assert node.inputs == []
    assert node.outputs is None
    assert 'uuid' in node.metadata
    assert len(node.metadata) == 1
    assert not node.invoked

def test_runnable_node_invoke():
    node = TestRunnableNode()
    result = node.invoke()
    assert isinstance(result, AIMessage)
    assert result.content == "Dummy response"
    assert node.invoked

@pytest.mark.asyncio
async def test_runnable_node_ainvoke():
    node = TestRunnableNode()
    result = await node.ainvoke()
    assert isinstance(result, AIMessage)
    assert result.content == "Async dummy response"
    assert node.invoked

@pytest.mark.asyncio
async def test_runnable_node_astream():
    node = TestRunnableNode()
    chunks = []
    async for chunk in node.astream():
        chunks.append(chunk)
    assert len(chunks) == 3
    assert all(isinstance(chunk, AINodeMessageChunk) for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == "Streaming dummy response"
    assert node.invoked

def test_runnable_node_add_parent():
    parent = DummyNode()
    child = DummyNode()
    child._add_parent(parent)
    assert parent in child.parents

def test_runnable_node_clear_parents():
    parent1 = DummyNode()
    parent2 = DummyNode()
    child = DummyNode()
    child._add_parent(parent1)
    child._add_parent(parent2)
    child._clear_parents()
    assert child.parents == []

def test_runnable_node_rshift():
    node1 = DummyNode()
    node2 = DummyNode()
    result = node1 >> node2
    assert result == node2
    assert node1 in node2.parents

def test_runnable_node_lshift():
    node1 = DummyNode()
    node2 = DummyNode()
    result = node2 << node1
    assert result == node1
    assert node1 in node2.parents

def test_runnable_node_rrshift():
    node1 = DummyNode()
    node2 = DummyNode()
    result = None >> node1 >> node2
    assert result == node2
    assert node1 in node2.parents
    assert node1.parents == []

def test_runnable_node_combine_inputs():
    node = TestRunnableNode()
    parents_result = [
        SystemMessage(content="System message"),
        SystemMessage(content="Other system message"),
        HumanMessage(content="Human message"),
    ]
    result = node._combine_inputs({}, None, parents_result)
    assert len(result) == 2
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert "System message" in result[0].content
    assert "Other system message" in result[0].content
    assert result[1].content == "Human message"

def test_runnable_node_combine_inputs_with_chat_prompt_value():
    node = TestRunnableNode()
    parents_result = [HumanMessage(content="Human message")]
    input_result = ChatPromptValue(messages=[AIMessage(content="AI message")])
    result = node._combine_inputs({"input": input_result}, None, parents_result)
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Human message"

def test_runnable_node_in_network():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
    
    assert node1 in network.nodes
    assert node2 in network.nodes
    assert network.get_parents(node2) == [node1]

def test_runnable_node_metadata():
    node = TestRunnableNode()
    node.metadata["key"] = "value"
    assert node.metadata["key"] == "value"

def test_runnable_node_chat_model_name():
    node = TestRunnableNode(chat_model_name="test_model")
    assert node.chat_model_name == "test_model"

def test_runnable_node_verbose():
    node = TestRunnableNode(verbose=True)
    assert node.verbose

class DummyModifier(NetworkModifier):
    def on_begin_invoke(self, network):
        network.metadata["modifier_called"] = True

def test_runnable_node_with_modifier():
    with RunnableNetwork() as network:
        node = DummyNode()
    
    modifier = DummyModifier()
    network.add_modifier(modifier)
    
    network.invoke()
    assert network.metadata.get("modifier_called") == True

@pytest.mark.asyncio
async def test_runnable_node_process_parents():
    class TestNode(TestRunnableNode):
        async def ainvoke(self, input, config=None, **kwargs):
            return AIMessage(content="Test")

    with RunnableNetwork() as network:
        node1 = TestNode()
        node2 = TestNode()
        node3 = TestRunnableNode()

    result = await node3._aprocess_parents({}, None)
    assert len(result) == 2
    assert all(isinstance(r, AIMessage) for r in result)
    assert [r.content for r in result] == ["Test", "Test"]

class DummyRunnable(Runnable):
    def invoke(self, input, config=None):
        return input

def test_lshift_with_runnable():
    node = TestRunnableNode()
    runnable = DummyRunnable()
    result = node << runnable
    assert isinstance(node.parents[0], FromRunnableNode)

def test_rshift_with_runnable():
    node = TestRunnableNode()
    runnable = DummyRunnable()
    result = node >> runnable
    assert isinstance(result, FromRunnableNode)
    assert result.parents[0] == node

def test_rrshift_with_runnable():
    runnable = DummyRunnable()
    node = TestRunnableNode()
    result = runnable >> node
    assert isinstance(result, TestRunnableNode)
    assert isinstance(result.parents[0], FromRunnableNode)

def test_add_parent_existing():
    node = TestRunnableNode()
    parent = TestRunnableNode()
    node._add_parent(parent)
    node._add_parent(parent)  # Adding the same parent again
    assert len(node.parents) == 1
    assert node.parents[0] == parent

def test_add_parent_with_index():
    node = TestRunnableNode()
    parent1 = TestRunnableNode()
    parent2 = TestRunnableNode()
    node._add_parent(parent1)
    node._add_parent(parent2, parent_index=0)
    assert node.parents == [parent2, parent1]

@pytest.mark.asyncio
async def test_astream_with_ainodemessagechunk():
    class TestStreamNode(TestRunnableNode):
        async def _astream_chat_model(self, *args, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="Test"))
            yield ChatGenerationChunk(message=AIMessageChunk(content="NodeChunk"))

        async def astream(self, input: Dict[str, Any] = {}, config=None, **kwargs) -> AsyncIterator[AINodeMessageChunk]:
            async for item in self._astream_chat_model():
                if isinstance(item, ChatGenerationChunk):
                    yield AINodeMessageChunk(content=item.message.content, node=self)
                else:
                    yield item

    node = TestStreamNode()
    chunks = []
    async for chunk in node.astream():
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert all(isinstance(chunk, AINodeMessageChunk) for chunk in chunks)
    assert chunks[0].content == "Test"
    assert chunks[1].content == "NodeChunk"

def test_is_message_with_various_types():
    assert _is_message(HumanMessage(content="Test"))
    assert _is_message(AIMessage(content="Test"))
    assert _is_message(SystemMessage(content="Test"))
    assert _is_message(ChatPromptValue(messages=[HumanMessage(content="Test")]))
    assert _is_message({"role": "user", "content": "Test"})
    assert _is_message("Test")
    assert _is_message(("user", "Test"))
    assert not _is_message(123)

def test_repr_with_different_states():
    node = TestRunnableNode(name="TestNode")
    assert "TestNode" in repr(node)
    assert "(no outputs)" in repr(node)

    node.outputs = AIMessage(content="Test output")
    repr_string = repr(node)
    assert "TestNode" in repr_string
    assert "Test output" in repr_string

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])